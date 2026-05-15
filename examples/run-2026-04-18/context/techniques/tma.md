# Tensor Memory Accelerator (TMA)

## What TMA is

The **Tensor Memory Accelerator (TMA)** is a hardware feature introduced in
NVIDIA Hopper (sm_90a) and extended on Blackwell (sm_100) that performs
**asynchronous** memory copies between global memory (GMEM) and shared memory
(SMEM) for the thread blocks (CTAs) of a kernel.

TMA replaces pointer-arithmetic loads with **tensor descriptors** that encode
the tensor's address, strides, shape, and tile shape. The hardware then
streams blocks of data in or out, overlapping the copy with computation,
freeing warps from manual load/store issue, and reducing register pressure.

Benefits on B200:

- Hardware-accelerated async memory transfers overlap with MMA.
- Better coalescing and automatic swizzle for bank-conflict avoidance.
- Simpler kernel code (no manual `offs_* + stride_* × pid_*` arithmetic).
- Pairs cleanly with multi-stage pipelines and warp specialization.

TMA descriptors can be built either **on the host** (simpler, older path) or
**on the device** (more flexible, newer Triton path). Both are in active use;
pick the one that fits your kernel's constraints. Details below.

---

## On-host TMA

### How it works

A TMA descriptor is allocated and initialized in **host memory**, then copied
by value to GMEM and passed to the kernel as a parameter. The kernel uses the
descriptor's `load`/`store` methods for every block-level transfer.

### How to integrate into a Triton program

In the host program, import `TensorDescriptor` and allocate one descriptor per
tensor whose access you want to accelerate:

```python
from triton.tools.tensor_descriptor import TensorDescriptor

a_desc = TensorDescriptor(
    a,                                  # the pointer to the tensor
    a.shape,                            # the shape of the tensor
    a.stride(),                         # the stride of the tensor
    [BLOCK_SIZE_M, BLOCK_SIZE_K],       # the block size of each TMA load/store
)
b_desc = TensorDescriptor(b, b.shape, b.stride(), [BLOCK_SIZE_K, BLOCK_SIZE_N])
c_desc = TensorDescriptor(c, c.shape, c.stride(), [BLOCK_SIZE_M, BLOCK_SIZE_N])

kernel[grid](a_desc, b_desc, c_desc, ...)
```

In the kernel, replace ranged-pointer loads with descriptor-based block
transfers:

```python
@triton.jit
def matmul_kernel(a_desc, b_desc, c_desc, M, N, K,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr):
    pid   = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid %  tl.cdiv(N, BLOCK_SIZE_N)

    a = a_desc.load([pid_m * BLOCK_SIZE_M, 0])            # offset coordinates
    b = b_desc.load([0, pid_n * BLOCK_SIZE_N])
    acc = tl.dot(a, b)
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], acc)
```

### Pros/cons

- **+** Descriptor lives for the kernel's full lifetime; no per-CTA init cost.
- **+** Simplest path;
- **−** Descriptor layout is fixed at launch. If you need a different block
  shape per CTA (e.g. ragged grouped GEMM with variable M), you pay host round-trips
  or need multiple descriptors.

---

## On-device TMA

### How it works

A chunk of GMEM is reserved up front (via a custom Triton allocator). Each
CTA builds its **own** TMA descriptor inside the kernel, writing it into its
slice of the reserved memory. This lets the tile shape, base pointer, or
stride vary per program.

### How to integrate into a Triton program

In the host program, register a global-memory allocator and import `Optional`:

```python
from typing import Optional
import torch, triton

def alloc_fn(size: int, alignment: int, stream: Optional[int]):
    return torch.empty(size, device="cuda", dtype=torch.int8)

triton.set_allocator(alloc_fn)
```

In the kernel, build the descriptor inside the program using
`tl.make_tensor_descriptor` and then load/store as usual:

```python
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr):

    a_desc = tl.make_tensor_descriptor(
        a_ptr,                                  # the pointer to the tensor
        shape=[M, K],                           # the shape of the tensor
        strides=[stride_am, stride_ak],         # the stride of the tensor
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],  # block size of each TMA op
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N])
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, N], strides=[stride_cm, stride_cn],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])

    pid   = tl.program_id(0)
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid %  tl.cdiv(N, BLOCK_SIZE_N)

    a = a_desc.load([pid_m * BLOCK_SIZE_M, 0])
    b = b_desc.load([0, pid_n * BLOCK_SIZE_N])
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], tl.dot(a, b))
```

### Pros/cons

- **+** Per-CTA block shape / base pointer. Fits any kernel where descriptors differ across programs.
- **+** No host-side round-trip for shape changes.
- **−** Descriptor construction is per-program work — amortize it across many
  K-steps or it is pure overhead.
- **−** Requires the `triton.set_allocator` dance and Optional import; easy
  to miss when porting a kernel from the on-host style.

---

## Key difference vs traditional ranged-pointer loads

**Traditional**

```python
offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
a_ptrs  = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
a = tl.load(a_ptrs, mask=...)
```

**TMA**

```python
a = a_desc.load([pid_m * BLOCK_SIZE_M, k_offset])
```

TMA simplifies memory access patterns and delegates coalescing/swizzle to the
hardware.

## Which path to pick on B200

| Situation | Prefer |
|-----------|--------|
| Static shapes, same tile for every CTA (plain GEMM) | On-host |
| Variable per-CTA tile (grouped GEMM, gather, variable M) | On-device |
| Need to pass descriptors through helper device functions | On-device |
| Minimal change from an existing pointer-arithmetic kernel | On-host |


## Attribution

Adapted from `KernelAgent/kernel_perf_agent/kernel_opt/database/docs/tma.md`,
`on_host_tma.py`, and `on_device_tma.py` (Apache-2.0, Meta Platforms Inc.).
