# GPU Kernel Optimization Techniques


---

## 1. Tile Scheduling & Persistent Kernels

### Standard vs. Persistent Grid

**Standard:** Grid = total tiles. Each SM gets one tile, then the thread block terminates.
- Overhead: ~10–20 µs scheduler latency per kernel launch
- No inter-tile data reuse in registers

**Persistent:** Grid = `NUM_SMS` (or a small multiple). Each SM loops over many tiles.
```python
# Triton
start_pid = tl.program_id(0)
for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
    pid_m, pid_n = swizzle(tile_id, ...)
    # load A-tile and B-tile, compute, store
```
- Benefits: amortizes launch overhead, keeps L2 warm across tiles, enables pipeline decoupling
- Required for TMA + warp specialization (must loop to keep async engines busy)

### 2D Swizzle for L2 Reuse

Row-major tile ordering wastes L2: each new M-row evicts all of B. Swizzling groups
`GROUP_M` consecutive M-tiles under the same N-column so B-tiles stay in L2.

```python
# pid → (pid_m, pid_n) with GROUP_M grouping
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_M * num_pid_n
group_id   = pid // num_pid_in_group
group_size = min(num_pid_m - group_id * GROUP_M, GROUP_M)
pid_m = group_id * GROUP_M + (pid % group_size)
pid_n = (pid % num_pid_in_group) // group_size
```

Typical `GROUP_M = 8`. Effective when N-tiles (B rows) fit in L2 for one group pass.

---

## 2. TMA (Tensor Memory Accelerator)

TMA offloads addressing from CUDA cores to a dedicated hardware unit. Enables async 2D bulk copies with hardware-managed barriers.

### Descriptor Creation

```python
# Triton on-device
a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K])
```

---

## 3. FP8 Block-Scaled GEMM

### Scale Factor Layout

**SM90 (H100):** `float32` scales, one per 128 channels (BLOCK_K=128).
Layout: `[M//128, K//128]` or `[K//128, M//128]` transposed.

**SM100 (B200):** `UE8M0` packed 4-per-int32, one per 32 or 128 channels.
```cpp
// Pack 4 FP32 exponents into one int32
uint8_t ue8m0 = (fp32_bits >> 23) & 0xFF;  // extract exponent
packed = (ue8m0_3 << 24) | (ue8m0_2 << 16) | (ue8m0_1 << 8) | ue8m0_0;
```

### Block-Scaled dot in Triton (B200)

```python
# NVFP4/MXFP8 approach
c = tl.dot_scaled(a_tile, a_scale, "e2m1",   # FP4 data + scale
                  b_tile, b_scale, "e4m3",   # FP8 data + scale
                  c_acc, out_dtype=tl.float32)
```
Hardware handles scale broadcast internally. Scale preshuffling is required:
```python
# Blackwell 5D pack: (M//32//4, K//VEC//4, 32, 4, 4)
scale_reshaped = scale.reshape(rep_m, rep_k, 32, 4, 4).trans(0,3,2,1,4).reshape(BLOCK_M, BLOCK_K//VEC)
```

---

## 4. Warp Specialization (B200 / Hopper)

On Hopper+ (SM90+), warps within a block can be split into producer and consumer groups that run asynchronously:

```python
# Triton: warp_specialize=True in tl.range
for tile_id in tl.range(start, end, step, warp_specialize=True):
    # producer warps: issue TMA loads
    tl.async_copy(a_desc, smem_a, ...)
    # consumer warps: run GEMM on previous tile
    c += tl.dot(smem_a_prev, smem_b_prev)
```

**Benefits:**
- Producer stalls on TMA don't block consumer warps
- Effective pipeline depth = `min(producer_latency, compute_latency)`
- Especially effective when BLOCK_M is large (high compute/byte ratio)

**Epilogue Subtiling:** Reduces the shared memory needed for the output tile, freeing space for more pipeline stages:
```python
# Instead of writing BLOCK_M × BLOCK_N at once, write in 2 subtiles
acc_sub = acc.reshape(BLOCK_M, 2, BLOCK_N // 2)
tl.store(c_ptr + ..., acc_sub[:, i, :])   # i = 0, 1
```

---

## 5. Software Pipelining

### Multi-Stage Buffering

Allocate `NUM_STAGES` independent shared memory slots. While computing stage `s`, load stage `s+1`:

```python
# Prologue: fill pipeline
for i in range(NUM_STAGES - 1):
    tl.async_copy(a_desc, smem_a[i], k=i)

# Main loop
for k in range(K // BLOCK_K):
    tl.wait_async_copy(smem_a[k % NUM_STAGES])
    c += tl.dot(smem_a[k % NUM_STAGES], smem_b[k % NUM_STAGES])
    tl.async_copy(a_desc, smem_a[(k + NUM_STAGES - 1) % NUM_STAGES], k=k + NUM_STAGES - 1)
```

Optimal `NUM_STAGES`: `ceil(memory_latency / compute_time_per_tile)` — typically 3–5 for B200.

### Stage Merging for Large BLOCK_K

When `NUM_STAGES >= 10`, shared memory may be insufficient. **Merge K-blocks:** double `BLOCK_K`, halve `NUM_STAGES`. Net effect: same pipeline depth, half the shared memory.

---

## 6. Memory Hierarchy Optimizations

### L2 Cache Partitioning

B200 has ~60 MB L2. For GEMM with `BLOCK_N=256`, each B-tile = `256 × BLOCK_K × 1 byte = 32KB`. With `GROUP_M=8`, one N-column sweep reuses the same B-tile for 8 M-tiles → effective bandwidth ÷8 for B.

Rule of thumb: `GROUP_M × BLOCK_N × BLOCK_K × sizeof(dtype) ≤ L2_capacity / num_concurrent_SMs`.

### Shared Memory Bank Conflicts

Each shared memory bank serves 32-bit words. Threads in the same warp accessing the same bank serialize.

**Swizzle formula (128B):**
```
physical_col = logical_col XOR ((logical_row / 8) % 8) * 1
```
Access pattern after swizzle: contiguous 128B rows map to different banks.

Always use swizzle when:
- Loading `BLOCK_K` columns in parallel (all threads in a warp access same row)
- `BLOCK_K × sizeof(T)` is a multiple of 128 bytes

### Register Pressure

B200 has 256 registers/thread. Exceeding → register spilling to L2 → 10-100× slowdown.

Reduce register pressure:
- Smaller `BLOCK_M × BLOCK_N` accumulator (e.g., 64×128 instead of 128×256)
- Epilogue subtiling (write output in 2 passes)
- Split-K: multiple kernels each computing `K // split_k` → reduces K-loop register state

---

## 7. Profiling Guide

### Stage 1: torch.profiler (identify bottleneck stage)

```bash
modal run bench/modal_profile.py --workload-index 8   # compute-bound
modal run bench/modal_profile.py --workload-index 1   # routing-bound
modal volume get flashinfer-trace profiles/ ./profiler_results/
# Open in https://ui.perfetto.dev
```

Look for the dominant stage in the profiler trace (use whatever `record_function` labels the bench script emits for the kernel under study).

### Stage 2: NCU (identify micro-bottleneck)

```bash
modal run bench/modal_ncu.py --workload-index 8 --set detailed
```

Key metrics to examine:

| Metric | Bottleneck | Fix |
|---|---|---|
| `sm__warps_active.avg < 80%` | Low occupancy | Reduce register/smem per thread |
| `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` high | Memory bound | Increase arithmetic intensity |
| `smsp__warp_issue_stalled_mio_throttle_per_warp_active` | Shared mem pressure | Reduce bank conflicts, swizzle |
| `smsp__warp_issue_stalled_lg_throttle_per_warp_active` | L2/HBM bound | More compute per byte |
| `smsp__warp_issue_stalled_math_throttle_per_warp_active` | Compute bound | Increase NUM_STAGES or BLOCK_M |
| `smsp__inst_executed_pipe_tensor_op_hmma` low | Underutilizing tensor cores | Larger BLOCK_M×N, align to 16 |

---

## 8. Split-K for Small-M GEMM

When M is tiny (1–4) and K is large (7168), standard GEMM leaves most SMs idle.
Split-K partitions the K dimension across `SK` CTAs that atomically accumulate
partial results, exposing `SK×` more SM parallelism.

### 8.1 Triton Split-K Pattern

```python
@triton.jit
def split_k_gemm(A_ptr, W_ptr, C_ptr, Lock_ptr, K, N,
                 BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, ...):
    pid     = tl.program_id(0)
    pid_sk  = tl.program_id(1)   # which K-chunk this CTA handles

    k_start = pid_sk * (K // SPLIT_K)
    k_end   = k_start + (K // SPLIT_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k_start, k_end, BLOCK_K):
        a = tl.load(A_ptr + ...)
        w = tl.load(W_ptr + ...)
        acc += tl.dot(a, w)

    # Atomic accumulate into output (with spinlock for non-atomic dtypes)
    tl.atomic_add(C_ptr + ..., acc.to(tl.float32))
```

Alternatively, use a two-pass approach: write partial sums to a `[SPLIT_K, M, N]`
scratch buffer and reduce with a second kernel (avoids atomic precision loss).

### 8.2 Split-K Selection Heuristic (from GemLite)

```python
def pick_split_k(M, K):
    # Rule: target ~128-256 K-elements per CTA per iteration
    if M <= 2:   return min(32, K // 128)
    if M <= 8:   return min(16, K // 256)
    if M <= 32:  return min(8,  K // 512)
    return 1   # large M: no split needed
```

For small M (1–4) with large K (≈7k+), SPLIT_K=8–32 is typical.
Grid becomes `(ceil(M / BLOCK_M) × grid_n, SPLIT_K)` (multiply by an outer batch axis if the workload has one).
