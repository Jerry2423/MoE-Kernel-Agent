# Skill: Advanced Triton Kernel Optimization

## Overview
This skill provides a set of heuristic-based optimizations ("Tricks") for writing high-performance Triton kernels. These techniques focus on memory hierarchy management, L2 cache utilization, and autotuning efficiency, particularly relevant for GEMM, GEMV, and Split-K workloads.

---

## 1. Memory Loading & Instruction Order
The order in which data is loaded from global memory into SRAM can significantly impact instruction pipelining and memory latency hiding.

* **Mechanism:** In some kernels, loading weights first before the activation (or vice-versa) changes how the GPU schedules memory requests.
* **Implementation:** Don't hardcode the order. Instead, expose it as a tunable parameter for the Triton autotuner.

```python
# Implementation in the kernel
if load_order == 0:
    a = tl.load(a_ptr, mask=a_mask)
    b = tl.load(b_ptr)
else:
    b = tl.load(b_ptr)
    a = tl.load(a_ptr, mask=a_mask)

# In the autotuning config
triton.Config({'BLOCK_SIZE_M': 128, ...}, num_warps=4, num_stages=3, kwarg={'load_order': [0, 1]})
```

## 2. Cache Eviction Policies

For small batch sizes (like GEMV) or Split-K operations, the L2 cache behavior is a primary bottleneck.

- evict_last: Use this for data that will be reused frequently within the same SM or across iterations (e.g., activations in GEMV). This tells the hardware to keep this data in the cache as long as possible.

- evict_first: Use this for "streaming" data that is read once and not reused immediately (e.g., weights in a large GEMV), preventing it from "polluting" the cache and kicking out more useful data.

```python
# Prioritize keeping activations in L2
a = tl.load(a_ptr, mask=a_mask, eviction_policy='evict_last')
# De-prioritize weights
b = tl.load(b_ptr, eviction_policy='evict_first')
```

## 3. Loop Swapping & Stability

Standard Python range in kernels can sometimes lead to suboptimal IR generation or even compiler crashes on specific GPU architectures.

- Optimization: Replace for k in range(num_pid_k): with for k in tl.range(0, num_pid_k, 1, num_stages=1).

- Benefit: This explicit range often avoids crashes and can result in tighter assembly loops.

## 4. Autotuning & Config Pruning

Autotuning can be slow if the search space includes invalid configurations (e.g., when Split-K values don't align with group_size).

- Technique: Use a pruning function to skip invalid or known-bad configurations before the benchmark runs.

- Syntax:
```python
def kernel_config_pruner(configs, named_args):
    # Logic to filter out invalid BLOCK_SIZE or Split-K combinations
    return [c for c in configs if ... ]

@triton.autotune(
    configs=[...],
    prune_configs_by={'early_config_prune': kernel_config_pruner}
)
```

## 5. Atomic Operations & Initialization

Atomic additions can be expensive. Minimizing the overhead of zero-initializing the output buffer is a common "free" win.

- Pre-hooks: Use a pre_hook in your triton.Config to zero-initialize the output buffer on the device side before the kernel launches. This is faster than manual initialization inside a kernel or a separate torch.zero_ call.

- Memory Consistency: For tl.atomic_add, the default memory synchronization is acq_rel. Switching to relaxed or release often yields better throughput if strict ordering isn't required for your specific algorithm.

| Operation | Preferred Parameter | Default |
|-----------|-------------------|---------|
| Atomic Memory Sync | `relaxed` or `release` | `acq_rel` |
| Reshape | `can_reorder=False` | `True` |

```python
# Example Pre-hook for zeroing output
def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

config = triton.Config(
    {...}, 
    pre_hook=init_to_zero("c_ptr")
)

def forward(...):
    output = torch.empty(...)
    kernel_call[grid](..., c_ptr=output)
    return output
```

