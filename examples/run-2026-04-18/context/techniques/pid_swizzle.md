# PID Swizzling (Super-Grouped Tile Order)

## What it is

PID swizzling is a GPU optimization technique used in Triton kernels that
**remaps program IDs** (`pid_m`, `pid_n`) to create better L2 cache locality,
specifically for GEMM-shaped kernels. Adjacent programs end up accessing
memory addresses that are close to each other, so L2 lines fetched by one
program are hit by its neighbors.

## Traditional approach

Row-major mapping:

```python
pid_m = pid // num_pid_n
pid_n = pid %  num_pid_n
```

This launches programs in long horizontal stripes. Programs that execute
concurrently on nearby SMs touch addresses that are far apart in memory — the
L2 is thrashed instead of amortized.

## Swizzled approach

Form **super-groups** of `GROUP_SIZE_M` rows × `num_pid_n` columns, and
traverse each super-group in column-major order:

```python
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_SIZE_M * num_pid_n

group_id       = pid // num_pid_in_group
first_pid_m    = group_id * GROUP_SIZE_M
group_size_m   = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

pid_m = first_pid_m + ((pid %  num_pid_in_group) %  group_size_m)
pid_n =               ( pid %  num_pid_in_group) // group_size_m
```

Effects:

- Neighboring programs in the launch order now share a column (or narrow band
  of columns). L2 lines loaded for operand B are reused across the super-group.
- The last super-group may be short (`num_pid_m % GROUP_SIZE_M != 0`); the
  `group_size_m = min(...)` clamp handles that without branching in the
  producer loop.

## When to use it

Large GEMMs where the B operand (or weight) is loaded many times across
programs — the classic matmul case. Not worth the complexity for small M or
for kernels whose memory traffic is already bounded by A.

Typical `GROUP_SIZE_M` values are 4 or 8; larger values start to reduce
parallelism.

## Attribution

Adapted from `KernelAgent/kernel_perf_agent/kernel_opt/database/docs/pid_swizzle.py`
(Apache-2.0, Meta Platforms Inc.).
