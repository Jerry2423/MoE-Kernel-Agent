# Persistent Programming Style

## What it is

The persistent programming style is a kernel design pattern where a **fixed**
number of thread blocks is launched — typically equal to the number of
streaming multiprocessors (SMs) — instead of launching blocks proportional to
the problem size. It is particularly effective when the problem size exceeds
the GPU's parallel capacity.

## Traditional approach

An unoptimized Triton kernel launches a grid sized to the problem:

```python
grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
kernel[grid](...)
```

Each block processes exactly one tile. When the number of tiles exceeds the
number of SMs, block start-up and retirement overhead is paid repeatedly, and
cache-resident state is lost at every block boundary.

## Persistent approach

Launch a fixed number of blocks — often `torch.cuda.get_device_properties(0).multi_processor_count`
— and have each block loop over tile IDs with a stride equal to the total
number of blocks:

```python
NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

@triton.jit
def persistent_kernel(..., NUM_SMS: tl.constexpr, NUM_TILES: tl.constexpr):
    start_pid = tl.program_id(0)
    for tile_id in range(start_pid, NUM_TILES, NUM_SMS):
        pid_m = tile_id // num_pid_n
        pid_n = tile_id %  num_pid_n
        # ...body uses pid_m, pid_n
```

These blocks **persist** and loop until all work is done.

## Advantages

- **Better resource utilization** — the launched grid matches hardware width exactly.
- **Reduced launch overhead** — a single kernel launch covers an arbitrarily large problem.
- **Improved occupancy stability** — all SMs stay busy throughout execution.
- **Better cache locality** — state (e.g. partial accumulators, descriptors) can be
  reused across tiles within the same block.
- **Load balancing** — work is distributed evenly; no stragglers from the last wave.

## When it applies on B200

Best suited when: many tiles per SM (≥4), tile work is uniform, and the kernel
has reusable per-block state (descriptors, accumulators, warp-specialized
pipelines).

## Attribution

Adapted from `KernelAgent/kernel_perf_agent/kernel_opt/database/docs/persistence.py`
(Apache-2.0, Meta Platforms Inc.).
