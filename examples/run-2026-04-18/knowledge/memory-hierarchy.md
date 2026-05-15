---
topic: memory-hierarchy
applies_to: all three stages (routing, GEMM1, GEMM2); dominant for seq_len ≤ ~512
source: .claude/skills/OPTIMIZATION_TECHNIQUES.md (§1 L2, §6), context/triton_tricks.md (§1, §2, §5), context/techniques/tma.md, context/hardware/b200.md
confidence: high
---

# Memory hierarchy — global → L2 → shared → register

## Summary

On B200 the memory path is `HBM3e (8 TB/s, 180 GB) → L2 (126 MB, split across two dies) → SMEM (228 KB/SM) → registers (256/thread @ 64 warps)`. The design goal for every stage of the MoE kernel is to keep operand reuse as close to registers as possible: activations in L2 (`evict_last`), weights streamed once (`evict_first`), and shared-memory writes swizzled to eliminate bank conflicts.

## When to use

- GEMV-ish or GEMV-shaped stages (`M ≤ 4` after routing) — L2 cache behavior dominates, per `triton_tricks.md §2` (lines 27–40).
- Any kernel loading one operand once while the other is reused — apply `eviction_policy='evict_first'` on the streaming operand.
- TMA path for GMEM↔SMEM moves when per-CTA tile shape is static (on-host) or varies per CTA (on-device) (`tma.md`, "Which path to pick").
- Swizzle SMEM layout whenever `BLOCK_K × sizeof(T)` is ≥ 128 bytes.

## Code pattern

```python
# evict-policy tuning (triton_tricks.md §2)
a = tl.load(a_ptr, mask=a_mask, eviction_policy='evict_last')   # reused activations
b = tl.load(b_ptr,              eviction_policy='evict_first')  # streamed weights

# on-device TMA descriptor (tma.md §On-device)
a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, K], strides=[K,1],
                                   block_shape=[BLOCK_M, BLOCK_K])
a = a_desc.load([pid_m * BLOCK_M, k_off])

# load early, use late — overlaps HBM latency with MMA (triton_tricks.md §1)
a_next = a_desc.load([pid_m*BLOCK_M, k + BLOCK_K])  # issue early
acc = tl.dot(a_cur, b_cur, acc)                      # consume previous
```

## Tradeoffs

- L2 super-grouping (`GROUP_M × BLOCK_N × BLOCK_K × sizeof(dtype) ≤ L2_capacity / concurrent_SMs`) costs latency on the last super-group if `num_pid_m % GROUP_M != 0`.
- TMA descriptors remove pointer-arithmetic loads but add per-program setup cost on the device path — amortize over many K-steps (`tma.md` §Pros/cons).
- `evict_first` on a weight can starve a reuser downstream — only apply when you know the weight isn't touched again within the persistent loop.
- 128 B swizzle assumes column stride is aligned; breaks silently if you forget to size SMEM to a multiple of the swizzle width.

## Pitfalls

- Forgetting `eviction_policy` on streamed weights pollutes L2 and evicts activations, visible as low `lts__t_sector_hit_rate.pct` in NCU.
- Register spills from oversized accumulators: B200 = 256 regs/thread before spill (`b200.md` row `register_file_kb_per_sm`). Spills go to L2 and cost 10–100× (§6).
- Writing to non-swizzled SMEM from a warp where all threads share a row serializes banks 32-way.
- On-device TMA needs `triton.set_allocator(alloc_fn)` set up in host code (`tma.md:95–105`); missing this yields obscure init failures when porting a kernel from on-host style.
