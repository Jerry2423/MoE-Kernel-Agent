# Open questions — things Phase 1 could not resolve from `context/` alone

## Scheduler-level

- What token-count distribution per expert should we assume for Phase 2 autotune workload selection? `context/moe_workloads.jsonl` names 19 workloads but was not exhaustively read in Phase 1 (schema discipline). Phase 2 should read it before picking the representative set.
- Does the bench harness expect routing to run as a separate launch from the GEMMs, or is a single persistent fused kernel allowed? (problem spec) describes two specialized persistent kernels today. Need to confirm by reading `bench/` (Phase 2 territory).
- Is `launch_with_pdl=True` available from Triton, or Cuda-launch only? If only Cuda, our Triton routing kernel cannot use it.

## Precision / correctness

- `atol=1, rtol=0.3, required_matched_ratio=0.9` ((problem spec)) is unusually generous. What is the failing-ratio budget in practice — does the harness count NaNs as mismatches?
- Does the harness accept fp32 scratch output that we cast to bf16 in a tiny post-kernel, or must the single Triton `run()` return bf16 directly? (problem spec) says `[T,H] bf16` out, but DPS contract specifics aren't spelled out.

## Numerical / hardware specifics

- `tl.dot_scaled` on B200 with FP8 e4m3 × e4m3 and UE8M0 scales — does the fp8e4m3 path accept the same scale encoding, or do we need manual post-scaling? Decide in Phase 2.2 after peeking at the Triton version in `fi-bench`.
- Does `warp_specialize=True` in `tl.range` produce correct code on current Triton+B200 for our exact tile shapes?

## Routing edge cases

- When a token selects 0 local experts (i.e., all 8 of its top-k are on other ranks), does the bench expect the output row to be zero (consistent with `torch_ref.py:129` initializing `output` to zero), or does the reference kernel already guarantee that tokens with empty local bitmatrix are skipped?
- The DSv3 routing uses `sorted=False` everywhere (`torch_ref.py:107,111,119`). Are ties between equal biased scores broken deterministically enough that bitwise-exact correctness is achievable, or does the bench tolerance already absorb ties?

## Profiling / tooling (Phase 3 only)

- Is `modal run -m bench.modal_ncu --preset ako` always available, or does it require an extra image build? (Referenced in `context/INDEX.md` row "NCU canonical preset".) Defer until Phase 3.
