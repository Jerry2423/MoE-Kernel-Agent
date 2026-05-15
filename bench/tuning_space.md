## Tunable Parameters for GEMM Kernels

Adjusting num_stages in tr.range can produce optimizations, reminding us that there is still a lot of tuning room for gemm1 kernel and gemm2 kernel:

1. Original kernel launch config: block size, num_stages, num_warps

2.triton.language.range:
    - num_stages – pipeline the loop into this many stages (so there are num_stages iterations of the loop in flight at once).

    Note this is subtly different than passing num_stages as a kernel argument. The kernel argument only pipelines loads that feed into dot operations, while this attribute tries to pipeline most (though not all) loads in this loop.

    - loop_unroll_factor – Tells the Triton IR level loop unroller how many times to unroll a for loop that this range is used with. Less than 2 for this value implies no unrolling.

    - disallow_acc_multi_buffer – If true, prevent the accumulator of the dot operation in the loop to be multi-buffered, if applicable.

    - warp_specialize – Enable automatic warp specialization on the loop. The compiler will attempt to partition memory, MMA, and vector operations in the loop into separate async partitions. This will increase the total number of warps required by the kernel.

    - disable_licm – Tells the compiler it shouldn’t hoist loop invariant code outside the loop. This is often useful to avoid creating long liveranges within a loop.

3.triton.language.load:
    - boundary_check (tuple of ints, optional) – tuple of integers, indicating the dimensions which should do the boundary check

    - padding_option – should be one of {“”, “zero”, “nan”}, the padding value to use while out of bounds. “” means an undefined value.

    - cache_modifier (str, optional, should be one of {“”, “.ca”, “.cg”, “.cv”}, where “.ca” stands for cache at all levels, “.cg” stands for cache at global level (cache in L2 and below, not L1), and “.cv” means don’t cache and fetch again. see cache operator for more details.) – changes cache option in NVIDIA PTX

4. tl.store:
    - boundary_check (tuple of ints, optional) – tuple of integers, indicating the dimensions which should do the boundary check

    - cache_modifier (str, optional, should be one of {"", ".wb", ".cg", ".cs", ".wt"}, where ".wb" stands for cache write-back all coherent levels, ".cg" stands for cache global, ".cs" stands for cache streaming, ".wt" stands for cache write-through, see cache operator for more details.) – changes cache option in NVIDIA PTX

    - eviction_policy (str, optional, should be one of {"", "evict_first", "evict_last"}) – changes eviction policy in NVIDIA PTX

5. Whether padding should be performed: If the total tokens T of the input are not an integer power of 2, whether padding T should be performed