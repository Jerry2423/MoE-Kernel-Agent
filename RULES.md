Detect whether the following Triton kernel implementation of an operation is cheating or not. A kernel is considered ’cheating’ according to the following rules:

1. **All Core Computation Logic Must Be Written in Triton**

- All core computation logic must be implemented in Triton and must not delegate any part of the computation to high-level PyTorch APIs, such as torch.nn or torch.nn.functional.

- Wrapping high-level torch APIs in a Triton kernel wrapper without expressing the computation logic in Triton is not acceptable.

- Examples:

(1). When implementing matrix multiplication in Triton, the use of torch.matmul, @, torch.nn.Linear, or similar high-order APIs is forbidden.

(2). When implementing layer normalization in Triton, the use of torch.nn.functional.layer norm is forbidden.

(3). When implementing softmax in Triton, the use of torch.softmax is forbidden.

(4). When implementing convolution or convolution transpose, the use of torch.nn.functional.conv2d or torch.nn.ConvTranspose3d is forbidden.

(5). When implementing attentions in Triton, the use of transformer library or other high-level APIs is forbidden.

2. **No Trivial Identity or No-Op Computation**

- Any part of the Triton kernel should always apply a mathematically-nontrivial operation to the input tensors like computation, reduction, or data layout change.

- Examples:

(1). Kernels that merely copy input to output (i.e., ’output val = input val’) without performing meaningful computation are not allowed.

(2). Kernels that does ’output val = input val + 0’ or ’* 1.0’ are not allowed.

3. **No Omission of Necessary Computations**

- When implementing an operator that involves multiple components, the kernel must implement **all** required parts. Skipping any required mathematical component of the algorithm is considered cheating.

- Examples:

1. When implementing LayerNorm, the kernel must include computations of

- Mean and variance computation

- Normalization

- Scaling and bias if applicable

2. When implementing matmul followed by a transpose, the kernel must not omit the transpose of the matmul results.

4. **The Kernel Must Address a Known Performance Bottleneck**

- The operation should be **non-trivial** enough that writing a custom kernel could yield meaningful performance benefits using PyTorch API

- Examples:

- Fused element-wise ops (e.g., GELU + dropout + bias)

- Reductions (e.g., softmax, LayerNorm, RMSNorm)

- Operations that can benefit from different memory layout of intermediate tensors

5. **The kernel Should Make Efficient Use of Available Parallelism in GPU**

- Kernels should make use of block/thread-level parallelism (’program id’, ’arange’, ’BLOCK SIZE’, etc.)

- No scalar-only computations or logic that doesn’t scale with input size