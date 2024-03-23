# Triton Puzzles - solution

My solution to https://github.com/srush/Triton-Puzzles

Things I learned about Triton:
- For `tl.load()` and `tl.store()`, directly use an array of block pointers (something like `x_ptr + tl.arange(0, B1)[:, None] * N + tl.arange(0, B0)[None, :]`) to load/store a block of data. Nothing is mentioned about CUDA coalesced memory access i.e. which thread id corresponds to which axis. However, there is no concept of 'thread' in Triton, so perhaps the Triton compiler will take care of that for us.
- Use `tl.static_range()` for for-loops. This is often use when we iterate along a certain axis.
- Block-level operation is pretty much handled by Triton. We can write it as a normal tensor program - reshape, broadcasting, `tl.dot()`, element-wise op.
