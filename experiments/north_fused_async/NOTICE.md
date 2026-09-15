# Attribution

The affine projection grouping in kernel.h and fusion.py is adapted from
Apple's MLX quantized.h, shipped in MLX 0.32.2 (Copyright © 2023–2024 Apple Inc.).
BF16 sigmoid behavior was aligned with MLX unary_ops.h. Attention projection
ordering and the dense projections in prep.h follow MLX's GEMV implementation.
The normalization in prep.py follows MLX's RMSNorm reduction and rounding.
The following license covers the adapted portions:

MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


The unmodified North reference implementation is loaded from mlx-vlm
cdc745ad8a32d162f6d8e9d08be256910d663ac2. Its MIT license is retained in the
source checkout. Model weights are downloaded separately under the checkpoint's
license; this experiment does not redistribute them.
