# Quantization

We typically store a model's weights in the [bfloat16 floating-point format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format). Which means that each parameter takes 2 bytes.

We can halve the memory bandwidth usage by simply using [fp8 formats](https://arxiv.org/abs/2209.05433). We can even go further with smaller formats like [int4 quantization](https://arxiv.org/pdf/2301.12017) all the way down to [1.58 bits quantization](https://en.wikipedia.org/wiki/1.58-bit_large_language_model).

## Hardware Support

Quantization is particularly useful in `memory-bound` regimens because it drastically reduces the amount of data movements. Besides, modern chips now support lower precision arithmetic with higher flops per second than with higher precision.

For instance, according to [this](https://www.nvidia.com/en-us/data-center/h100/), `H100` GPUs can do `1979` teraFLOPS in `bf16` while they can do `3958` teraFLOPS in `fp8`.

Note that there are no mentions of `int4` in the table. This is because `int4` is not hardware supported by the `H100`, so there would need to be a conversion to `fp8` before using the tensor core. Therefore, `int4` would not yield compute throughput gains, only bandwidth gains. *It is important to check your hardware's specification.*

## Scales

Simply rounding weights to the nearest integer would degrade performance too much because model weights can have very different magnitudes (e.g., outliers). To solve this, we introduce a new tensor called `scales`.

The `scales` map the small integer range (e.g., `-127` to `127`) back to the original floating point range. A quantized dot product between activations \\(x\\) and quantized weights \\(W_{quantized}\\) with scales \\(S\\) looks like this: \\[(x\cdot W_{quantized}) \times S\\]

We first apply the matrix multiplication between \\(x\\) and \\(W_{quantized}\\) using the smaller dtype (fast), then we scale up the result using an element-wise product with \\(S\\) in the original `dtype` but with a much smaller tensor.

If the original weights had a shape of `(d_in, d_out)`, the scales typically have a shape of `(1, d_out)`. This is called **Channel-wise Quantization**. The scales are tiny compared to the weights, adding negligible memory overhead.

### Obtaining the Quantized Weights and Scales

There are different methods of obtaining the weights and scales. A simple and common approach is called **Symmetric Block-wise Quantization**. We map the absolute maximum value of a row/column to the maximum integer value (e.g., 127 for int8).

1. Calculate the absolute maximum value for the channel: \\(\alpha = \max(|W|)\\).
2. Calculate the **Scale** (\\(S\\)): \\(S = \frac{\alpha}{127}\\).
3. Calculate **Quantized Weights** (\\(W_{quantized}\\)): \\(W_{quantized} = \text{round}(\frac{W}{S})\\)
4. **Dequantization** (Forward Pass): \\(\text{output} = (x @ W_{quantized}) \times S$\\)

## Code Example

```python
import numpy as np

# 1. Create random weights in float16
# Shape: (Input Dim, Output Dim)
d_in, d_out = 64, 128
weights = np.random.normal(size=(d_in, d_out)).astype(np.float16)

# 2. Calculate Scales (Channel-wise)
# We want one scale per output column -> Shape (1, d_out)
# We use int8, so max_int is 127
max_val = np.max(np.abs(weights), axis=0, keepdims=True)
scales = max_val / 127.0

# 3. Quantize
# Divide, Round, and Cast to int8
weights_quantized = (weights / scales).round().astype(np.int8)

print(f"Original Memory: {weights.nbytes} bytes")
print(f"Quantized Memory: {weights_quantized.nbytes + scales.nbytes} bytes")

# 4. Dequantization (Forward Pass)
# We simulate the matmul. In reality, hardware does this in mixed precision.
x = np.random.randn(1, d_in).astype(np.float16)

# Integer Matmul
out_int = x @ weights_quantized

# Rescale to float
out_float = out_int * scales
```
