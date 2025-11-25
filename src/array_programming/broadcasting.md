# Broadcasting

We said in the [previous chapter](./operators.md) that arrays must have the same shape to apply element wise operators. This is not exactly true.

1. If one of the axes is exactly `1`, this axis will be replicated along the corresponding axis on the other array.
    - This means that we can add `[4, 2] + [1, 2]`
2. If one array has less dimension than the other, `NumPy` will read both shapes right to left as long as the dimensions match, or if one of them is 1. Then it will virtually add `1` sized dimension to the smaller array.
    - This means that we can add `[32, 64, 64] + [64, 64]`
    - We can add any scalar to any array
    - We cannot implicitly add `[4, 2] + [4,]`, we need to first add a dimension ourselves `[4, 2] + [4,][:, None]`

**Performance Note:** The replication is virtual. NumPy sets the stride to 0 for broadcasted dimensions, meaning the data is not physically copied. A broadcasted axis is "free" in terms of memory.

We can add new axes of size one by slicing the array with an extra `None` or `np.newaxis` at the required position. We can also simply call `arr.reshape(newshape)`.

```python
import numpy as np

# An array full of 1 of shape (4, 2)
ones = np.ones((2, 4, 2))
# Shape (2, 2)
toadd = np.array([[0, 5], [10, 20]])

# Reshape from (2, 2) to (2, 1, 2)
toadd = toadd.reshape(2, 1, 2)
# Alternatively, we could write toadd = toadd[:, None, :]

print(f'{ones + toadd=}')
```

*stdout*

```python
ones + toadd=array([[[ 1.,  6.],
        [ 1.,  6.],
        [ 1.,  6.],
        [ 1.,  6.]],

       [[11., 21.],
        [11., 21.],
        [11., 21.],
        [11., 21.]]])
```

Broadcasting is used in many cases to scale an array or to apply a bias on a whole axis.

## 1D Masking

It is also widely used for masking. Let's look at a concrete example. We have a matrix with 1024 rows and 256 columns, we know that the 30 last rows are padding and contain garbage values.

`NumPy` comes with a very convenient function called `np.arange(size)` which creates an array of shape `(size,)` where each value is its index. We can use it to create a mask to keep the first first 994 elements by doing `np.arange(arr.shape[0]) < non_padded`.

```python
import numpy as np

# Matrix: (1024 rows, 256 cols)
arr = np.random.normal(size=(1024, 256))

padding = 30
valid_rows = arr.shape[0] - padding

# Create a column vector mask: Shape (1024, 1)
# 1. np.arange creates (1024,)
# 2. Comparison creates boolean (1024,)
# 3. Slicing [:, None] adds the axis -> (1024, 1)
mask = (np.arange(arr.shape[0]) < valid_rows)[:, None]

# Broadcast: (1024, 256) * (1024, 1)
# The mask is virtually replicated across all 256 columns
masked_arr = arr * mask

# Sum down the rows (collapsing axis 0)
# Result is (256,) containing the sum of valid elements for each column
print(masked_arr.sum(axis=0).shape) # (256,)
```

## 2D Masking

It is also extremely common in LLMs to build a 2D mask for the attention mechanism. Tokens are only allowed to attend to themselves and to the tokens that came before them. Using broadcasting we can easily build this mask:

```python
import numpy as np

seq_len = 4
# Create indices [0, 1, 2, 3]
indices = np.arange(seq_len)

# Logic: Is query position (i) >= key position (j)?
# (4, 1) >= (1, 4) -> Broadcasts to (4, 4)
is_causal = indices[:, None] >= indices[None, :]

# Create the additive mask
# 0.0 for valid, -inf for invalid (to be zeroed by softmax later)
mask = np.where(is_causal, 0.0, -np.inf)

print(mask)
```

*stdout*

```python
[[  0. -inf -inf -inf]
 [  0.   0. -inf -inf]
 [  0.   0.   0. -inf]
 [  0.   0.   0.   0.]]
```

## Implementing a matrix multiplication with broadcasting

Some algorithms like [Gated Linear Attention](https://arxiv.org/pdf/2312.06635) use a broadcasted multiplication followed by a reduction to implement a matrix multiplication in order to maintain better numerical stability even though the performance is worse and it cannot be done on accelerated tensor cores.

```python
# A: (32, 64)
# B: (64, 16)
a = np.random.normal(size=(32, 64))
b = np.random.normal(size=(64, 16))

# 1. Expand A to (32, 64, 1)
# 2. Expand B to (1, 64, 16)
# 3. Broadcast Multiply -> Result is (32, 64, 16)
intermediate = a[:, :, None] * b[None, :, :]

# 4. Sum over the middle dimension (k=64)
out = intermediate.sum(axis=1)

print(f'{intermediate.shape=}')
print(f'{out.shape=}')

# Verify against standard MatMul
np.testing.assert_almost_equal(out, a @ b)
```

*stdout*

```python
intermediate.shape=(32, 64, 16)
out.shape=(32, 16)
```
