# Broadcasting

We said in the [previous chapter](./operators.md) that arrays must have the same shape to apply element wise operators. This is not exactly true. If one of the axes is exactly `1`, this axis will be replicated along the corresponding axis on the other array. The replication is only logical and does not actually materialize into a larger allocation. Note that a 1-sized axis is completely free in memory.

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

It is also widely used for masking. Let's look at a concrete example. We have a matrix with 1024 rows and 256 columns, we know that the 30 last rows are padding and contain garbage values. We want to find the sum of each rows along the column axis.

`NumPy` comes with a very convenient function called `np.arange(size)` which creates an array of shape `(size,)` where each value is its index. We can use it to create a mask to keep the first first 994 elements by doing `np.arange(arr.shape[0]) < non_padded`.

```python
import numpy as np

# Random matrix of shape (1024, 256)
arr = np.random.normal(size=(1024, 256))

padding = 30
non_padded = arr.shape[0] - padding
# Mask of shape (1024,)
mask = np.arange(arr.shape[0]) < non_padded

# Broadcast and apply the mask
masked_arr = arr * mask[:, None]
# Reduce the first axis
masked_arr.sum(axis=0)
```

## 2D Masking

It is also extremely common in LLMs to build a 2D mask for the attention mechanism. Tokens are only allowed to attend to themselves and to the tokens that came before them. Using broadcasting we can easily build this mask:

```python
import numpy as np

batch_size = 32
sequence_length = 256
num_heads = 4

qk = np.random.normal(size=(batch_size, sequence_length, num_heads, sequence_length))

indices = np.arange(sequence_length)

# Index i >= Index j
# Shape (sequence_length, sequence_length)
mask_2d = indices[:, None] >= indices[None, :]
# Replace the false indices with -infinity so they become 0 in the softmax
mask_2d = np.where(mask_2d, 1.0, -np.inf)
# Apply 2D mask against 4D array
qk_masked = qk * mask_2d[None, :, None, :]
# scores = softmax(qk_masked)
```

## Implementing a matrix multiplication with broadcasting

Some algorithms like [Gated Linear Attention](https://arxiv.org/pdf/2312.06635) use a broadcasted multiplication followed by a reduction to implement a matrix multiplication in order to maintain better numerical stability even though the performance is worse and it cannot be done on accelerated tensor cores.

```python
import numpy as np


a = np.random.normal(size=(32, 64))
b = np.random.normal(size=(64, 16))

# First multiply all indices
a_b = a[:, :, None] * b[None, :, :]
print(f'{a_b.shape=}')

# Then reduce the contracted dimension
out = a_b.sum(axis=1)
print(f'{out.shape=}')

np.testing.assert_almost_equal(out, a @ b)
```

*stdout*

```python
import numpy as np


a = np.random.normal(size=(32, 64))
b = np.random.normal(size=(64, 16))

# First multiply all indices
a_b = a[:, :, None] * b[None, :, :]
print(f'{a_b.shape=}')

# Then reduce the contracted dimension
out = a_b.sum(axis=1)
print(f'{out.shape=}')

np.testing.assert_almost_equal(out, a @ b)
```