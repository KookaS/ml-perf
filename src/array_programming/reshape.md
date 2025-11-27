# Reshaping And Transposing

It is very common to want to change how interpret our data. For instance, we might want to flatten a `(28, 28)` image into a single `(784,)` vector.

Both reshape and transpose are designed to be metadata-only operations. They change the `metadata` (`shape` and `stride`) without touching the underlying `buffer`.

## Reshaping

- Reshaping changes the logical dimensions of the array while keeping the total number of elements constant.
- It only changes the logical shape, the values at physical indices remain constant.
  - For instance, if we reshape from `(10,)`to `(5, 2)`, the value at index `arr[2]` before reshape will be the same as the value at index `arr[1, 0]` ater the reshape.
- The product of the new shape must equal the product of the old shape. `prod(new_shape) == prod(old_shape)`.

```python
import numpy as np

original = np.arange(12).reshape(2, 3, 2)

reshaped = original.reshape(3, 4)

print(f"{original.shape=}, {original.strides=}")
print(f"{reshaped.shape=}, {reshaped.strides=}")


print(f"{original=}")
print(f"{reshaped=}")
```

*stdout*

```python
original.shape=(2, 3, 2), original.strides=(48, 16, 8)
reshaped.shape=(3, 4), reshaped.strides=(32, 8)

original=array([[[ 0,  1],
        [ 2,  3],
        [ 4,  5]],

       [[ 6,  7],
        [ 8,  9],
        [10, 11]]])

reshaped=array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
```

Conveniently, we do not have to write out all the dimensions when we reshape. Passing `-1` will infer the size of the remaining dimension.

```python
# We have a buffer of 7840 elements
data = np.arange(7840)

# We want 28x28 images, but we don't want to manually calc the batch size.
# NumPy calculates: 7840 / (28 * 28) = 10
formatted = data.reshape(-1, 28, 28)

print(f'{formatted.shape=}') # (10, 28, 28)
```

*stdout*

```python
formatted.shape=(10, 28, 28)
```

## Transposing

Transposing swaps axes. It means that after a transposition, elements in the array have logically moved.

Let's imagine an array of shape `(10, 32, 64)`.

- Let's transpose the last two axes (we can use `swapaxes(1, 2)`). The array becomes `(10, 64, 32)`. The value at index `[0, 1, 2]` will now be at index `[0, 2, 1]`.
- As mentioned earlier, no data is actually moved, we just change the stride of the array.
- There are many APIs for transposing.
  - Arrays with one or two dimensions can use `.transpose()` or `.T`.
  - Any array can use `.transpose(*indices)` (equivalent to permute in `PyTorch`) where indices maps the new axes to the old axes. For instance `(10, 32, 64).transpose(2, 0, 1)` becomes `(64, 10, 32)`.
  - Any array can use `.swapaxes(axis1, axis2)` to swap the two axes provided.

```python
import numpy as np

original = np.arange(10).reshape(2, 5)

# Transpose
transposed = original.T

print(f"{original.shape=}, {original.strides=}")
print(f"{transposed.shape=}, {transposed.strides=}")


print(f"{original=}")
print(f"{transposed=}")
```

*stdout*

```python
original.shape=(2, 5), original.strides=(40, 8)
transposed.shape=(5, 2), transposed.strides=(8, 40)

original=array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9]])

transposed=array([[0, 5],
       [1, 6],
       [2, 7],
       [3, 8],
       [4, 9]])
```

## The Performance Trap: Contiguity

NumPy arrays are laid out in Row-Major order (C-style) by default. This means iterating over the last dimension is stepping 1 item at a time in memory (contiguous).

When you Transpose, you break this contiguity. The stride of the last dimension is no longer 1.

- **Reshaping a Contiguous Array:** Free (View).
- **Reshaping a Non-Contiguous Array:** Expensive (Force Copy).

If you attempt to reshape an array that has been transposed, NumPy is often forced to physically copy the data into a new, contiguous buffer to satisfy the reshape request.

| Operation | Action | Cost |
| :--- | :--- | :--- |
| **reshape** | Updates shape/strides | ✅ **Free** (usually) |
| **transpose** | Swaps shape/strides | ✅ **Free** (always) |
| **reshape after transpose** | Reorganizes Memory | ❌ **Expensive** (Copy) |
