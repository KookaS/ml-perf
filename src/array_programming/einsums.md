# Einsums

Einsums are the lifeblood of tensor arithmetic in ML. They provide a clear syntax to express high dimensional tensor operations. Furthermore, they are often more efficient than using a mix of traditional operators because linear algebra libraries are able to reorder the operations to minimize the materialized size.

## Syntax

We write an einsum using `np.einsum(subscripts, *operands)` function.

1. `subscripts` is a python string defining the operation to apply to the operands.
    - The string is formatted as such `dims_1,dims_2,...->dims_out`
    - We give a name to each dimension of each operand for instance a batch of images could be `bwh` (batch, width, height.)
    - We separate operands with `,`. For instance `bwh,whd` (where `d` is the model dimension.)
    - We specify the output dimensions after `->`. For instance `bwh,whd->bd`.
2. `*operands` are an arbitrary amount of arrays to which the operation will be applied. For instance `np.einsum('bwh,whd->bd', images, weights)`

## Understanding Einsums

1. **Repeating Letters:** If an index appears in two inputs (e.g., `j` in `ij`, `jk`), it implies multiplication along that dimension.
2. **Omitted Letters (Reduction):** If an index appears in the input but not the output, it is summed over (reduced).
3. **Output Order:** You can rearrange the output dimensions arbitrarily (e.g., `ij` -> `ji` is a transpose).

| Operation | Standard API | Einsum Notation |
| :--- | :--- | :--- |
| **Transpose** | `A.T` | `ij -> ji` |
| **Sum** | `A.sum()` | `ij ->` |
| **Column Sum** | `A.sum(axis=0)` | `ij -> j` |
| **Dot Product** | `a @ b` | `i, i ->` |
| **Matrix Mul** | `A @ B` | `ik, kj -> ij` |
| **Batch MatMul** | `A @ B` | `bik, bkj -> bij` |
| **Outer Product** | `np.outer(a, b)` | `i, j -> ij` |

## Broadcasting with Ellipsis (`...`)

In Deep Learning, we often write code that shouldn't care about the number of batch dimensions (e.g., handling both (`batch`, `sequence`, `feature`) and (`batch`, `sequence`, `num_heads`, `feature`)).

`einsum` supports `...` to represent "all other dimensions".

```python
# Apply a linear layer (Weights: i, j) to a tensor
# of ANY shape ending in 'i'
# ...i, ij -> ...j
output = np.einsum('...i,ij->...j', input_tensor, weights)
```

## Code Examples

```python
import numpy as np

batch = 10
width = 28
height = 64
d_model = 512

images = np.random.normal(size=(batch, width, height))
weights = np.random.normal(size=(width, height, d_model))

print(f"{np.einsum('bwh,whd->bd', images, weights).shape=}")
```

*stdout*

```python
np.einsum('bwh,whd->bd', images, weights).shape=(10, 512)
```

This reduces both the `width` and the `height`. But we could also just reduce the `width` for instance, batch the `height` and write the output in a different order. For instance `bwh,whd->dbh`.

*stdout*

```python
np.einsum('bwh,whd->dbh', images, weights).shape=(512, 10, 64)
```

## Path Optimizations

When multiplying three or more matrices, the order of operations matters significantly for memory.

`(A @ B) @ C` vs `A @ (B @ C)`

If `A` is `(1000, 2)`, `B` is `(2, 1000)`, and `C` is `(1000, 1000)`:

1. `A @ B` creates a `(1000, 1000)` intermediate matrix (1M elements.)
2. `B @ C` creates a `(2, 1000)` intermediate matrix (2k elements.)

The second path is orders of magnitude more memory efficient. `np.einsum` (with `optimize=True`) automatically finds this path.

```python
import numpy as np

# A chain of 3 matrix multiplications
# Dimensions chosen to make one path disastrously memory heavy
a = np.random.normal(size=(1000, 2))
b = np.random.normal(size=(2, 1000))
c = np.random.normal(size=(1000, 1000))

# Naive chaining (Left-to-Right)
# Creates (1000, 1000) intermediate!
res_naive = (a @ b) @ c

# Einsum Optimization
# Automatically detects that contracting (b, c) first is cheaper
res_einsum = np.einsum('ij,jk,kl->il', a, b, c, optimize=True)

np.testing.assert_allclose(res_naive, res_einsum)
```

## Code Visualization

`einsum` can be difficult to debug. It helps to visualize it as a nested loop.

### Single reduced dimension

Let's visualize `bwh,whd->db`. We are reducing `w` and `h`, and transposing the result to `d, b`.

```python
import numpy as np

batch = 10
width = 28
height = 64
d_model = 512

images = np.random.normal(size=(batch, width, height))
weights = np.random.normal(size=(width, height, d_model))


manual_out = np.zeros((d_model, batch, height))

# One loop per non reduced dimension
for b in range(batch):
  for h in range(height):
    for d in range(d_model):
      manual_out[d, b, h] = images[b, :, h] @ weights[:, h, d]


einsum_out = np.einsum('bwh,whd->dbh', images, weights)
np.testing.assert_almost_equal(manual_out, einsum_out)
```

We loop over all our batch dimensions, we extract vectors of size `w` that we dot product and write at the correct (transposed) output dimension.

### Multiple Reduced Dimension

The `bwh,whd->db` einsum is more interesting because it reduces both `w` and `h`. Concretely, the only difference with the above einsum is that we will revisit the same `d, b` output tile multiple times, so we need to reduce intermediate dot products into their corresponding output indices.

```python
...

manual_out = np.zeros((d_model, batch))

# One loop per non reduced dimension
for b in range(batch):
  for h in range(height):
    for d in range(d_model):
        # The 'w' dimension is reduced via the dot product (@)
        # We accumulate (+=) because 'h' is also being reduced
        manual_out[d, b] += images[b, :, h] @ weights[:, h, d]


einsum_out = np.einsum('bwh,whd->db', images, weights)
np.testing.assert_almost_equal(manual_out, einsum_out)
```
