# Basic Operators

Most operators applicable to scalars have also been implemented on arrays thanks to [operator overloading](https://www.geeksforgeeks.org/python/operator-overloading-in-python/). The requirement is that all the shapes must match. We will explore cases where shapes do not match in the next chapter about [Broadcasting](./broadcasting.md).

**Important:** In NumPy, the `*` operator represents element-wise multiplication (the Hadamard product), not matrix multiplication. For matrix multiplication, use `@` or `np.matmul`.

```python
import numpy as np

shape = (4, 2, 3)
# An array full of 1 of shape (4, 2, 3)
ones = np.ones(shape)
# An array full of 2 of shape (4, 2, 3)
twos = np.full(shape, 2)

print(f'{ones + twos=}')
print(f'{ones - twos=}')
print(f'{ones * twos=}')
print(f'{ones / twos=}')
print(f'{ones // twos=}')
print(f'{ones == twos=}')
```

*stdout*

```python
ones + twos=array([[[3., 3., 3.], ...
ones - twos=array([[[-1., -1., -1.], ...
ones * twos=array([[[2., 2., 2.], ...
ones / twos=array([[[0.5, 0.5, 0.5], ...
ones // twos=array([[[0., 0., 0.], ...
ones == twos=array([[[False, False, False], ...
```

## Type Promotion (Upcasting)

When you apply an operator to two arrays of different data types, NumPy cannot simply guess which type to use. Instead, it follows a strict set of rules called Type Promotion (or upcasting) to find the smallest data type that can safely represent the result of the operation.

The general hierarchy is: `bool` -> `int` -> `float`.

## How it works

`NumPy` looks for the "common denominator" that prevents data loss:

- `int32` + `int32` -> `int32`
- `int32` + `float32` -> `float64` (Safe default behavior)
- `float32` + `float16` -> `float32`

```python
import numpy as np

shape = (4, 2, 3)
# 1s of type int32
ints = np.ones(shape, dtype=np.int32)
# 2s of type float32
floats = np.full(shape, 2, dtype=np.float32)

print(f'{(ints + ints).dtype=}')
print(f'{(ints + floats).dtype=}')
print(f'{(floats + floats).dtype=}')
```

*stdout*

```python
(ints + ints).dtype=dtype('int32')
(ints + floats).dtype=dtype('float64')
(floats + floats).dtype=dtype('float32')
```
