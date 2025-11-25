# Basic Operators

Most operators applicable to scalars have also been implemented on arrays thanks to [operator overloading](https://www.geeksforgeeks.org/python/operator-overloading-in-python/). The requirement is that all the shapes must match. We will explore cases where shapes do not match in the next chapter about [Broadcasting](./broadcasting.md).

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
```

*stdout*

```python
ones + twos=array([[[3., 3., 3.], ...
ones - twos=array([[[-1., -1., -1.], ...
ones * twos=array([[[2., 2., 2.], ...
ones / twos=array([[[0.5, 0.5, 0.5], ...
ones // twos=array([[[0., 0., 0.], ...
```
