# What is an Array ?

At a high level, an array is an abstract representation of an n-dimensional arrangement of numbers. All the numbers share the same underlying data type (for instance, `int32`.) It exposes 4 necessary pieces of information:

1. **The Data Pointer:** The memory address where the data begins.
2. **The Dtype:** The type of every element (e.g., int32, float16). This tells the CPU how many bytes to read per element.
3. **The Shape:** The logical dimensions of the array.
4. **The Stride:** An extra piece of information that specifies the number of bytes to step in each dimension to reach the next element. This decouples the data layout from the logical shape, allowing for zero-copy operations like transposing.

## The Physical Reality

Under the hood, the data is simply a 1D buffer of contiguous memory. The shape and dtype are used to calculate the total memory required for allocation.

For example, in C, the allocation looks like this:

```c
// A (10, 10) matrix and a (100,) vector allocate the exact same memory.
size_t size = product(shape) * dtype.itemsize;
void* buffer = malloc(size);
```

To the memory allocator, the shape is irrelevant. It only cares about the total number of bytes. The shape is a logical construct used by the software to:

1. **Compute real memory addresses:** It maps logical coordinates $(x, y)$ to a flat memory offset.
2. **Determine validity:** It prevents accessing memory outside the allocated buffer (bounds checking).
3. **Define semantics:** It dictates how operations broadcast across dimensions.

The CPU finds an element at logical index `(i, j)` using this fundamental formula:

```python
address = data_pointer + (i * stride[0]) + (j * stride[1]) 
```

## Creating Arrays

We can build an array from a python list:

```python
import numpy as np

data = [[0, 1], [2, 3]]

arr = np.array(data, dtype=np.int32)
print(arr, arr.shape)
```

*stdout*

```python
[[0 1]
 [2 3]] (2, 2)
```

We can also build an array from a simple scalar

```python
import numpy as np

scalar = np.array(10, dtype=np.int32)
print(scalar.shape)
```

*stdout*

```python
()
```
