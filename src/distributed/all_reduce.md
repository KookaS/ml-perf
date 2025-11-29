# All Reduce

An all reduce takes a sharded array, and it combines the elements of an axis using an accumulation function, typically but not necessarily, the sum.

For instance, a vector of length `256` whose single axis would be sharded over 4 devices:

![img](./all_reduce.png)

Each TPU initially holds 64 unique elements, after the all reduce, they all hold a vector which is replica of the sum of the vectors initially held by each chip.
