# Data Parallelism

Data parallelism means sharding the batch over multiple chips.

Let's explore this einsum:

```python
out = torch.einsum('bd,df->bf', activations, weights)
```

Let's set `b` to 5120, `d` to `2048` and `f` to `1024`. If we have 2 GPUs, each GPU will see one half of `b` (2560 vectors of size `d` each) and have a full replica of the weights.

In a forward pass, this means we never have to synchronize and we just run half our batch on a chip, and another half on the other chip. During the backward pass, we have to average the gradients from each GPU (using an [all-reduce](./all_reduce.md)) to update our weights with the same value and maintain them replicated.
