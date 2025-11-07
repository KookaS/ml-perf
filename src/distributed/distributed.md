# Distributed Computations

When a single chip is not enough, we can use multiple chips working together to either lower our latency, to increase the size of our model, or to increase our batch size.

As we scale our model to large number of parameters, we end up using more memory than we can fit on a single device. At this point, we need to add another device to make our model's weights fit.

There are also scenarios where parameters do fit on a single device, but we want to process massive amounts of data. This also forces us to distribute our computations.

## Collective Operations and Sharding

We will end up distributing our computations differently depending on the model's architecture, the workload, the type, and the number of devices we have access to.

We typically call "sharding" the act distributing an axis of a model on multiple devices.

There are three Collective Operations typically used to synchronize the state of computations across devices, or to move from one sharding to another; [all gather](./all_gather.md), [all reduce](./all_reduce.md), and [all to all](./all_to_all.md). We will introduce them, then explore how they are used in different scenarios.
