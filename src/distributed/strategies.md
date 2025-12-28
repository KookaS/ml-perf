# Sharding Strategies (intro)

We can split computations and data across devices to reduce the flops and the amount of memory needed per device. There are many ways we can split data and computations across devices. This chapter explores common sharding strategies.

## Pseudo API

We will illustrate this chapter using a fake distributed API over `numpy`. This API contains all the building blocks required to enable distributed computations. The code snippets would conceptually be run on all devices in parallel. Devices are assigned a `device_id` to differentiate.

The API revolves around an abstract class that we have to inherit from. We have to implement 3 methods ourselves `load_checkpoint`, `forward` and `backward`.

The API provides pre implemented methods `device_id`, `num_devices`, `barrier`, `send`, `receive`, `all_gather`, `all_reduce`, `all_to_all`.

We also have an `inference_loop` method that simply loops over a stream of inputs and streams back the outputs to an output stream.

We do not provide the loss function and simply assume it is separately provided by the optimizer API.

```python
from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from io import Reader, Writer

class ShardedEngine(ABC):
    
    @property
    def device_id(self) -> int:
        """The rank of the current device."""
        ...

    @property
    def num_devices(self) -> int:
        """Total number of devices in the cluster (World Size).""" npt.ArrayLike
        ...

    def barrier(self) -> None:
        """Blocks until all devices reach this line."""
        pass
    
    # --- Point-to-Point Communication ---
    def send(self, dest_id: int, arr: npt.ArrayLike) -> None:
        ...
    
    def receive(self, src_id: int) -> npt.ArrayLike:
        ...

    def send_async(self, src: npt.ArrayLike, dst: npt.ArrayLike, target_device_id: int) -> Future:
        ...

    # --- Collective Communication ---
    def all_gather(self, arr: npt.ArrayLike, axis: int = 0) -> npt.ArrayLike:
        """Concatenates arrays from all devices along the specified axis."""
        ...

    def all_reduce(self, arr: npt.ArrayLike, op: str = 'sum') -> npt.ArrayLike:
        """Reduces arrays from all devices (e.g., sum) and broadcasts the result."""
        ...

    def all_to_all(self, arr: npt.ArrayLike, axis: int = 0) -> npt.ArrayLike:
        """Scatters chunks of the array to different devices."""
        ...

    # --- Model Lifecycle ---
    @abstractmethod
    def load_checkpoint(self, params: dict[str, npt.ArrayLike]) -> None:
        ...

    @abstractmethod
    def forward(self, x: npt.ArrayLike) -> npt.ArrayLike:
        ...

    @abstractmethod
    def backward(self, grads: npt.ArrayLike) -> dict[str, npt.ArrayLike]:
        ...
    
    def inference_loop(self, input_stream: Reader[npt.ArrayLike], output_stream: Writer[npt.ArrayLike]) -> None:
        for x in iter(input_stream):
            output_stream.write(self.forward(x))
```

## Unsharded Example

Let's start with an unsharded example on a single device. We will start with a 2 layers model with a `ReLU` activation in between. \\[\text{ReLU}(x W_0) W_1\\]

```python

def relu(x):
    return x * (x > 0)

class SingleDevice(ShardedEngine):

    def __init__(self, model_dim: int, hidden_dim: int):
        self.w0 = np.zeros((model_dim, hidden_dim), dtype=np.float32)
        self.w1 = np.zeros((hidden_dim, model_dim), dtype=np.float32)
        # Context tape to store activations for backward pass
        self.activations = []


    def load_checkpoint(self, params: dict[str, npt.ArrayLike]) -> None:
        # Load weights into local memory
        self.w0[...] = params['layer_0/weights'][...]
        self.w1[...] = params['layer_1/weights'][...]

    def forward(self, x: npt.ArrayLike) -> npt.ArrayLike:
        # 1. Save Input
        self.activations.append(x)
        
        # 2. Linear Layer 0
        # (Batch, Model) @ (Model, Hidden) -> (Batch, Hidden)
        z = np.einsum('bd,df->bf', x, self.w0)
        
        # 3. Activation
        self.activations.append(z) # Save pre-activation for the backward pass
        x = relu(z)
        
        # 4. Linear Layer 1
        # (Batch, Hidden) @ (Hidden, Model) -> (Batch, Model)
        out = np.einsum('bf,fd->bd', x, self.w1)
        return out

    def backward(self, grads: npt.ArrayLike) -> dict[str, npt.ArrayLike]:
        """
        grads: Incoming gradient dL/d(Output) of shape (Batch, Model_Dim)
        """
        # --- Backprop Layer 1 ---
        # Retrieve input to Layer 1 (Output of ReLU)
        # Shape: (Batch, Hidden)
        z = self.activations.pop()
        h_relu = relu(z)
        
        # dL/dW1 = h_relu.T @ grads
        w1_grad = np.einsum('bf,bd->fd', h_relu, grads)
        
        # Propagate gradient to h_relu: dL/dh = grads @ W1.T
        grads = np.einsum('bd,fd->bf', grads, self.w1)

        # --- Backprop ReLU ---
        # Apply derivative of ReLU: 1 if h > 0 else 0
        grads = grads * (z > 0)

        # --- Backprop Layer 0 ---
        # Retrieve input to Layer 0 (Original X)
        # Shape: (Batch, Model)
        x_input = self.activations.pop()
        
        # dL/dW0 = x_input.T @ grads
        w0_grad = np.einsum('bd,bf->df', x_input, grads)
        
        return {'layer_0/weights': w0_grad, 'layer_1/weights': w1_grad}
```
