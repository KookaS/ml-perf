# Jax vs PyTorch

While `Jax`'s and `PyTorch`'s APIs look similar, they handle compilation very differently. This matters a lot when writing code in either library and when thinking about performance.

## Tracing (Jax)

`Jax` uses the `tracing` approach. When we we compile a `Python` method using `jax.jit`, we set a global variable called the `Tracer`. When our `Python` code encounters a `Jax` method, it appends an instruction to the global `Tracer`. At the end of our function, the `Tracer` has a full graph of instructions that it finally compiles.

This means that:

- The only code that will be compiled will be the `Jax` methods we encountered during compilation.
- `If..else` statements and `for loops` are only evaluated at compile time and their evaluation will be constant at run time.
- Runtime dependent control flow has to be implemeted using `Jax` APIs like `jax.lax.cond` and `jax.lax.fori_loop`.

Another particularity is that `jax.jit` will compile your method for a specific set of input shapes and dtypes. Changing your input shape will force a recompilation of the program.

Furthermore, jitted methods are purely functional. We cannot mutate a value in-place. *Performance Note*: Although the API is functional (create new arrays), the compiler optimizes this into in-place updates under the hood, so you don't lose performance.

Let's illustrate what this means:

![image](./jax.png)

### Printing (Jax)

```python
import jax

@jax.jit
def add(a, b):
    print(a, b)
    return a + b

add(1, 2)
add(3, 4)
```

*stdout*

```plaintext
JitTracer<~int32[]> JitTracer<~int32[]>
```

The `print` statement is not a `Jax` method, so it only prints at compile time. We only have one `stdout line` even though we called the method twice because it only printed during compilation and the second call is cached. If we wanted to print actual runtime values, we would use `jax.debug.print`.

### Runtime If Statement (Jax)

```python
import jax

@jax.jit
def conditional(a, b):
    if a > b:
        return a
    return jnp.exp(b)

conditional(3, 4)
```

*stderr*

```plaintext
TracerBoolConversionError:
    Attempted boolean conversion of traced array with shape bool[].
```

We attempted to use a runtime value in an `if` statement, resulting in a compile-time error. We can fix this using `jax.lax.cond` to ensure that the `Tracer` knows about the `if` statement and compiles it.

```python
import jax
import jax.numpy as jnp

@jax.jit
def conditional(a, b):
    return jax.lax.cond(a > b, lambda: a, lambda: jnp.exp(b))

# (Using floats because a and exp(b) must have the same type)
conditional(1., 2.)
```

### Static Arguments

We can define static arguments to be passed to the method. These arguments will not be traced, however they can be used for control flow during compilation.

Let's look at this code:

```python
from functools import partial
import jax
import jax.numpy as jnp

@partial(jax.jit, static_argnames=('add_residuals',))
def linear_layer(x, w0, add_residuals: bool = False):
    y = x @ w0
    # add_residuals is static so it can be used in the `if` statement
    if add_residuals:
        return x + y
    return y

x = jnp.ones((32, 128))
w0 = jnp.ones((128, 128))
```

When we compile the method with `add_residuals = False`, the `Tracer` never sees the `x + y` operation, so it never gets compiled and the `Tracer` never knows this line of code existed. If you call the function again with `add_residuals = True`, Jax **MUST** recompile the whole function.

We can even pass functions or complex objects as static arguments!

```python
from typing import Callable

@partial(jax.jit, static_argnames=('activation',))
def linear_layer(x, w0, activation: Callable[[jax.Array], jax.Array] | None = None):
    y = x @ w0
    if activation:
        return activation(y)
    return y

x = jnp.ones((32, 128))
w0 = jnp.ones((128, 128))

linear_layer(x, w0, jax.nn.relu)
```

## Bytecode Interception (PyTorch)

`PyTorch`'s approach puts less weight on the developer. Any method that works in eager mode will also work with `torch.compile`. This is achieved by intercepting `Python`'s bytecode and dynamically modifying it right before execution. This throws all of `Jax`'s limitations out of the window.

Some `Python` operations cannot be compiled directly by `torch.compile`. For instance `print` or `numpy calls`. When `torch.compile` encounters these operations, it falls back to `Python`; we call this a `Graph Break`. `Graph Breaks` are slow and should be kept to a minimum to reach maximum performance.

![image](./torch.png)

### Printing (PyTorch)

```python
import torch

@torch.compile
def flexible_function(x):
    # 1. This math is captured into Graph A (Fast)
    y = x * 2
    
    # 2. GRAPH BREAK! 
    # The compiler pauses. Python executes this print.
    print(f"Python sees the value: {y[0]}")

    # 3. Compilation resumes. This math is captured into Graph B (Fast)
    z = y + 10
    return z

x = torch.randn(5)
flexible_function(x)
```

*stdout*

```python
Python sees the value: -1.1428250074386597
```

We print the runtime value at the cost of a graph break.

### Runtime If Statement (PyTorch)

```python
import torch

@torch.compile
def conditional(a, b):
    if a.sum() > b.sum():
        return a
    return torch.exp(b)

a = torch.randn(5)
b = torch.randn(5)

conditional(a, b)
```

This code compiles without errors unlike `Jax`. However, it introduces a `Graph Break`. We can fix it by staying in graph with an API like `torch.where`.
