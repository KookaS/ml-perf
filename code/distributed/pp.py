import torch
from torch import einsum
import torch.distributed as dist
import torch.multiprocessing as mp
import os

class MlpPp:

    def __init__(self, rank, batch_size, d_model, d_ff, device):
        self.activations = []
        self.rank = rank
        self.batch_size = batch_size
        self.d_model = d_model
        self.d_ff = d_ff
        # init the weights for the optimizer
        if rank % 2 == 0:
            self.weights = torch.zeros((d_model, d_ff), dtype=torch.float32, device=device)
        else:
            self.weights = torch.zeros((d_ff, d_model), dtype=torch.float32, device=device)

    def load_checkpoint(self, params):
        """
        X[B, D] @ Win[L_z, D, F][i] @y Wout[L_z, F, D][i]
        L_z is layer sharded by Z
        i microbatch
        """
        self.weights[...] = params[f'layer_{self.rank}/weights']

    def forward(self, x):
        """
        Z[B, F] = X[B, D] @ Win[D, F] (layer 0 --> send Z)

        A[B, F] = Activation (Z) (layer 1)
        Out [B, D] = A[B, F] @ Wout[F, D] (layer 1)
        """
        self.activations = []
        if self.rank % 2 == 0:
            self.activations.append(x) # store input
            z = einsum('bd,df->bf', x, self.weights.to(dtype=torch.bfloat16))
            dist.send(z, dst=self.rank + 1)
            return None
        else:
            x = torch.zeros((self.batch_size,self.d_ff), dtype=torch.bfloat16, device=self.weights.device)
            dist.recv(x, src=self.rank - 1)
            self.activations.append(x) # store pre-activation
            x = torch.nn.functional.relu(x)
            self.activations.append(x) # store post-activation
            return einsum('bf,fd->bd', x, self.weights.to(dtype=torch.bfloat16))

    def backward(self, out_grad):
        """
        dWout[F, D] = A.T[F, B] @ dOut[B, D] (layer 1)
        
        dA[B, F] = dOut[B, D] @ Wout.T[D, F] (layer 1)
        dZ[B, F] = dA[B, F] * Act'(Z)[B, F] (layer 1 --> send dZ)
        dWin[D, F] = X.T[D, B] @ dZ[B, F] (layer 0)

        dX[B, F] = dZ[B, F] @ Win.T[F, D] (layer 0)
        """
        if self.rank % 2 == 0:
            z_grad = torch.zeros((self.batch_size, self.d_ff), dtype=torch.bfloat16, device=self.weights.device)
            dist.recv(z_grad, src=self.rank + 1)
            x = self.activations.pop()
            w_in_grad = einsum('bd,bf->df', x, z_grad)

            x_grad = einsum('bf,df->bd', z_grad, self.weights.to(dtype=torch.bfloat16))
            return {'layer_0/weights': w_in_grad, 'input': x_grad}
        else:
            a = self.activations.pop()
            w_out_grad = einsum('bf,bd->fd', a, out_grad)

            a_grad = einsum('bd,fd->bf', out_grad, self.weights.to(dtype=torch.bfloat16))
            z = self.activations.pop()
            z_grad = a_grad * (z > 0)
            dist.send(z_grad, dst=self.rank - 1)
            return {'layer_1/weights': w_out_grad}


# --- The Runner ---
def worker_fn(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    device_type = "cpu"
    device = f"{device_type}:{rank}"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    B, D, F = 8, 64, 256

    # Model Init
    torch.manual_seed(42)
    params = {
        'layer_0/weights': torch.randn(D, F, dtype=torch.float32),
        'layer_1/weights': torch.randn(F, D, dtype=torch.float32),
    }
    model = MlpPp(rank, B, D, F, device)
    model.load_checkpoint(params)

    x = torch.randn(B, D, dtype=torch.bfloat16, device=device)
    out = model.forward(x) # None or Tensor if last

    # simulated loss gradient (dLoss/dOut)
    grad_out = torch.randn(B, D, dtype=torch.bfloat16, device=device)
    grads = model.backward(grad_out) # None or Tensor if first

    # Verification - print sequentially by rank
    for printing_rank in range(world_size):
        if rank == printing_rank:
            print(f"--- Simulation on {device_type.upper()} Rank {rank} ---")
            
            if rank == 1:
                if out is not None:
                    print(f"  Output Shape {out.shape} (Expected: {B, D})")
            
            if 'input' in grads:
                print(f"  Grad X: {grads['input'].shape} (Expected: {B, D})")
            
            for key, val in grads.items():
                if 'weights' in key:
                    print(f"  Grad {key}: {val.shape}")
            
            print("-" * 30)
        
        dist.barrier()  # Wait for each rank to finish printing

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 2
    mp.start_processes(worker_fn, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True, start_method="fork")
