import torch
from torch import einsum
import torch.distributed as dist
import torch.multiprocessing as mp
import os

class MlpTp:

    def __init__(self, rank, d_model, d_ff, device):
        self.activations = []
        self.rank = rank
        self.d_model = d_model
        self.d_ff = d_ff
        # init the weights for the optimizer
        self.w_in = torch.zeros((d_model, d_ff // dist.get_world_size()), dtype=torch.float32, device=device)
        self.w_out = torch.zeros((d_ff // dist.get_world_size(), d_model), dtype=torch.float32, device=device)

    def load_checkpoint(self, params):
        """
        X[B, D] @ Win[D, Fy] @y Wout[Fy, D]
        """
        local_d_ff = self.d_ff // dist.get_world_size()
        start = local_d_ff * self.rank
        end = start + local_d_ff
        self.w_in[...] = params['layer_in/weights'][:, start:end]
        self.w_out[...] = params['layer_out/weights'][start:end, :]
    
    def forward(self, x):
        """
        Z[B, Fy] = X[B, D] @ Win[D, Fy]
        A[B, Fy] = Activation (Z)
        Out[B, D]{Uy} = A[B, Fy] @ Wout[Fy, D]
        Out[B, D] = AllReduce(Out[B, D]{Uy})
        """
        self.activations = []
        self.activations.append(x)
        z = einsum('bd,df->bf', x, self.w_in.to(dtype=torch.bfloat16)) # weights bf16 could be stored in activations
        self.activations.append(z)
        a = torch.nn.functional.relu(z)
        out = einsum('bf,fd->bd', a, self.w_out.to(dtype=torch.bfloat16)) # weights bf16 could be stored in activations
        out = out.contiguous()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    def backward(self, out_grad):
        """
        dWout[Fy, D] = A.T[Fy, B] @ dOut[B, D]
        
        dA[B, Fy] = dOut[B, D] @ Wout.T[D, Fy]
        dZ[B, Fy] = dA[B, Fy] * Act'(Z)[B, Fy]
        dWin[D, Fy] = X.T[D, B] @ dZ[B, Fy]

        dX[B, D]{Uy} = dZ[B, Fy] @ Win.T[Fy, D]
        dX[B, D] = AllReduce(dX[B, D]{Uy})
        """
        z = self.activations.pop()
        a = torch.nn.functional.relu(z)
        w_out_grad = einsum('bf,bd->fd', a, out_grad)

        a_grad = einsum('bd,fd->bf', out_grad, self.w_out.to(dtype=torch.bfloat16))
        z_grad = a_grad * (z > 0)
        x = self.activations.pop()
        w_in_grad = einsum('bd,bf->df', x, z_grad)

        x_grad = einsum('bf,df->bd', z_grad, self.w_in.to(dtype=torch.bfloat16))
        x_grad = x_grad.contiguous() # bf16
        dist.all_reduce(x_grad, op=dist.ReduceOp.SUM) # chain rule, summing partial parts of the model
        
        return {'layer_out/weights': w_out_grad, 'layer_in/weights': w_in_grad, 'input': x_grad}


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
        'layer_in/weights': torch.randn(D, F, dtype=torch.float32),
        'layer_out/weights': torch.randn(F, D, dtype=torch.float32),
    }
    model = MlpTp(rank, D, F, device)
    model.load_checkpoint(params)

    x = torch.randn(B, D, dtype=torch.bfloat16, device=device)
    out = model.forward(x)

    # simulated loss gradient (dLoss/dOut)
    grad_out = torch.randn(B, D, dtype=torch.bfloat16, device=device)
    grads = model.backward(grad_out)

    # Verification
    if rank == 0:
        print(f"--- Simulation on {device_type.upper()} ---")
        print(f"Rank {rank}: TP Backward Complete.")
        # Check Shapes
        print(f"Grad Win: {grads['layer_in/weights'].shape} (Expected: {D}, {F//world_size})")
        print(f"Grad Wout: {grads['layer_out/weights'].shape} (Expected: {F//world_size}, {D})")
        print(f"Grad X:   {grads['input'].shape} (Expected: {B}, {D})")

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 4
    mp.spawn(worker_fn, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
