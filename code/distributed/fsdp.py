import torch
from torch import einsum
import torch.distributed as dist
import torch.multiprocessing as mp
import os

class MlpFsdp:

    def __init__(self, rank, d_model, d_ff, device):
        self.activations = []
        self.rank = rank
        self.d_model = d_model
        self.d_ff = d_ff
        # init the weights for the optimizer
        self.w_in = torch.zeros((d_model // dist.get_world_size(), d_ff), dtype=torch.float32, device=device)
        self.w_out = torch.zeros((d_ff, d_model // dist.get_world_size()), dtype=torch.float32, device=device)

    def load_checkpoint(self, params):
        """
        X[Bx, D] @ Win[Dx, F] @ Wout[F, Dx]
        """
        local_d_model = self.d_model // dist.get_world_size()
        start = local_d_model * self.rank
        end = start + local_d_model
        self.w_in[...] = params['layer_in/weights'][start:end, :]
        self.w_out[...] = params['layer_out/weights'][:, start:end]

    def forward(self, x):
        """
        Win[D, F] = AllGather_Dx(Win[Dx, F])
        Z[Bx, F] = X[Bx, D] @ Win[D, F]
        A[Bx, F] = Activation (Z)
        Wout[F, D] = AllGather_Dx(Wout[F, Dx])
        Out[Bx, D] = A[Bx, F] @ Wout[F, D]
        """
        self.activations = []
        F, D = self.d_ff, self.d_model

        # ALL-GATHER Win
        w_in_global = torch.zeros((D, F), dtype=torch.bfloat16, device=self.w_in.device) # bf16
        # both tensors just got casted so they are contiguous
        dist.all_gather_into_tensor(w_in_global, self.w_in.to(dtype=torch.bfloat16)) # we could cache Win as bf16

        self.activations.append(x)
        z = einsum('bd,df->bf', x, w_in_global)
        del w_in_global

        # ALL-GATHER Wout
        w_out_global = torch.zeros((D, F),  dtype=torch.bfloat16, device=self.w_out.device) #bf16
        w_out_local = self.w_out.t().to(dtype=torch.bfloat16) # we could cache Wout as bf16
        dist.all_gather_into_tensor(w_out_global, w_out_local)
        del w_out_local
        w_out_global = w_out_global.t() # bf16[F, D]

        self.activations.append(z)
        a = torch.nn.functional.relu(z)
        out = einsum('bf,fd->bd', a, w_out_global)
        del w_out_global

        return out

    def backward(self, out_grad):
        """
        dWout[F, D]{Ux} = A.T[F, Bx] @Bx dOut[Bx, D]
        dWout[F, Dx] = ReduceScatter_Dx(dWout[F, D]{Ux})
        
        Wout[F, D] = AllGather_Dx(Wout[F, Dx])
        dA[Bx, F] = dOut[Bx, D] @ Wout.T[D, F]
        dZ[Bx, F] = dA[Bx, F] * Act'(Z)[Bx, F]
        dWin[D, F]{Ux} = X.T[D, Bx] @Bx dZ[Bx, F]
        dWin[Dx, F] = ReduceScatter_Dx(dWin[D, F]{Ux})

        Win.T[D, F] = AllGather_Dx(Win.T[Dx, F])
        dX[Bx, D] = dZ[Bx, F] @ Win.T[F, D]
        """
        F, Dx = self.w_out.shape
        D = self.d_model

        z = self.activations.pop()
        a = torch.nn.functional.relu(z)
        w_out_grad = einsum('bf,bd->fd', a, out_grad)

        # REDUCE-SCATTER dWout
        # torch always reduces along the primary dimension
        w_out_grad_local = torch.zeros((Dx, F), dtype=torch.bfloat16, device=self.w_out.device) # bf16[Dx, F]
        w_out_grad = w_out_grad.transpose(0, 1).to(dtype=torch.bfloat16) # bf16[D, F]{Ux}
        dist.reduce_scatter_tensor(w_out_grad_local, w_out_grad, op=dist.ReduceOp.AVG) # averaging noise from different data samples
        del w_out_grad
        w_out_grad_local = w_out_grad_local.transpose(0, 1) # bf16[F, Dx]

        # ALL-GATHER Wout
        w_out = torch.zeros((D, F), dtype=torch.bfloat16, device=self.w_out.device) # bf16[D, F]
        w_out_local = self.w_out.t().to(dtype=torch.bfloat16) # [Dx, F]
        dist.all_gather_into_tensor(w_out, w_out_local)
        del w_out_local
        w_out = w_out.t() # bf16[F, D]

        a_grad = einsum('bd,fd->bf', out_grad, w_out)
        z_grad = a_grad * (z > 0)
        x = self.activations.pop()
        w_in_grad = einsum('bd,bf->df', x, z_grad) # bf16[D, F]{Ux}

        # REDUCE-SCATTER dWin
        w_in_grad_local = torch.zeros((Dx, F), dtype=torch.bfloat16, device=self.w_in.device) # bf16[Dx, F]
        w_in_grad = w_in_grad.contiguous()
        dist.reduce_scatter_tensor(w_in_grad_local, w_in_grad, op=dist.ReduceOp.AVG) # averaging noise from different data samples
        del w_in_grad

        # AL-GATHER Win
        w_in = torch.zeros((D, F), dtype=torch.bfloat16, device=self.w_in.device)
        w_in_local = self.w_in.to(dtype=torch.bfloat16)
        dist.all_gather_into_tensor(w_in, w_in_local)
        del w_in_local

        x_grad = einsum('bf,df->bd', z_grad, w_in)

        return {'layer_out/weights': w_out_grad_local, 'layer_in/weights': w_in_grad_local, 'input': x_grad}


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
    model = MlpFsdp(rank, D, F, device)
    model.load_checkpoint(params)

    # Data Sharding
    B_local = B // world_size
    start = rank * B_local
    end = start + B_local

    x = torch.randn(B, D, dtype=torch.bfloat16) # global unsharded input
    x_local = torch.zeros((B_local, D), dtype=torch.bfloat16, device=device)
    x_local[...] = x[start:end, :]
    out_local = model.forward(x_local)

    # simulated loss gradient (dLoss/dOut)
    grad_out = torch.randn(B, D, dtype=torch.bfloat16) # global unsharded gradients
    grad_out_local = torch.zeros((B_local, D), dtype=torch.bfloat16, device=device)
    grad_out_local[...] = grad_out[start:end, :]
    grads = model.backward(grad_out_local)

    # Verification
    if rank == 0:
        print(f"--- Simulation on {device_type.upper()} ---")
        print(f"Rank {rank}: TP Backward Complete.")
        # Check Shapes
        print(f"Grad Win: {grads['layer_in/weights'].shape} (Expected: {D//world_size}, {F})")
        print(f"Grad Wout: {grads['layer_out/weights'].shape} (Expected: {F}, {D//world_size})")
        print(f"Grad X:   {grads['input'].shape} (Expected: {B_local}, {D})")

    dist.destroy_process_group()

if __name__ == "__main__":
    WORLD_SIZE = 4
    mp.start_processes(worker_fn, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True, start_method="fork")
