import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from benchopt import BaseSolver
from tqdm.auto import tqdm

# -----------------------------------------------------------------------------
# Muon optimizer implementation


def zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration to compute the 0-th power/orthogonalize G."""
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T

    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon optimizer implementation.

    Muon (Jordan et al. 2024) uses spectral norm (Newton-Schulz) for matrix parameters (2D+) and
    AdamW for vector parameters (1D).

    Args:
        params:
            Iterable of parameters to optimize or dicts defining parameter
            groups
        lr: float, optional (default: 1e-2)
            Learning rate for matrix parameters
        momentum: float, optional (default: 0.95)
            Momentum factor for matrix parameters (traditional momentum)
        nesterov: bool, optional (default: True)
            Whether to use Nesterov momentum
        ns_steps: int, optional (default: 5)
            Number of Newton-Schulz iterations
        adamw_lr: float, optional (default: 1e-3)
            Learning rate for AdamW (vector parameters)
        adamw_betas: tuple, optional (default: (0.9, 0.999))
            Betas for AdamW
        adamw_eps: float, optional (default: 1e-8)
            Epsilon for AdamW
        adamw_wd: float, optional (default: 0.0)
            Weight decay for AdamW
    """

    def __init__(
        self,
        params,
        lr=1e-2,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_lr=1e-3,
        adamw_betas=(0.9, 0.999),
        adamw_eps=1e-8,
        adamw_wd=0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum >= 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)

        # Compile Newton-Schulz function once
        self.newton_schulz5 = torch.compile(zeropower_via_newtonschulz5)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            adamw_lr = group["adamw_lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            wd = group["adamw_wd"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                G = p.grad
                state = self.state[p]

                # Matrix parameters: use spectral norm (Newton-Schulz)
                if p.dim() >= 2:
                    # Initialize momentum buffer
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(G)

                    # Apply spectral norm via Newton-Schulz
                    G_normalized = self.newton_schulz5(
                        G.reshape(G.shape[0], -1), steps=ns_steps
                    ).view(G.shape)

                    # Scale by sqrt(d_out / d_in)
                    d_out = G.shape[0]
                    d_in = G.shape[1:].numel()
                    G_normalized = G_normalized * (d_out / d_in) ** 0.5

                    # Update momentum
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(G_normalized)

                    # Apply update with optional Nesterov
                    if nesterov:
                        update = G_normalized + momentum * buf
                    else:
                        update = buf

                    p.data.add_(update, alpha=-lr)

                # Vector parameters: use AdamW
                else:
                    # Initialize AdamW state
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    state["step"] += 1

                    # Weight decay
                    if wd != 0:
                        p.data.mul_(1 - adamw_lr * wd)

                    # Update biased first and second moments
                    exp_avg.mul_(beta1).add_(G, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(G, G, value=1 - beta2)

                    # Bias correction
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = adamw_lr / bias_correction1

                    # Compute update
                    denom = (exp_avg_sq.sqrt() / bias_correction2**0.5).add_(eps)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)


# learning rate schedule: stable then decay to 0
def get_lr(step, num_steps, cooldown_frac=0.4):
    x = step / num_steps  # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        return (1 - x) / cooldown_frac


# The benchmark solvers must be named `Solver` and
# inherit from `BaseSolver` for `benchopt` to work properly.
class Solver(BaseSolver):
    # Name to select the solver in the CLI and to display the results.
    name = "Muon"

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    # All parameters 'p' defined here are available as 'self.p'.
    parameters = {
        "learning_rate": [0.02],
        "momentum": [0.95],
        "nesterov": [True],
        "adamw_lr": [3e-4],
        "adamw_wd": [0.0],
        "cooldown_frac": [0.4],
        "num_steps": [6200],
        "batch_size": [64],
        "slurm_nodes": [1, 2],
    }
    slurm_params = {
        "slurm_gres": "gpu:4",
        "slurm_ntasks_per_node": 4,
    }

    # List of packages needed to run the solver.
    requirements = []

    sampling_strategy = "callback"

    def set_objective(self, train_dataloader, model):
        # Use submitit helpers to setup distributed training easily.
        try:
            import submitit

            submitit.helpers.TorchDistributedEnvironment().export()
            ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
        except (ImportError, RuntimeError):
            ddp = False
        if ddp:
            print("Running in Distributed Data Parallel (DDP) mode")
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            assert torch.cuda.is_available()
            # TorchDistributedEnvironment sets the visible devices to the
            # current rank, so we can use the default device.
            device = torch.device("cuda", 0)
            torch.cuda.set_device(device)
            dist.init_process_group(backend="nccl", device_id=device)
            self.dist = dist
        else:
            self.rank = 0
            self.world_size = 1
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.dist = None
        model = model.to(device=device)
        model.device = device  # store the device in the model
        self.train_dataloader = train_dataloader

        # use mixed precision if cuda is available
        self.ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )

        # Torch compile the model and the optimizer step function
        self.model = torch.compile(model, dynamic=False, fullgraph=True)
        Muon.step = torch.compile(torch.no_grad()(Muon.step))

    def __del__(self):
        # Clean up communication resources
        if getattr(self, "dist", None) is not None:
            self.dist.destroy_process_group()

    def get_next(self, stop_val):
        return stop_val + 250

    def warm_up(self):
        n_iter = self.num_steps
        self.num_steps = 10
        self.run_once(stop_val=10)
        self.num_steps = n_iter

    def run(self, cb):
        # Create Muon optimizer (it will automatically handle matrix vs vector params)
        self.optimizer = Muon(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            nesterov=self.nesterov,
            adamw_lr=self.adamw_lr,
            adamw_wd=self.adamw_wd,
        )

        train_loader = self.train_dataloader.get_distributed_data_generator(
            batch_size=self.batch_size * 1024 * self.world_size,
            rank=self.rank,
            world_size=self.world_size,
        )

        if self.dist is not None:
            self.dist.barrier()  # wait for all processes to be ready

        step = 0
        with tqdm(total=self.num_steps, desc="Training") as progress:
            while cb():
                self.model.train()

                step += 1
                progress.update()
                if step == self.num_steps:
                    break

                # Zero gradients
                self.optimizer.zero_grad()

                data = next(train_loader)
                with self.ctx:
                    loss, *_ = self.model(*data)
                loss.backward()

                if self.dist is not None:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            self.dist.all_reduce(param.grad, op=self.dist.ReduceOp.AVG)

                # determine and set the learning rate for this iteration
                scale_lr = get_lr(step, self.num_steps, self.cooldown_frac)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate * scale_lr
                    param_group["adamw_lr"] = self.adamw_lr * scale_lr

                # step the optimizer
                self.optimizer.step()

    def get_result(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for all operations to finish
        return dict(model=self.model, dist=self.dist)
