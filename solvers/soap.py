from benchopt import BaseSolver

import os
from contextlib import nullcontext

from tqdm.auto import tqdm

import torch
import torch.distributed as dist

from benchmark_utils.soap import SOAP


# learning rate schedule: stable then decay
def get_lr(step, num_step, cooldown_frac=0.4):
    x = step / num_step  # progress in training
    assert 0 <= x < 1
    if x < 1 - cooldown_frac:
        return 1.0
    else:
        return (1 - x) / cooldown_frac


class Solver(BaseSolver):
    name = "SOAP"

    parameters = {
        "learning_rate": [1e-4],
        "weight_decay": [5e-3],
        "num_steps": [6200],
        "batch_size": [64],
        "precondition_frequency": [10],
        "max_precond_dim": [10000],
        "merge_dims": [False],
        "precondition_1d": [False],
        "normalize_grads": [False],
        "correct_bias": [True],
        "slurm_nodes": [1, 2],
    }
    slurm_params = {
        "slurm_gres": "gpu:4",
        "slurm_ntasks_per_node": 4,
    }

    requirements = []

    sampling_strategy = "callback"

    def set_objective(self, train_dataloader, model):
        try:
            import submitit

            submitit.helpers.TorchDistributedEnvironment().export()
            ddp = int(os.environ.get("RANK", -1)) != -1
        except (ImportError, RuntimeError):
            ddp = False
        if ddp:
            print("Running in Distributed Data Parallel (DDP) mode")
            self.rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            assert torch.cuda.is_available()
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
        model.device = device
        self.train_dataloader = train_dataloader

        self.ctx = (
            torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
            if torch.cuda.is_available()
            else nullcontext()
        )

        self.model = torch.compile(model, dynamic=False, fullgraph=True)
        SOAP.step = torch.compile(torch.no_grad(SOAP.step))

    def __del__(self):
        if getattr(self, "dist", None) is not None:
            self.dist.destroy_process_group()

    def get_next(self, stop_val):
        return stop_val + 250

    def warm_up(self):
        self.run_once(stop_val=10)

    def run(self, cb):
        param_dict = {
            pn: p for pn, p in self.model.named_parameters() if p.requires_grad
        }
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        self.optimizer = SOAP(
            optim_groups,
            lr=torch.tensor(self.learning_rate),
            betas=(0.95, 0.95),
            precondition_frequency=self.precondition_frequency,
            max_precond_dim=self.max_precond_dim,
            merge_dims=self.merge_dims,
            precondition_1d=self.precondition_1d,
            normalize_grads=self.normalize_grads,
            correct_bias=self.correct_bias,
        )

        train_loader = self.train_dataloader.get_distributed_data_generator(
            batch_size=self.batch_size,
            world_size=self.world_size,
            rank=self.rank,
        )

        if self.dist is not None:
            self.dist.barrier()

        step = 0
        with tqdm(total=self.num_steps, desc="Training") as progress:
            while cb():
                self.model.train()
                self.optimizer.zero_grad(set_to_none=True)

                step += 1
                progress.update()
                if step == self.num_steps:
                    break

                data = next(train_loader)
                with self.ctx:
                    loss, *_ = self.model(*data)
                loss.backward()
                if self.dist is not None:
                    for param in self.model.parameters():
                        self.dist.all_reduce(
                            param.grad, op=self.dist.ReduceOp.AVG
                        )

                scale_lr = get_lr(step, self.num_steps)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = torch.tensor(
                        self.learning_rate * scale_lr
                    )

                self.optimizer.step()

    def get_result(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return dict(model=self.model, dist=self.dist)
