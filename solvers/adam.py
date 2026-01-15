from benchopt import BaseSolver

from contextlib import nullcontext

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from benchmark_utils.lr_scheduler import get_lr
from benchmark_utils.running_setup import get_running_setup


class Solver(BaseSolver):

    name = 'Adam'

    parameters = {
        'learning_rate': [1e-3],
        'weight_decay': [1e-4],
        'num_steps': [6200],
        'batch_size': [64],
        "slurm_nodes": [1, 2],
        "sin_init": [True],
    }
    slurm_params = {
        "slurm_gres": "gpu:4",
        "slurm_ntasks_per_node": 4,
    }

    sampling_strategy = 'callback'

    def set_objective(self, train_dataloader, model):

        # Setup distributed training if needed
        self.dist, self.rank, self.world_size, device = get_running_setup()

        if self.sin_init:
            print("Using sinusoidal initialization")
            from benchmark_utils.sin_init import sinusoidal_
            model.init_func = sinusoidal_
            model.initialize_weights(seed=42)

        model = model.to(device=device)
        model.device = device  # store the device in the model
        self.train_dataloader = train_dataloader

        # use mixed precision if cuda is available
        self.ctx = (
            torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
            if torch.cuda.is_available() else nullcontext()
        )

        # Torch compile the model and the optimizer step function
        self.model = torch.compile(model, dynamic=False, fullgraph=True)
        AdamW.step = torch.compile(torch.no_grad(AdamW.step))

    def __del__(self):
        # Clean up communication resources
        if getattr(self, "dist", None) is not None:
            self.dist.destroy_process_group()

    def get_next(self, stop_val):
        return stop_val + 250

    def warm_up(self):
        self.run_once(stop_val=10)

    def run(self, cb):

        # configure the optimizer
        # List all parameters that require gradients
        param_dict = {
            pn: p for pn, p in self.model.named_parameters()
            if p.requires_grad
        }

        # create optim groups. Any parameters that is 2D will be weight
        # decayed, otherwise no. i.e. all weight tensors in
        # matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        # Create AdamW optimizer
        # TODO: consider using a ZeroRedundancyOptimizer
        self.optimizer = AdamW(
            optim_groups,
            lr=torch.tensor(self.learning_rate),
            betas=(0.9, 0.95),
            fused=True
        )

        train_loader = self.train_dataloader.get_distributed_data_generator(
            batch_size=self.batch_size,
            world_size=self.world_size,
            rank=self.rank,
        )

        if self.dist is not None:
            self.dist.barrier()  # wait for all processes to be ready

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

                # determine and set the learning rate for this iteration
                scale_lr = get_lr(step, self.num_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = torch.tensor(
                        self.learning_rate * scale_lr
                    )
                # step the self.optimizer
                self.optimizer.step()

    def get_result(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # wait for all operations to finish
        return dict(model=self.model, dist=self.dist)
