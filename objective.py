from benchopt import BaseObjective, safe_import_context

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import torch


# The benchmark objective must be named `Objective` and
# inherit from `BaseObjective` for `benchopt` to work properly.
class Objective(BaseObjective):

    # Name to select the objective in the CLI and to display the results.
    name = "Deep Learning Optimization with NanoGPT"

    # URL of the main repo for this benchmark.
    url = "https://github.com/tomMoral/benchmark_nanogpt"

    requirements = ["pytorch", "tqdm"]

    # Minimal version of benchopt required to run this benchmark.
    # Bump it up if the benchmark depends on a new feature of benchopt.
    min_benchopt_version = "1.7"

    def set_data(self, train_dataloader, val_dataloader, model):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model

    def evaluate_result(self, model, dist=None):
        model.eval()
        val_batch_size = 64  # Batch of 64 for validation
        if dist is not None:
            # In distributed mode, we use the distributed data generator
            rank, size = dist.get_rank(), dist.get_world_size()
            val_loader = self.val_dataloader.get_distributed_data_generator(
                batch_size=val_batch_size, rank=rank, world_size=size
            )
        else:
            # In non-distributed mode, we use the regular data generator
            val_loader = self.val_dataloader.get_distributed_data_generator(
                batch_size=val_batch_size, rank=0, world_size=1
            )

        with torch.no_grad():
            # Compute the validation loss
            val_loss, n_batches = 0.0, 0
            for data in val_loader:
                loss, *_ = self.model(*data)
                val_loss += loss.item()
                n_batches += 1
            val_loss /= n_batches

            if dist is not None:
                # Average the validation loss across all processes
                val_loss_tensor = torch.tensor(val_loss, device=model.device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
                val_loss = val_loss_tensor.item()

        del val_loader

        # This method can return many metrics in a dictionary. One of these
        # metrics needs to be `value` for convergence detection purposes.
        return dict(
            value=val_loss,
        )

    def get_one_result(self):
        # Return one solution. The return value should be an object compatible
        # with `self.evaluate_result`. This is mainly for testing purposes.
        return dict(model=self.model)

    def get_objective(self):
        # Define the information to pass to each solver to run the benchmark.
        # The output of this function are the keyword arguments
        # for `Solver.set_objective`. This defines the
        # benchmark's API for passing the objective to the solver.
        # It is customizable for each benchmark.
        return dict(
            train_dataloader=self.train_dataloader,
            model=self.model,
        )
