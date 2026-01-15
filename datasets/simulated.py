import torch
from benchopt import BaseDataset

from benchmark_utils.sin_init import sinusoidal_


class DataLoader:
    def __init__(self, train=True):
        self.train = train
        self.X = torch.randn(32, 10)

    def get_distributed_data_generator(self, batch_size, rank, world_size):
        # Simulate a data loader that yields batches of data
        if self.train:
            # For training, we yield the data in batches
            while True:
                yield self.X,
        else:
            # For validation, we yield the entire dataset
            yield self.X,


def initialize_weights(model, seed=42):
    init_rng = torch.Generator()
    init_rng.manual_seed(seed)
    init_func = getattr(model, 'init_func', torch.nn.init.normal_)
    if isinstance(model, torch.nn.Linear):
        init_func(model.weight, mean=0.0, std=0.02, generator=init_rng)
        if model.bias is not None:
            torch.nn.init.zeros_(model.bias)


# All datasets must be named `Dataset` and inherit from `BaseDataset`
class Dataset(BaseDataset):

    # Name to select the dataset in the CLI and to display the results.
    name = "Simulated"

    def get_data(self):

        model = torch.nn.Linear(10, 1)
        model.init_func = torch.nn.init.normal_
        model.initialize_weights = (
            lambda seed: initialize_weights(model, seed=seed)
        )

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            train_dataloader=DataLoader(train=True),
            val_dataloader=DataLoader(train=False),
            model=model
        )
