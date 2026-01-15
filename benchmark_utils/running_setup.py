import os
import torch
import torch.distributed as t_dist


def get_running_setup():

    # Use submitit helpers to setup distributed training easily.
    try:
        import submitit
        submitit.helpers.TorchDistributedEnvironment().export()
        ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
    except (ImportError, RuntimeError):
        ddp = False
    if ddp:
        print("Running in Distributed Data Parallel (DDP) mode")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        assert torch.cuda.is_available()
        # TorchDistributedEnvironment sets the visible devices to the
        # current rank, so we can use the default device.
        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        t_dist.init_process_group(backend="nccl", device_id=device)
        dist = t_dist
    else:
        rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dist = None

    return dist, rank, world_size, device
