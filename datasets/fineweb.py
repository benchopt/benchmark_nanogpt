from benchopt import BaseDataset
from benchopt.config import get_data_path

import torch
import glob
from pathlib import Path
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

from benchmark_utils.model_gpt2 import GPT, GPTConfig


def download_data(data_dir, n_chunks=104):

    # Download the GPT-2 tokens of Fineweb10B from huggingface. This
    # saves about an hour of startup time compared to regenerating them.
    for i in range(n_chunks):
        chunk = "val" if i == 0 else "train"
        fname = f"fineweb_{chunk}_{i:06d}.bin"
        if not (data_dir / fname).exists():
            hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", filename=fname,
                            repo_type="dataset", local_dir=data_dir)


def _load_data_shard(file):
    # header is 256 int32
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        pin_memory = torch.cuda.is_available()
        # avoid pin_memory copy by @YouJiacheng
        tokens = torch.empty(
            num_tokens, dtype=torch.uint16, pin_memory=pin_memory
        )
        f.seek(256 * 4)
        # avoid bytes->array copy by @YouJiacheng
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, (
            "number of tokens read does not match header"
        )
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, max_tokens=None):
        self.filename_pattern = filename_pattern
        self.max_tokens = max_tokens
        self.seq_len = 1024

    def get_distributed_data_generator(self, batch_size, rank=0, world_size=1):
        files = [
            Path(file) for file in sorted(glob.glob(self.filename_pattern))
        ]
        effective_batch_size = batch_size * world_size * self.seq_len
        if self.max_tokens is not None:
            assert (self.max_tokens % effective_batch_size) == 0, (
                "max_tokens must be multiple of effective batch size"
            )

        # Compute local batch size per process
        # We use sequence length 1024 and load the token stream as a flat array
        local_batch_size = batch_size * self.seq_len

        # use itertools.cycle(files) for multi-epoch training
        file_iter = iter(files)
        tokens, pos = _load_data_shard(next(file_iter)), 0

        cuda_args = (
            dict(device="cuda", non_blocking=True)
            if torch.cuda.is_available() else {}
        )
        if self.max_tokens is not None:
            progress = tqdm(total=self.max_tokens, desc="Validation")
        while True:
            if pos + effective_batch_size + 1 >= len(tokens):
                tokens, pos = _load_data_shard(next(file_iter)), 0
            buf = tokens[pos + rank * local_batch_size:][:local_batch_size + 1]
            # no sync on host side;
            inputs = (
                buf[:-1].to(dtype=torch.int32, **cuda_args)
                .view(-1, self.seq_len)
            )
            targets = (
                buf[1:].to(dtype=torch.int64, **cuda_args)
                .view(-1, self.seq_len)
            )
            pos += effective_batch_size
            if self.max_tokens is not None:
                progress.update(effective_batch_size)
            yield inputs, targets
            if self.max_tokens is not None:
                if pos >= self.max_tokens:
                    break


class Dataset(BaseDataset):

    name = "Fine-web"
    parameters = {
        'n_chunks': [4],
        'debug': [False]
    }

    # List of packages needed to run the dataset. See the corresponding
    # section in objective.py
    requirements = ["huggingface_hub"]

    def get_data(self):
        print("Get data")

        data_dir = get_data_path("fineweb10B")
        download_data(data_dir, n_chunks=self.n_chunks)

        # from scratch (random weights)
        config = GPTConfig(
            vocab_size=50257, n_layer=12, n_head=6, n_embd=768,
            # max_seq_len=4*64*1024 - This is for Rotary Positional Embedding
        )
        model = GPT(config)

        # The dictionary defines the keyword arguments for `Objective.set_data`
        return dict(
            train_dataloader=DistributedDataLoader(
                str(data_dir / "fineweb_train_*.bin")
            ),
            val_dataloader=DistributedDataLoader(
                str(data_dir / "fineweb_val_*.bin"),
                max_tokens=10485760 // 20 if self.debug else 10485760
            ),
            model=model,
        )
