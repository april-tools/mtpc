import os
import glob
import yaml
import torch
import numpy as np


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, f"number of tokens read does not match header? found {len(tokens)} but expected {ntok}"
    return tokens


class LocalDistributedDataLoader:
    def __init__(self, filename_pattern, model, B, T, process_rank, num_processes, device='cuda', split=None):
        # NOTE: We currently do not use model or split - added options just to make interface the same
        # as the huggingface models
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.device = device

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        # When using HF datasets, we need additional information
        # such as the PAD token id so that we can create the attention masks
        # and the max sequence length so that we make sure we do not accidentally
        # try to run the model with a different seq len from the serialized one
        self.config = self.load_config(filename_pattern)

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])
        return self

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def load_config(self, filename_pattern):
        # We assume the config file, if it exists, is called config.yaml
        dir_path = os.path.dirname(filename_pattern)
        conf_path = os.path.join(dir_path, 'config.yaml')
        try:
            with open(conf_path, 'r') as file:
                config = yaml.safe_load(file)
        except Exception:
            config = None
        return config

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position: self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets

        if self.config is None:
            # These are the datasets that do not have variable sized sequences
            # So attention mask is just a mask of ones
            attention_mask = torch.ones_like(x, dtype=torch.int32)
        else:
            raise NotImplementedError('Need to implement padding')

        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        if self.device == 'cuda':
            x, y, attention_mask = x.cuda(), y.cuda(), attention_mask.cuda()
        return dict(input_ids=x, labels=y, attention_mask=attention_mask)

    def seek(self, num_steps):
        # Move the dataloader forward num_steps
        B = self.B
        T = self.T
        for i in range(num_steps):
            self.current_position += B * T * self.num_processes
            if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
                self.advance()
