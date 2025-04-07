import torch
import numpy as np

from transformers import AutoTokenizer
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node


class HFDistributedDataLoader:
    def __init__(self, hf_dataset, model, B, T, process_rank, num_processes, device, split='train'):
        self.hf_dataset = hf_dataset
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.device = device
        self.split = split

        self.tokenizer = AutoTokenizer.from_pretrained(model,
                model_max_length=self.T,
                padding_side='right',
                use_fast=True)

        self.reset()

    def reset(self):
        self.dataset = load_dataset(self.hf_dataset, split=self.split, streaming=True)
        self.dataset = split_dataset_by_node(self.dataset, rank=self.process_rank, world_size=self.num_processes)
        self.dataset = self.dataset.batch(self.B)
        self.dataset = self.dataset.map(lambda x: self.tokenizer(x['text']))

    def next_batch(self):
        batch = next(iter(self.dataset.take(1)))
        x = batch['input_ids'][:, :-1]
        y = batch['input_ids'][:, 1:]
        attention_mask = batch['attention_mask'][:, :-1]
        if self.device == 'cuda':
            x, y, attention_mask = x.cuda(), y.cuda(), attention_mask.cuda()
        return dict(input_ids=x, labels=y, attention_mask=attention_mask)

    def seek(self, num_steps):
        self.dataset = self.dataset.skip(num_steps)
