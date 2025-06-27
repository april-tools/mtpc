import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import disable_caching
from datasets.distributed import split_dataset_by_node


if int(os.environ.get('HF_CACHE_ACTIVE', 1)) != 1:
    disable_caching()


class HFDistributedDataLoader(object):
    """A generic Huggingfaced Dataloader that is compatible with DDP.

    To override this class, see other examples in the folder.
    """

    def __init__(
        self,
        hf_dataset: str,
        hf_model: str,
        B: int | None,
        T: int,
        process_rank: int,
        num_processes: int,
        device: str = "cuda",
        split="train",
        as_iterable: bool = True,
        shuffle: bool = True,
    ):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.hf_model = hf_model
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.device = device
        self.split = split
        self.as_iterable = as_iterable
        self.shuffle = shuffle
        self.features = None
        self.num_proc = int(os.environ.get('HF_DATASETS_NUM_PROC', 1))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model,
            model_max_length=self.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

    @property
    def has_process(self):
        try:
            self.process(None)
        except NotImplementedError:
            return False
        except Exception:
            return True
        return True

    @property
    def has_filter(self):
        try:
            self.filter(None)
        except NotImplementedError:
            return False
        except Exception:
            return True
        return True

    def reset(self):
        self.dataset = self.load_dataset()
        if self.shuffle:
            # TODO: Below should change if we use multiple epochs
            self.dataset = self.dataset.shuffle(42)
        if self.as_iterable:
            self.dataset = self.dataset.to_iterable_dataset()
            self.dataset = self.dataset.with_format('torch')
            if self.has_process:
                self.dataset = self.dataset.map(self.process)
            if self.has_filter:
                self.dataset = self.dataset.filter(self.filter)
            self.dataset = split_dataset_by_node(
                self.dataset, rank=self.process_rank, world_size=self.num_processes
            )
            if self.B is not None:
                self.dataset = self.dataset.batch(self.B, drop_last_batch=True)
        else:
            self.dataset = self.dataset.with_format('torch')
            # num_proc can only be used when not using iterable dataset
            if self.has_process:
                self.dataset = self.dataset.map(self.process, num_proc=self.num_proc)
            if self.has_filter:
                self.dataset = self.dataset.filter(function=self.filter, num_proc=self.num_proc)
            self.dataset = split_dataset_by_node(
                self.dataset, rank=self.process_rank, world_size=self.num_processes
            )
            if self.B is not None:
                self.dataset = self.dataset.batch(self.B, num_proc=self.num_proc, drop_last_batch=True)
        self.dataset_iterator = None
        return self

    def load_dataset(self):
        return load_dataset(self.hf_dataset, split=self.split, num_proc=self.num_proc, features=self.features, token=os.getenv('HF_TOKEN'))

    @property
    def model_max_length(self):
        # use max_length=T + 1 because we use input_ids[:, :-1] and labels[:, 1:]
        return self.T + 1

    def process(self, x):
        raise NotImplementedError()

    def filter(self, x):
        raise NotImplementedError()

    def next_batch(self):
        if self.dataset_iterator is None:
            self.dataset_iterator = iter(self.dataset)
        fields = next(self.dataset_iterator)
        batch = dict()
        for k in fields:
            # Filter outputs to only include needed
            if k in ("input_ids", "labels", "attention_mask"):
                if isinstance(fields[k], list):
                    fields[k] = torch.cat(fields[k])
                if self.device == "cuda":
                    batch[k] = fields[k].cuda()
                else:
                    batch[k] = fields[k]
        return batch

    def seek(self, num_steps):
        self.reset()
        self.dataset = self.dataset.skip(num_steps * self.num_processes)
        self.dataset_iterator = iter(self.dataset)
