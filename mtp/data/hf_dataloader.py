from transformers import AutoTokenizer
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node


class HFDistributedDataLoader(object):
    """A generic Huggingfaced Dataloader that is compatible with DDP.

    To override this class, see other examples in the folder.
    """

    def __init__(
        self,
        hf_dataset: str,
        hf_model: str,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
        device: str = "cuda",
        split="train",
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_model, model_max_length=self.T, padding_side="right", use_fast=True
        )

        self.reset()

    def reset(self):
        self.dataset = self.load_dataset()
        self.dataset = split_dataset_by_node(
            self.dataset, rank=self.process_rank, world_size=self.num_processes
        )
        self.dataset = self.dataset.batch(self.B)
        self.dataset = self.dataset.map(lambda x: self.process(x))

    def load_dataset(self):
        return load_dataset(self.hf_dataset, split=self.split, streaming=True)

    def process(self, x):
        raise NotImplementedError()

    def next_batch(self):
        fields = next(iter(self.dataset.take(1)))
        batch = dict()
        if self.device == "cuda":
            for k in fields:
                # Filter outputs to only include needed
                if k in ("input_ids", "labels", "attention_mask"):
                    batch[k] = fields[k].cuda()
        return batch

    def seek(self, num_steps):
        self.dataset = self.dataset.skip(num_steps)
