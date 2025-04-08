import re

from .local_dataloader import LocalDistributedDataLoader
from .hf_dataloader import HFDistributedDataLoader
from .sharegpt import ShareGPTDataLoader


class DistributedDataLoader:

    _resolvers = {}

    @classmethod
    def resolve(
        cl,
        dataset: str,
        hf_model: str,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
        device: str = "cuda",
        split="train",
    ):
        for pattern, constructor in cl._resolvers.items():
            if re.match(pattern, dataset):
                return constructor(
                    dataset, hf_model, B, T, process_rank, num_processes, device, split
                )
        raise ValueError('Could not resolve: %s' % dataset)

    @classmethod
    def register(cl, key, constructor):
        # Register resolvers based on pattern matching
        cl._resolvers[key] = constructor


DistributedDataLoader.register('.+\.bin', LocalDistributedDataLoader)
DistributedDataLoader.register('Aeala/ShareGPT_Vicuna_unfiltered', ShareGPTDataLoader)
