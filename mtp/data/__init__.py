import re

from mtp.data.local_dataloader import LocalDistributedDataLoader
from mtp.data.sharegpt import ShareGPTDataLoader
from mtp.data.tuluv3 import TuluDataLoader
from mtp.data.tuluv3_evabyte import EvaByteTuluDataLoader


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
        split: str = "train",
        as_iterable: bool = True,
        shuffle: bool = False,
    ):
        obj = None
        for pattern, constructor in cl._resolvers.items():
            if re.match(pattern, dataset, re.DOTALL):
                if constructor is LocalDistributedDataLoader:
                    assert (
                        as_iterable is True
                    ), "Only iterable supported for LocalDataLoader"
                    assert (
                        shuffle is False
                    ), "Shuffling not supported for LocalDataLoader"
                    obj = constructor(
                        dataset,
                        hf_model,
                        B,
                        T,
                        process_rank,
                        num_processes,
                        device,
                        split,
                    )
                else:
                    # iterable is only supported for HF class
                    obj = constructor(
                        dataset,
                        hf_model,
                        B,
                        T,
                        process_rank,
                        num_processes,
                        device,
                        split,
                        as_iterable,
                        shuffle,
                    )
                obj = obj.reset()
        if obj is None:
            raise ValueError("Could not resolve: %s" % dataset)
        else:
            return obj

    @classmethod
    def register(cl, key, constructor):
        # Register resolvers based on pattern matching
        cl._resolvers[key] = constructor


DistributedDataLoader.register(".+\.bin", LocalDistributedDataLoader)
DistributedDataLoader.register("Aeala/ShareGPT_Vicuna_unfiltered", ShareGPTDataLoader)
DistributedDataLoader.register("allenai/tulu-3-sft-mixture", TuluDataLoader)
DistributedDataLoader.register("agrv/tulu-v3-sft-evabyte-seq-len-8196", EvaByteTuluDataLoader)
