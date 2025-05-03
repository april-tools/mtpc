import warnings

from datasets import Value, Sequence, Features

from trl import DataCollatorForCompletionOnlyLM

from mtp.models.loss import IGNORE_TOKEN_ID
from mtp.data.hf_dataloader import HFDistributedDataLoader


class EvaByteTuluDataLoader(HFDistributedDataLoader):
    def __init__(
        self,
        hf_dataset: str,
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
        assert shuffle is False, 'We already shuffled this dataset before splitting'
        assert T == 4 * 2048, 'This dataset is pre-tokenised to seqlen 8192'
        assert hf_model in ["EvaByte/EvaByte", "EvaByte/EvaByte-SFT"], 'Dataset is pre-tokenised for EvaByte'

        super().__init__(
            hf_dataset,
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

        self.features = Features({
            "id": Value("string"),
            "source": Value("string"),
            "messages": [{"role": Value("string"), "content": Value("string")}],
            "input_ids": Sequence(Value("int16"), length=self.T),
            "labels": Sequence(Value("int16"), length=self.T),
            "attention_mask": Sequence(Value("bool"), length=self.T),
        })
