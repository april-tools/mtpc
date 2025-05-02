import warnings

from datasets import load_dataset

from trl import DataCollatorForCompletionOnlyLM

from mtp.models.loss import IGNORE_TOKEN_ID
from mtp.data.hf_dataloader import HFDistributedDataLoader


# We use below to silence warnings from the DataCollator that is
# not finding the assistant label due to truncated lengths
# we filter these examples out later anyway
warnings.filterwarnings("ignore", category=UserWarning, module="trl")


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
        if hf_model in ["EvaByte/EvaByte", "EvaByte/EvaByte-SFT"]:
            self.data_collator = DataCollatorForCompletionOnlyLM(
                tokenizer=self.tokenizer,
                response_template="<|start_header_id|>assistant<|end_header_id|>\n\n",
                ignore_index=IGNORE_TOKEN_ID,
                mlm=False,
            )
        else:
            raise NotImplementedError(
                "Cannot yet handle response_template for %s" % hf_model
            )

    def filter(self, x):
        return x

    def process(self, x):
        return x

    def load_dataset(self):
        if self.split == "train":
            return load_dataset("agrv/tulu-v3-sft-evabyte-seq-len-8196", split=self.split)
        elif self.split == "valid":
            return load_dataset("agrv/tulu-v3-sft-evabyte-seq-len-8196", split='validation')
        else:
            raise ValueError("Tulu v3 dataset has no %s split" % self.split)
