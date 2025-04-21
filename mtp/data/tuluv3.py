import torch
import warnings
import transformers

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM

from mtp.models.loss import IGNORE_TOKEN_ID
from mtp.data.hf_dataloader import HFDistributedDataLoader


# We use below to silence warnings from the DataCollator that is
# not finding the assistant label due to truncated lengths
# we filter these examples out later anyway
warnings.filterwarnings("ignore", category=UserWarning, module="trl")


class TuluDataLoader(HFDistributedDataLoader):
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
        super().__init__(
            hf_dataset, hf_model, B, T, process_rank, num_processes, device, split
        )
        if hf_model in ["EvaByte/EvaByte", "EvaByte/EvaByte-SFT"]:
            self.data_collator = DataCollatorForCompletionOnlyLM(
                tokenizer=self.tokenizer,
                response_template="<|start_header_id|>assistant<|end_header_id|>",
                ignore_index=IGNORE_TOKEN_ID,
                mlm=False,
            )
        else:
            raise NotImplementedError(
                "Cannot yet handle response_template for %s" % hf_model
            )

    def filter(self, example):
        n = example["labels"].shape[1]
        active = (example["labels"] == IGNORE_TOKEN_ID).sum().item()
        seq_len = example["attention_mask"].sum().item()
        return (n - active > 0) and (seq_len < self.model_max_length - 1)

    def process(self, x):
        tokens = self.tokenizer.apply_chat_template(
            x["messages"],
            tokenize=True,
            return_dict=True,
            padding="max_length",
            truncation=True,
        )
        out = self.data_collator([tokens])

        # NOTE: in our implementation we expect label @ i to be target for input_id @ i
        input_ids = out["input_ids"][:, :-1]
        attention_mask = out["attention_mask"][:, :-1]
        labels = out["labels"][:, 1:]

        output = dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

        return output

    def load_dataset(self):
        if self.split == "train":
            return load_dataset(
                "allenai/tulu-3-sft-mixture", split=self.split
            )
        else:
            raise ValueError("Tulu v3 dataset has no %s split" % self.split)
