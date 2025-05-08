import warnings

from datasets import Value, Sequence, Features

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
        split: str = "train",
        as_iterable: bool = True,
        shuffle: bool = True,
    ):
        assert shuffle is True, 'You probably want to shuffle this dataset'
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
        self.features = Features({
            "id": Value("string"),
            "source": Value("string"),
            "messages": [{"role": Value("string"), "content": Value("string")}],
        })

    def filter(self, example):
        n = len(example["labels"])
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
        # return_tensors='np' is needed as the non-iterable hf datasets use
        # rely on PyArrow, which is not compatible with PT tensors.
        # By default the datasets.map() will silently cast to lists -_-
        out = self.data_collator([tokens], return_tensors='pt')

        # NOTE: in our implementation we expect label @ i to be target for input_id @ i
        input_ids = out["input_ids"][0, :-1]
        attention_mask = out["attention_mask"][0, :-1]
        labels = out["labels"][0, 1:]

        output = dict(
            input_ids=input_ids, labels=labels, attention_mask=attention_mask, **x
        )

        return output
