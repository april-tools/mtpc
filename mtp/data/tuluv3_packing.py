import warnings

from datasets import Value, Sequence, Features

from trl import DataCollatorForCompletionOnlyLM

from mtp.models.loss import IGNORE_TOKEN_ID
from mtp.data.hf_dataloader import HFDistributedDataLoader


# We use below to silence warnings from the DataCollator that is
# not finding the assistant label due to truncated lengths
# we filter these examples out later anyway
warnings.filterwarnings("ignore", category=UserWarning, module="trl")


class TuluPackedDataLoader(HFDistributedDataLoader):
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

    @property
    def model_max_length(self):
        # use max_length=T because we use full input_ids and labels
        return self.T

    def filter(self, example):
        n = len(example["labels"])
        inactive = (example["labels"] == IGNORE_TOKEN_ID).sum().item()
        return (n - inactive > 0) and (n <= self.model_max_length)

    def process(self, x):
        tokens = self.tokenizer.apply_chat_template(
            x["messages"],
            tokenize=True,
            return_dict=True,
            padding='do_not_pad',
            truncation=False,
        )
        # return_tensors='np' is needed as the non-iterable hf datasets use
        # rely on PyArrow, which is not compatible with PT tensors.
        # By default the datasets.map() will silently cast to lists -_-
        out = self.data_collator([tokens], return_tensors='pt')

        # NOTE: in our implementation we expect label @ i to be target for input_id @ i
        # input_ids = out["input_ids"][0, :-1]
        # labels = out["labels"][0, 1:]

        input_ids = out["input_ids"]
        # We wouldn't condition on <eot_id> anyway, so we can drop it
        input_ids[0, -1] = self.tokenizer.added_tokens_encoder['<file_sep>']

        labels = out["labels"].clone()
        labels[0, :-1] = out["labels"][0, 1:]
        labels[0, -1] = IGNORE_TOKEN_ID

        output = dict(
            input_ids=input_ids, labels=labels, **x
        )

        return output
