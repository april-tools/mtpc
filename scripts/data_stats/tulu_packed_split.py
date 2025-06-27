import os
import argparse
import numpy as np

from datasets import DatasetDict, Dataset, Value, Sequence, Features

from mtp.data import TuluPackedDataLoader
from mtp.utils.packing import pack_by_length
from mtp.models.loss import IGNORE_TOKEN_ID


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-length", type=int, default=2048 * 4)

    args = parser.parse_args()

    assert (
        args.seq_length % 2048 == 0
    ), "Sequence length must be a multiple of 2048 for EvaByte compatibility"

    # NOTE: We drop examples that do not fit in the sequence length
    # so changing the sequence length will change the number of outputs
    # TuluPackedDataLoader does not pad the examples, so we can pack them later
    ds = TuluPackedDataLoader(
        "allenai/tulu-3-sft-mixture",
        "EvaByte/EvaByte-SFT",
        None,
        args.seq_length,
        0,
        1,
        device="cpu",
        split="train",
        as_iterable=False,
        shuffle=True,
    )
    ds = ds.reset()

    # Pack the dataset
    packed_rows = pack_by_length(
        ds.dataset,
        max_len=args.seq_length,
        num_bins=5,
        num_proc=int(os.environ.get('HF_DATASETS_NUM_PROC', 1)),
        pad_id=ds.tokenizer.pad_token_type_id,
        ignore_token_id=IGNORE_TOKEN_ID,
    )
    features = Features(
        {
            "id": [Value("string")],
            "source": [Value("string")],
            "messages": [[{"role": Value("string"), "content": Value("string")}]],
            "input_ids": Sequence(Value("int16"), length=ds.T),
            "labels": Sequence(Value("int16"), length=ds.T),
        }
    )
    pds = Dataset.from_list(packed_rows, features=features)
    # Re-shuffle after packing
    pds = pds.shuffle(42)

    split_dataset = pds.train_test_split(test_size=0.01)

    train_dataset = split_dataset["train"]
    valid_dataset = split_dataset["test"]

    # Create a DatasetDict if needed
    dataset_dict = DatasetDict({"train": train_dataset, "valid": valid_dataset})

    dataset_dict.push_to_hub(
        f"agrv/tulu-v3-sft-evabyte-packed-seq-len-{args.seq_length}",
        token=os.environ["HF_TOKEN"],
    )
    dataset_dict.cleanup_cache_files()
