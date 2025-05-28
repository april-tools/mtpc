import os
import argparse

from datasets import DatasetDict

from mtp.data import DistributedDataLoader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq-length', type=int, default=2048 * 4)

    args = parser.parse_args()

    # NOTE: We drop examples that do not fit in the sequence length
    # so changing the sequence length will change the number of outputs
    SEQ_LEN = args.seq_length
    dl = DistributedDataLoader.resolve(
        "allenai/tulu-3-sft-mixture", "EvaByte/EvaByte", None, SEQ_LEN, 0, 1, device='cpu', as_iterable=False, shuffle=True
    )

    split_dataset = dl.dataset.train_test_split(test_size=0.01)

    train_dataset = split_dataset["train"]
    valid_dataset = split_dataset["test"]

    # Create a DatasetDict if needed
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "valid": valid_dataset
    })

    dataset_dict.push_to_hub("agrv/tulu-v3-sft-evabyte-seq-len-8192", token=os.environ['HF_TOKEN'])
    dataset_dict.cleanup_cache_files()
