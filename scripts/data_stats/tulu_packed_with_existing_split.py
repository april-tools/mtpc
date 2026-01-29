import os
import argparse

from collections import defaultdict
from datasets import DatasetDict, Dataset, Value, Sequence, Features

from mtp.data import DistributedDataLoader
from mtp.data.tuluv3_packing import TuluPackedDataLoader
from mtp.utils.packing import pack_by_length
from mtp.models.loss import IGNORE_TOKEN_ID


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-length", type=int, default=2048 * 4)
    parser.add_argument(
        "--model",
        type=str,
        choices=("EvaByte/EvaByte-SFT", "benjamin/Llama3-2-3B-IT-Byte"),
        default="EvaByte/EvaByte-SFT",
    )

    args = parser.parse_args()

    assert (
        args.seq_length % 2048 == 0
    ), "Sequence length must be a multiple of 2048 for EvaByte compatibility"

    PRINT_EVERY = 2000

    repo, model_abbrv = args.model.split("/", 1)
    model_abbrv, _ = model_abbrv.split("-", 1)
    model_abbrv = model_abbrv.lower()

    features = Features(
        {
            "id": [Value("string")],
            "source": [Value("string")],
            "messages": [[{"role": Value("string"), "content": Value("string")}]],
            "input_ids": Sequence(Value("int16"), length=args.seq_length),
            "labels": Sequence(Value("int16"), length=args.seq_length),
        }
    )

    valid_ids = set()
    ds = DistributedDataLoader.resolve(
        f"agrv/tulu-v3-sft-evabyte-packed-seq-len-{args.seq_length}",
        "EvaByte/EvaByte-SFT",
        1,
        args.seq_length,
        0,
        1,
        split="valid",
        as_iterable=False,
    )
    for i, row in enumerate(iter(ds.dataset), 1):
        valid_ids.update(row["id"][0])
        if i % (PRINT_EVERY * 10) == 0:
            print(f"Processed {i} packed validation rows..")

    # NOTE: We drop examples that do not fit in the sequence length
    # so changing the sequence length will change the number of outputs
    # TuluPackedDataLoader does not pad the examples, so we can pack them later
    ds = TuluPackedDataLoader(
        "allenai/tulu-3-sft-mixture",
        args.model,
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

    pad_id = ds.tokenizer.pad_token_type_id

    train_rows, valid_rows = [], []
    rows = defaultdict(list)
    datasets = dict()

    for i, row in enumerate(iter(ds.dataset), 1):
        row_id = row["id"]
        if row_id in valid_ids:
            rows["valid"].append(row)
        else:
            rows["train"].append(row)
        if i % (PRINT_EVERY * 10) == 0:
            print(f"Fetched {i} rows..")
            break

    for split in rows.keys():
        ds = Dataset.from_list(rows[split]).with_format("torch")

        # Pack the dataset
        packed_rows = pack_by_length(
            ds,
            max_len=args.seq_length,
            num_bins=5,
            num_proc=int(os.environ.get("HF_DATASETS_NUM_PROC", 1)),
            pad_id=pad_id,
            ignore_token_id=IGNORE_TOKEN_ID,
        )
        pds = Dataset.from_list(packed_rows, features=features)
        # Re-shuffle after packing
        pds = pds.shuffle(42)
        datasets[split] = pds

    # Create a DatasetDict if needed
    dataset_dict = DatasetDict({"train": datasets["train"], "valid": datasets["valid"]})

    dataset_dict.push_to_hub(
        f"agrv/tulu-v3-sft-{model_abbrv}-packed-seq-len-{args.seq_length}",
        token=os.environ["HF_TOKEN"],
    )
    dataset_dict.cleanup_cache_files()
