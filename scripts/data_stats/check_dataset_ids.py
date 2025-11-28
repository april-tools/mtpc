import argparse
from datasets import load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a HuggingFace dataset and check unique IDs in validation set"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., agrv/tulu-v3-sft-llama3-packed-seq-len-8192)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="valid",
        help="Dataset split to analyze (default: valid)",
    )

    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} (split: {args.split})")
    ds = load_dataset(args.dataset, split=args.split)

    print(f"Dataset loaded with {len(ds)} rows")

    # Collect all IDs
    all_ids = set()
    for i, row in enumerate(ds, 1):
        # The "id" field contains a list of IDs
        row_ids = row["id"]
        if isinstance(row_ids, list):
            all_ids.update(row_ids)
        else:
            all_ids.add(row_ids)

        if i % 1000 == 0:
            print(f"Processed {i} rows, found {len(all_ids)} unique IDs so far...")

    print(f"\nTotal unique IDs found: {len(all_ids)}")

    # Sort lexicographically
    sorted_ids = sorted(all_ids)

    # Print first 50
    print("\nFirst 50 IDs (lexicographically sorted):")
    for i, id_val in enumerate(sorted_ids[:50], 1):
        print(f"{i:2d}. {id_val}")
