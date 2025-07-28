import os

from datasets import DatasetDict, Dataset

from mtp.data import DistributedDataLoader


if __name__ == "__main__":

    # This script creates a padded dataset which has the same examples
    # and split as the padded dataset.
    # Why? Because EvaByte is hard to use in validation mode with a packed dataset
    # We therefore take the packed validation set and create
    # an equivalent one but with padding.
    # Keep track of which rows are in which packed split
    train_ids, valid_ids = [], []

    S = 8192
    PRINT_EVERY = 2000

    ds = DistributedDataLoader.resolve(f'agrv/tulu-v3-sft-evabyte-packed-seq-len-{S}',
                                       'EvaByte/EvaByte-SFT', 1, S, 0, 1, split='train',
                                       as_iterable=False)
    for i, row in enumerate(iter(ds.dataset), 1):
        train_ids.extend(row['id'][0])
        if i % (PRINT_EVERY * 10) == 0:
            print(f'Processed {i} packed training rows..')

    ds = DistributedDataLoader.resolve(f'agrv/tulu-v3-sft-evabyte-packed-seq-len-{S}',
                                       'EvaByte/EvaByte-SFT', 1, S, 0, 1, split='valid',
                                       as_iterable=False)
    for i, row in enumerate(iter(ds.dataset), 1):
        valid_ids.extend(row['id'][0])
        if i % PRINT_EVERY == 0:
            print(f'Processed {i} packed validation rows..')

    print(f'Training examples: {len(train_ids)} of which unique {len(set(train_ids))}')
    print(f'Validation examples: {len(valid_ids)} of which unique {len(set(valid_ids))}')
    # assert len(train_ids) == len(set(train_ids))
    # assert len(valid_ids) == len(set(valid_ids))
    print(f'The packed dataset contains {len(train_ids)} train and {len(valid_ids)} validation examples')

    # Store the padded rows in a dictionary so we can easily look them up
    padded_rows = dict()
    # Deal with train
    ds = DistributedDataLoader.resolve(f'agrv/tulu-v3-sft-evabyte-seq-len-{S}',
                                       'EvaByte/EvaByte-SFT', None, S, 0, 1, split='train',
                                       as_iterable=False)
    for i, row in enumerate(iter(ds.dataset), 1):
        row_id = row['id']
        padded_rows[row_id] = row
        if i % (PRINT_EVERY * 10) == 0:
            print(f'Processed {i} padded train rows..')
    # Deal with valid
    ds = DistributedDataLoader.resolve(f'agrv/tulu-v3-sft-evabyte-seq-len-{S}',
                                       'EvaByte/EvaByte-SFT', None, S, 0, 1, split='valid',
                                       as_iterable=False)
    for i, row in enumerate(iter(ds.dataset), 1):
        row_id = row['id']
        padded_rows[row_id] = row
        if i % PRINT_EVERY == 0:
            print(f'Processed {i} padded validation rows..')

    num_errors = 0
    print('Converting packed dataset splits to padded ones')
    train_rows, valid_rows = [], []
    for row_id in train_ids:
        try:
            train_rows.append(padded_rows[row_id])
        except KeyError:
            # print(f'Could not find {row_id}')
            num_errors += 1
    for row_id in valid_ids:
        try:
            valid_rows.append(padded_rows[row_id])
        except KeyError:
            # print(f'Could not find {row_id}')
            num_errors += 1
    print(f'Converted to padding and dropped {num_errors} examples which could not be found')

    # NOTE: Below we just sample as many train rows as the validation set
    # this is because for some reason the hf library uses a huge amount of memory
    # for the operation below and it just wasn't worth spending time to debug this now.
    print(f'Creating new set with {len(valid_rows)} subsampled training examples')
    train_dataset = Dataset.from_list(train_rows[:len(valid_rows)], features=ds.features)
    print(f'Creating new validation set with {len(valid_rows)} validation examples')
    valid_dataset = Dataset.from_list(valid_rows, features=ds.features)

    # Create a DatasetDict if needed
    dataset_dict = DatasetDict({"train": train_dataset, "valid": valid_dataset})

    dataset_name = f"agrv/tulu-v3-sft-evabyte-padded-seq-len-{S}"
    print('Pushing dataset to HF: {dataset_name}')
    dataset_dict.push_to_hub(
            dataset_name,
            token=os.environ["HF_TOKEN"],
    )
    dataset_dict.cleanup_cache_files()
