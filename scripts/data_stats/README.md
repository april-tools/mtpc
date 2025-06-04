# Generate the Tulu dataset

## Create the packed Tulu split
```bash
HF_DATASETS_NUM_PROC=50 HF_CACHE_ACTIVE=1 python scripts/data_stats/tulu_packed_split.py --seq-length 8192
```


# LEGACY Code

## Create the unpacked Tulu split
```bash
HF_DATASETS_NUM_PROC=50 HF_CACHE_ACTIVE=1 python scripts/data_stats/tulu_split.py
```

## Plot histogram of num_tokens for all the dataset

### Distribution over lengths we get by removing any example longer than 8196
> python scripts/data_stats/tulu_example_lengths.py --seq-length 8196

### Distribution over lengths for whole dataset
> python scripts/data_stats/tulu_example_lengths.py --seq-length 2000000
