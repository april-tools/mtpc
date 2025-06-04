import torch
import numpy as np
import torch.nn.functional as F

from transformers import DataCollatorWithFlattening
from itertools import chain
from more_itertools import peekable


def pad_right(ss, l, fill_token):
    return F.pad(ss, (0, l), mode='constant', value=fill_token)


def split_idxs_to_sizes(split_idxs):
    # Convert from position to split to sizes of windows
    # i.e. for a seq of len 20  [2, 5, 9] ->  [2, 3, 4, 11]
    split_sizes = torch.zeros(len(split_idxs) + 1, dtype=torch.int)
    split_sizes[1:] = split_idxs
    # take pairwise diff
    split_sizes = (split_sizes - split_sizes.roll(1))[1:]
    return split_sizes


# Create target windows from packed targets
# We cannot just unfold the whole sequence if it is packed, because
# we would get invalid windows when scanning between examples
# We therefore need to break into pieces, pad, unfold, and then concat
def packed_targets_to_target_windows(yy, n, EOS_ID, IGNORE_TOKEN_ID):
    # yy is B, S
    B, S = yy.shape
    # yy is B x S
    yy = yy.ravel()
    # Split into parts when we find EOS or the end of the sequence
    split_idxs = torch.nonzero((yy == EOS_ID) | ((torch.arange(B*S) % S) == (S-1))).ravel()
    # Include the EOS in the sequence it ends
    split_idxs += 1
    split_sizes = tuple(split_idxs_to_sizes(split_idxs).tolist())
    parts = [pad_right(each, n-1, IGNORE_TOKEN_ID).unfold(0, n, 1) for each in yy.split(split_sizes)]
    return torch.concat(parts, dim=0).reshape(B, S, n)


def pack_by_length(ds, max_len=8192, num_bins=5, num_proc=1, pad_id=0, ignore_token_id=-100, get_length=lambda x: len(x['input_ids'])):
    # Go through bins in decreasing sequenced length
    assert np.log2(max_len) % 1 == 0
    assert num_bins > 0
    assert len(ds[0]['input_ids'].shape) == 1, 'Expected input_ids to be 1d tensor'
    bins =  [0] + [int(max_len ** 1/(2**i)) for i in range(num_bins - 1, -1, -1)]
    binned_subsets = []

    pad_values = {'input_ids': pad_id, 'labels': ignore_token_id, 'attention_mask': 0}
    # Split dataset into subsets depending on length
    for i in range(len(bins) - 1):
        subset = ds.filter(lambda x: bins[i] < get_length(x) <= bins[i+1], num_proc=num_proc)
        subset = subset.to_iterable_dataset() if len(subset) > 0 else []
        subset = peekable(iter(subset))
        binned_subsets.append(subset)

    packed_dataset = []
    # While there are still examples
    while (any(binned_subsets)):
        cur_len = 0
        examples_to_pack = []
        while True:
            found = False
            # Pick largest element (iter from right)
            for stream in binned_subsets[::-1]:
                if stream:
                    peeked_example = stream.peek()
                    example_length = get_length(peeked_example)
                    if cur_len + example_length <= max_len:
                        example = next(stream)
                        examples_to_pack.append(example)
                        cur_len += example_length
                        found = True
                        break
            if not found:
                # apply data collator with flattening
                # pad on the right
                packed = dict()
                for k in examples_to_pack[0].keys():
                    if k in ("input_ids", "labels", "attention_mask"):
                        packed[k] = torch.concat([example[k] for example in examples_to_pack], dim=0)
                        # Pad to max_len on the right
                        packed[k] = F.pad(packed[k], (0, max_len - len(packed[k])), mode='constant', value=pad_values[k])
                    else:
                        packed[k] = [example[k] for example in examples_to_pack]
                packed_dataset.append(packed)
                break
    return packed_dataset
