import torch
import torch.nn.functional as F


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
