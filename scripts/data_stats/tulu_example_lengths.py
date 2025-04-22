import matplotlib.pyplot as plt
import argparse

from mtp.data import DistributedDataLoader
from mtp.models.loss import IGNORE_TOKEN_ID


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-examples', type=int, default=-1)

    args = parser.parse_args()


    SEQ_LEN = 2048 * 4
    dl = DistributedDataLoader.resolve(
        "allenai/tulu-3-sft-mixture", "EvaByte/EvaByte", 1, SEQ_LEN, 0, 1, device='cpu'
    )
    seq_lengths, toks_to_predict = [], []

    i = 0
    while True:
        try:
            batch = dl.next_batch()
            seq_lengths.append(batch['attention_mask'].sum().item())
            toks_to_predict.append((batch['labels'] != IGNORE_TOKEN_ID).sum().item())
            i += 1
            if i == args.num_examples:
                raise StopIteration()
        except StopIteration:
            break

    print(f'Iterated over {i} examples')

    # Plot histogram
    plt.hist(seq_lengths, bins=20, edgecolor="black")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Sequence length of Tulu 3 examples in bytes")
    plt.tight_layout()
    plt.show()
