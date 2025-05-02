import matplotlib.pyplot as plt
import argparse

from mtp.data import DistributedDataLoader
from mtp.models.loss import IGNORE_TOKEN_ID


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-examples', type=int, default=-1)
    parser.add_argument('--seq-length', type=int, default=2048 * 4)

    args = parser.parse_args()


    # NOTE: We drop examples that do not fit in the sequence length
    # so changing the sequence length will change the number of outputs
    SEQ_LEN = args.seq_length
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
            if i % 10000 == 0:
                print(f'Processed {i} examples')
            if i == args.num_examples:
                raise StopIteration()
        except StopIteration:
            break

    print(f'Iterated over {i} examples')

    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(seq_lengths, bins=20, edgecolor="black")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Sequence length of Tulu 3 examples in bytes")
    plt.tight_layout()
    plt.show()
