import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

from itertools import groupby


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--results', help='Path to throughput.txt file (list of json).')

    args = parser.parse_args()


    with open(args.results, 'r') as f:
        rows = []
        for line in f:
            row = json.loads(line)
            rows.append(row)

    for model, stats in groupby(rows, lambda x: x['model']):
        stats = tuple(stats)
        n_tokens = np.array(tuple(s['n_token'] for s in stats))
        tps = np.array(tuple(s['tokens_per_second'] for s in stats))

        print(n_tokens)
        print(tps)

        plt.plot(n_tokens, tps, label=model)
    plt.show()
