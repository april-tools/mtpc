import os
import pickle
import argparse
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Converts Karpathy format to our dataloader format.')
    parser.add_argument('--input', type=str, required=True,
                        help='The .bin file to convert to our dataloader format.')
    parser.add_argument('--output', type=str, required=True,
                        help='The .bin file to write the converted file to.')

    args = parser.parse_args()
    
    filename = args.input
    with open(filename, 'rb') as f:
        tokens = np.frombuffer(f.read(), dtype=np.uint16)

    # with open('data/shakespeare_char/meta.pkl', 'rb') as f:
    #    meta = pickle.load(f)
    # print(meta)
    # print(''.join(meta['itos'][t] for t in tokens[:20]))

    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520
    header[1] = 1
    header[2] = len(tokens)

    # Concatenate
    output = header.tobytes() + tokens.tobytes()

    print('Writing header and %d tokens to %s...' % (len(tokens), args.output))

    with open(args.output, 'wb') as f:
        f.write(output)
