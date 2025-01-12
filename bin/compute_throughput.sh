#!/bin/bash

# In this script we evaluate generation throughput
# I.e. how many tokens we can generate per second (with a batch size of one)

for device in cpu cuda;
do
    torchrun -m nanogpt.generate device=cpu model=default >> $MTP_ROOT/results/throughput.jsonl
    # for n_token in 2 3 4 5 6;
    # do
    #    torchrun -m nanogpt.generate device=cpu model=default model.n_token=$n_token >> $MTP_ROOT/results/throughput.jsonl
    # done
done
