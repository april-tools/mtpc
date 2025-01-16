#!/bin/bash

# In this script we evaluate generation throughput
# I.e. how many tokens we can generate per second (with a batch size of one)

for device in cuda cpu;
do
    torchrun -m nanogpt.generate device=$device model=default >> $MTP_ROOT/results/throughput.jsonl
    for n_token in 2 4 6 8 10;
	do
		for n_component in 1 3 5;
		do
			torchrun -m nanogpt.generate device=$device model=mtp model.n_component=$n_component model.n_token=$n_token >> $MTP_ROOT/results/throughput.jsonl
		done
    done
done
