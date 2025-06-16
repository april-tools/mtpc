#!/bin/bash

num_examples=250

##############################################
####### Speculative Sampling #################
##############################################

# 3e-5
./bin/compute_throughput_speculative outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-5-lora-last-16-cp-n-8-r-32/model@0.pt tulu-valid $num_examples
./bin/compute_throughput_speculative outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-5-lora-last-16-cp-n-8-r-32/model@300.pt tulu-valid $num_examples
./bin/compute_throughput_speculative outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-5-lora-last-16-cp-n-8-r-32/model@600.pt tulu-valid $num_examples
./bin/compute_throughput_speculative outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-5-lora-last-16-cp-n-8-r-32/model@900.pt tulu-valid $num_examples

# 3e-4
./bin/compute_throughput_speculative outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-4-lora-last-16-cp-n-8-r-32/model@0.pt tulu-valid $num_examples
./bin/compute_throughput_speculative outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-4-lora-last-16-cp-n-8-r-32/model@300.pt tulu-valid $num_examples
./bin/compute_throughput_speculative outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-4-lora-last-16-cp-n-8-r-32/model@600.pt tulu-valid $num_examples
./bin/compute_throughput_speculative outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-4-lora-last-16-cp-n-8-r-32/model@900.pt tulu-valid $num_examples

##############################################
####### Speculative Argmax ###################
##############################################

# 3e-5
./bin/compute_throughput_speculative_argmax outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-5-lora-last-16-cp-n-8-r-32/model@0.pt tulu-valid $num_examples
./bin/compute_throughput_speculative_argmax outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-5-lora-last-16-cp-n-8-r-32/model@300.pt tulu-valid $num_examples
./bin/compute_throughput_speculative_argmax outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-5-lora-last-16-cp-n-8-r-32/model@600.pt tulu-valid $num_examples
./bin/compute_throughput_speculative_argmax outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-5-lora-last-16-cp-n-8-r-32/model@900.pt tulu-valid $num_examples

# 3e-4
./bin/compute_throughput_speculative_argmax outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-4-lora-last-16-cp-n-8-r-32/model@0.pt tulu-valid $num_examples
./bin/compute_throughput_speculative_argmax outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-4-lora-last-16-cp-n-8-r-32/model@300.pt tulu-valid $num_examples
./bin/compute_throughput_speculative_argmax outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-4-lora-last-16-cp-n-8-r-32/model@600.pt tulu-valid $num_examples
./bin/compute_throughput_speculative_argmax outputs/models/tulu3-evabyte/pack-1-epoch/1epoch-pack-tulu-evabyte-lr-3e-4-lora-last-16-cp-n-8-r-32/model@900.pt tulu-valid $num_examples


# To generate KL losses on 512 examples from the validation set:
# Ran on full-tulu-evabyte-lora-last-8-cp-n-8-r-32
# Ran on full-tulu-evabyte-lora-last-8-cp-n-8-r-8
# Ran on full-tulu-evabyte-lora-last-8-ff-n-8-r-1
# Results written to validate_models.jsonl
# ./bin/validate_models $model_folder
