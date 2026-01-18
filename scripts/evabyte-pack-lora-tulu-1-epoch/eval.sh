#!/bin/bash

##############################################
####### Speculative Sampling #################
##############################################

modelpath=outputs/models/tulu-pack-1-epoch-no-quant/1epoch-pack-tulu-evabyte-no-quant-lr-3e-4-lora-last-16-cp-n-8-r-32
./bin/compute_throughput_speculative $modelpath/model@900.pt   new-valid  1
./bin/compute_throughput_speculative $modelpath/model@600.pt   new-valid  1
./bin/compute_throughput_speculative $modelpath/model@300.pt   new-valid  1
./bin/compute_throughput_speculative $modelpath/model@900.pt   new-valid  2
./bin/compute_throughput_speculative $modelpath/model@600.pt   new-valid  2
./bin/compute_throughput_speculative $modelpath/model@300.pt   new-valid  2
./bin/compute_throughput_speculative $modelpath/model@900.pt   new-valid  3
./bin/compute_throughput_speculative $modelpath/model@600.pt   new-valid  3
./bin/compute_throughput_speculative $modelpath/model@300.pt   new-valid  3

modelpath=outputs/models/tulu-pack-1-epoch-no-quant/1epoch-pack-tulu-evabyte-no-quant-lr-3e-4-lora-last-16-cp-n-8-r-64
./bin/compute_throughput_speculative $modelpath/model@900.pt   new-valid  1
./bin/compute_throughput_speculative $modelpath/model@600.pt   new-valid  1
./bin/compute_throughput_speculative $modelpath/model@300.pt   new-valid  1
./bin/compute_throughput_speculative $modelpath/model@900.pt   new-valid  2
./bin/compute_throughput_speculative $modelpath/model@600.pt   new-valid  2
./bin/compute_throughput_speculative $modelpath/model@300.pt   new-valid  2
./bin/compute_throughput_speculative $modelpath/model@900.pt   new-valid  3
./bin/compute_throughput_speculative $modelpath/model@600.pt   new-valid  3
./bin/compute_throughput_speculative $modelpath/model@300.pt   new-valid  3

modelpath=outputs/models/tulu-pack-1-epoch-no-quant/tulu-evabyte-lr-3e-4-lora-last-16-hmm-n-8-r-32-eye
./bin/compute_throughput_speculative $modelpath/model@900.pt   new-valid  1
./bin/compute_throughput_speculative $modelpath/model@600.pt   new-valid  1
./bin/compute_throughput_speculative $modelpath/model@300.pt   new-valid  1
./bin/compute_throughput_speculative $modelpath/model@900.pt   new-valid  2
./bin/compute_throughput_speculative $modelpath/model@600.pt   new-valid  2
./bin/compute_throughput_speculative $modelpath/model@300.pt   new-valid  2
./bin/compute_throughput_speculative $modelpath/model@900.pt   new-valid  3
./bin/compute_throughput_speculative $modelpath/model@600.pt   new-valid  3
./bin/compute_throughput_speculative $modelpath/model@300.pt   new-valid  3

modelpath=outputs/models/tulu-pack-1-epoch-no-quant/tulu-evabyte-lr-3e-4-lora-last-16-hmm-n-16-r-32-eye
./bin/compute_throughput_speculative $modelpath/model@900.pt   new-valid  1
./bin/compute_throughput_speculative $modelpath/model@600.pt   new-valid  1
./bin/compute_throughput_speculative $modelpath/model@300.pt   new-valid  1
./bin/compute_throughput_speculative $modelpath/model@900.pt   new-valid  2
./bin/compute_throughput_speculative $modelpath/model@600.pt   new-valid  2
./bin/compute_throughput_speculative $modelpath/model@300.pt   new-valid  2
./bin/compute_throughput_speculative $modelpath/model@900.pt   new-valid  3
./bin/compute_throughput_speculative $modelpath/model@600.pt   new-valid  3
./bin/compute_throughput_speculative $modelpath/model@300.pt   new-valid  3


# modelpath=outputs/models/tulu-pack-1-epoch-no-quant/1epoch-pack-tulu-evabyte-beta-1-lr-3e-4-lora-last-16-cp-n-8-r-32
# ./bin/compute_throughput_speculative $modelpath/model@0.pt 
# ./bin/compute_throughput_speculative $modelpath/model@100.pt 
# ./bin/compute_throughput_speculative $modelpath/model@200.pt 
# ./bin/compute_throughput_speculative $modelpath/model@300.pt 
# ./bin/compute_throughput_speculative $modelpath/model@400.pt 
# ./bin/compute_throughput_speculative $modelpath/model@600.pt 
# ./bin/compute_throughput_speculative $modelpath/model@900.pt 
#
# modelpath=outputs/models/tulu-pack-1-epoch-no-quant/1epoch-pack-tulu-evabyte-beta-1-kl@1x100-lr-3e-4-lora-last-16-cp-n-8-r-32
# # ./bin/compute_throughput_speculative $modelpath/model@100.pt 
# # ./bin/compute_throughput_speculative $modelpath/model@300.pt 
# # ./bin/compute_throughput_speculative $modelpath/model@500.pt 
# ./bin/compute_throughput_speculative $modelpath/model@600.pt 
# ./bin/compute_throughput_speculative $modelpath/model@900.pt 
#
# modelpath=outputs/models/tulu-pack-1-epoch-no-quant/1epoch-pack-tulu-evabyte-beta-1-kl@1x10-lr-3e-4-lora-last-16-cp-n-8-r-32
# ./bin/compute_throughput_speculative $modelpath/model@300.pt 
# ./bin/compute_throughput_speculative $modelpath/model@600.pt 
# ./bin/compute_throughput_speculative $modelpath/model@900.pt 
#
# modelpath=outputs/models/tulu-pack-1-epoch-no-quant/1epoch-pack-tulu-evabyte-beta-1-kl@1x1-lr-3e-4-lora-last-16-cp-n-8-r-32
# ./bin/compute_throughput_speculative $modelpath/model@300.pt 
# ./bin/compute_throughput_speculative $modelpath/model@600.pt 
# ./bin/compute_throughput_speculative $modelpath/model@900.pt 

##############################################
####### Speculative Argmax ###################
##############################################

modelpath=outputs/models/tulu-pack-1-epoch-no-quant/1epoch-pack-tulu-evabyte-no-quant-lr-3e-4-lora-last-16-cp-n-8-r-32
./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt   new-valid  1
./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt   new-valid  1
./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt   new-valid  1
./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt   new-valid  2
./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt   new-valid  2
./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt   new-valid  2
./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt   new-valid  3
./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt   new-valid  3
./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt   new-valid  3

modelpath=outputs/models/tulu-pack-1-epoch-no-quant/1epoch-pack-tulu-evabyte-no-quant-lr-3e-4-lora-last-16-cp-n-8-r-64
./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt   new-valid  1
./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt   new-valid  1
./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt   new-valid  1
./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt   new-valid  2
./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt   new-valid  2
./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt   new-valid  2
./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt   new-valid  3
./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt   new-valid  3
./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt   new-valid  3

modelpath=outputs/models/tulu-pack-1-epoch-no-quant/tulu-evabyte-lr-3e-4-lora-last-16-hmm-n-8-r-32-eye
./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt   new-valid  1
./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt   new-valid  1
./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt   new-valid  1
./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt   new-valid  2
./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt   new-valid  2
./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt   new-valid  2
./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt   new-valid  3
./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt   new-valid  3
./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt   new-valid  3

modelpath=outputs/models/tulu-pack-1-epoch-no-quant/tulu-evabyte-lr-3e-4-lora-last-16-hmm-n-16-r-32-eye
./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt   new-valid  1
./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt   new-valid  1
./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt   new-valid  1
./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt   new-valid  2
./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt   new-valid  2
./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt   new-valid  2
./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt   new-valid  3
./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt   new-valid  3
./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt   new-valid  3

# modelpath=outputs/models/tulu-pack-1-epoch-no-quant/1epoch-pack-tulu-evabyte-no-quant-lr-3e-4-lora-last-16-cp-n-8-r-32
# ./bin/compute_throughput_speculative_argmax $modelpath/model@0.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt 

# modelpath=outputs/models/tulu-pack-1-epoch-no-quant/1epoch-pack-tulu-evabyte-beta-1-lr-3e-4-lora-last-16-cp-n-8-r-32
# ./bin/compute_throughput_speculative_argmax $modelpath/model@0.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@100.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@200.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@400.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt 

# modelpath=outputs/models/tulu-pack-1-epoch-no-quant/1epoch-pack-tulu-evabyte-beta-1-kl@1x100-lr-3e-4-lora-last-16-cp-n-8-r-32
# ./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt 
#
# modelpath=outputs/models/tulu-pack-1-epoch-no-quant/1epoch-pack-tulu-evabyte-beta-1-kl@1x10-lr-3e-4-lora-last-16-cp-n-8-r-32
# ./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt 
#
# modelpath=outputs/models/tulu-pack-1-epoch-no-quant/1epoch-pack-tulu-evabyte-beta-1-kl@1x1-lr-3e-4-lora-last-16-cp-n-8-r-32
# ./bin/compute_throughput_speculative_argmax $modelpath/model@300.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@600.pt 
# ./bin/compute_throughput_speculative_argmax $modelpath/model@900.pt 

##############################################
####### KL Validation Results ################
##############################################

# To generate KL losses on 512 examples from the validation set:
# Results written to validate_models.jsonl
# ./bin/validate_models $model_folder
