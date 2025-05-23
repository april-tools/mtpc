#!/bin/bash

# Ran on full-tulu-evabyte-lora-last-8-cp-n-8-r-32
# Ran on full-tulu-evabyte-lora-last-8-cp-n-8-r-8
# Ran on full-tulu-evabyte-lora-last-8-ff-n-8-r-1

./bin/compute_throughput_speculative outputs/models/tulu3-evabyte/  tulu-valid  250
./bin/compute_throughput_speculative outputs/models/tulu3-evabyte/  tulu-train  250
./bin/compute_throughput_speculative_argmax outputs/models/tulu3-evabyte/  tulu-train  250
./bin/compute_throughput_speculative_argmax outputs/models/tulu3-evabyte/  tulu-valid  250


# Runs only on the 2k step of
# Ran on full-tulu-evabyte-lora-last-8-cp-n-8-r-32
# Ran on full-tulu-evabyte-lora-last-8-cp-n-8-r-8
# Ran on full-tulu-evabyte-lora-last-8-ff-n-8-r-1

./bin/compute_throughput_speculative_at_2k outputs/models/tulu3-evabyte/ tulu-valid 250
./bin/compute_throughput_speculative_argmax_at_2k outputs/models/tulu3-evabyte/ tulu-valid 250
