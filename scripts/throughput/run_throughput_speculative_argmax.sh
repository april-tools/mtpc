#!/bin/bash

#                                       model folder        name   part     step
$MTP_ROOT/bin/compute_throughput_speculative_argmax outputs/models/evabyte/no-lora evabyte-no-lora 0 900
$MTP_ROOT/bin/compute_throughput_speculative_argmax outputs/models/evabyte/no-lora evabyte-no-lora 1 900
$MTP_ROOT/bin/compute_throughput_speculative_argmax outputs/models/evabyte/no-lora evabyte-no-lora 2 900

$MTP_ROOT/bin/compute_throughput_speculative_argmax outputs/models/evabyte/lora-continued evabyte-lora-continued 0 900
$MTP_ROOT/bin/compute_throughput_speculative_argmax outputs/models/evabyte/lora-continued evabyte-lora-continued 1 900
$MTP_ROOT/bin/compute_throughput_speculative_argmax outputs/models/evabyte/lora-continued evabyte-lora-continued 2 900

$MTP_ROOT/bin/compute_throughput_speculative_argmax outputs/models/llama/no-lora llama-no-lora 0 900
$MTP_ROOT/bin/compute_throughput_speculative_argmax outputs/models/llama/no-lora llama-no-lora 1 900
$MTP_ROOT/bin/compute_throughput_speculative_argmax outputs/models/llama/no-lora llama-no-lora 2 900

$MTP_ROOT/bin/compute_throughput_speculative_argmax outputs/models/llama/lora-continued llama-lora-continued 0 900
$MTP_ROOT/bin/compute_throughput_speculative_argmax outputs/models/llama/lora-continued llama-lora-continued 1 900
$MTP_ROOT/bin/compute_throughput_speculative_argmax outputs/models/llama/lora-continued llama-lora-continued 2 900
