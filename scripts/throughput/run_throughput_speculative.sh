#!/bin/bash

#                                       model folder        name   part     step
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/no-lora no-lora 0 900
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/no-lora no-lora 1 900
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/no-lora no-lora 2 900

$MTP_ROOT/bin/compute_throughput_speculative outputs/models/no-lora no-lora 0 600
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/no-lora no-lora 1 600
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/no-lora no-lora 2 600

$MTP_ROOT/bin/compute_throughput_speculative outputs/models/no-lora no-lora 0 300
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/no-lora no-lora 1 300
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/no-lora no-lora 2 300

$MTP_ROOT/bin/compute_throughput_speculative outputs/models/no-lora no-lora 0 0
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/no-lora no-lora 1 0
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/no-lora no-lora 2 0


$MTP_ROOT/bin/compute_throughput_speculative outputs/models/lora-last-16 lora-last-16 0 900
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/lora-last-16 lora-last-16 1 900
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/lora-last-16 lora-last-16 2 900

$MTP_ROOT/bin/compute_throughput_speculative outputs/models/lora-last-16 lora-last-16 0 600
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/lora-last-16 lora-last-16 1 600
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/lora-last-16 lora-last-16 2 600

$MTP_ROOT/bin/compute_throughput_speculative outputs/models/lora-last-16 lora-last-16 0 300
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/lora-last-16 lora-last-16 1 300
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/lora-last-16 lora-last-16 2 300

$MTP_ROOT/bin/compute_throughput_speculative outputs/models/lora-last-16 lora-last-16 0 0
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/lora-last-16 lora-last-16 1 0
$MTP_ROOT/bin/compute_throughput_speculative outputs/models/lora-last-16 lora-last-16 2 0
