#!/bin/bash
# Run Min-K%++ on StarCoder model
# This script runs the Min-K%++ threshold method on the StarCoder model
# to estimate the composition of programming languages in the pretraining data.

CUDA_VISIBLE_DEVICES=0 python baseline_method/src/labelshift/run_minkpp_threshold_starcoder.py \
  --samples_dir data_samples/starcoder \
  --target_model bigcode/starcoder \
  --mink_ratio 0.2 \
  --half \
  --trust_remote_code \
  --output_dir out \
  --run_name minkpp_threshold_starcoder

