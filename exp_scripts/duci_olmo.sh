#!/bin/bash

CUDA_VISIBLE_DEVICES=6 python baseline_method/src/labelshift/run_duci_categories.py \
  --local_samples_dir data_samples \
  --target_model allenai/OLMo-1B \
  --merge_web \
  --mia_method loss \
  --max_per_class 5000 \
  --calibration_split 0.3 \
  --threshold_method optimal \
  --half \
  --trust_remote_code \
  --output_dir out \
  --run_name duci_olmo1b_merge_web

