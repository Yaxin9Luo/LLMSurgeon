#!/bin/bash
# Run DUCI for 86-language composition estimation on StarCoder
#
# DUCI (Dataset Usage Cardinality Inference) estimates the proportion of each 
# programming language used in training StarCoder.

CUDA_VISIBLE_DEVICES=0,1 python baseline_method/src/labelshift/run_duci_categories_starcoder.py \
  --local_samples_dir data_samples/starcoder \
  --target_model bigcode/starcoder \
  --mia_method loss \
  --max_per_class 200 \
  --calibration_split 0.3 \
  --threshold_method optimal \
  --half \
  --trust_remote_code \
  --allow_missing \
  --output_dir out \
  --run_name duci_starcoder

