#!/bin/bash
# Run DUCI for 7-category composition estimation on Amber-13B
#
# DUCI (Dataset Usage Cardinality Inference) estimates the proportion of each 
# category used in training the Amber model.

CUDA_VISIBLE_DEVICES=1 python baseline_method/src/labelshift/run_duci_categories.py \
  --local_samples_dir data_samples \
  --target_model LLM360/Amber \
  --merge_web \
  --mia_method loss \
  --max_per_class 5000 \
  --calibration_split 0.3 \
  --threshold_method optimal \
  --half \
  --trust_remote_code \
  --output_dir out \
  --run_name duci_amber_merge_web

