#!/bin/bash
# Run DUCI for 7-category composition estimation on Pythia-2.8B
#
# DUCI (Dataset Usage Cardinality Inference) estimates the proportion of each 
# category used in training the model.

CUDA_VISIBLE_DEVICES=0 python baseline_method/src/labelshift/run_duci_categories_pythia.py \
  --local_samples_dir data_samples/pile \
  --target_model EleutherAI/gpt-neo-2.7B \
  --mia_method loss \
  --max_per_class 5000 \
  --calibration_split 0.3 \
  --threshold_method optimal \
  --half \
  --trust_remote_code \
  --output_dir out \
  --run_name duci_gptneo

