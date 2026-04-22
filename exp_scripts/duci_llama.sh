#!/bin/bash
# Run DUCI for 7-category composition estimation on LLaMA-7B
#
# DUCI (Dataset Usage Cardinality Inference) estimates the proportion of each 
# category (CommonCrawl, C4, GitHub, Wikipedia, Books, Arxiv, StackExchange)
# used in training the model.

CUDA_VISIBLE_DEVICES=2 python baseline_method/src/labelshift/run_duci_categories.py \
  --local_samples_dir data_samples \
  --target_model huggyllama/llama-7b \
  --mia_method loss \
  --merge_web \
  --max_per_class 5000 \
  --calibration_split 0.3 \
  --threshold_method optimal \
  --half \
  --trust_remote_code \
  --output_dir out \
  --run_name duci_llama7b_merge_web

