#!/bin/bash
# Run Min-K% on StarCoder model with Multi-GPU (DDP)
# This script uses torchrun to distribute the workload across multiple GPUs.
#
# Usage:
#   bash starcoder_mink_ddp.sh [NUM_GPUS]
#   
# Examples:
#   bash starcoder_mink_ddp.sh 4      # Use 4 GPUs
#   bash starcoder_mink_ddp.sh        # Use all available GPUs

# Default to all available GPUs if not specified
NUM_GPUS=${1:-$(nvidia-smi -L | wc -l)}

echo "Running Min-K% on StarCoder with $NUM_GPUS GPUs"

cd /data/hulk/yaxin/data_anatomy

torchrun --nproc_per_node=$NUM_GPUS \
  baseline_method/src/labelshift/run_mink_threshold_starcoder_ddp.py \
  --samples_dir data_samples/starcoder \
  --target_model bigcode/starcoder \
  --mink_ratio 0.2 \
  --half \
  --trust_remote_code \
  --output_dir out \
  --run_name mink_threshold_starcoder_ddp

