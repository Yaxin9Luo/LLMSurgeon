CUDA_VISIBLE_DEVICES=1 python baseline_method/src/labelshift/run_recall_threshold_pythia.py \
  --local_samples_dir data_samples/pile \
  --target_model EleutherAI/gpt-neo-2.7B \
  --num_shots 12 \
  --prefix_strategy global \
  --max_tokens 512 \
  --half \
  --trust_remote_code \
  --output_dir out \
  --run_name recall_threshold_gptneo