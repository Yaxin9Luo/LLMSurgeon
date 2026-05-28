CUDA_VISIBLE_DEVICES=3 python baseline_method/src/labelshift/run_minkpp_threshold_pythia.py \
  --local_samples_dir data_samples/pile \
  --target_model EleutherAI/gpt-neo-2.7B \
  --mink_ratio 0.2 \
  --half \
  --trust_remote_code \
  --output_dir out \
  --run_name minkpp_threshold_gptneo