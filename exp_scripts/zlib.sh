CUDA_VISIBLE_DEVICES=0 python baseline_method/src/labelshift/run_zlib_threshold_pythia.py \
  --local_samples_dir data_samples/pile \
  --mode ll_over_zlib \
  --target_model EleutherAI/gpt-neo-2.7B \
  --half \
  --trust_remote_code \
  --output_dir out \
  --run_name zlib_gptneo