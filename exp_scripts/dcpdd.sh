CUDA_VISIBLE_DEVICES=2 python baseline_method/src/labelshift/run_dcpdd_threshold_pythia.py \
  --local_samples_dir data_samples/pile \
  --target_model EleutherAI/gpt-neo-2.7B \
  --a 0.01 \
  --freq_strategy global \
  --max_tokens 512 \
  --max_ref_texts 2000 \
  --half \
  --trust_remote_code \
  --output_dir out \
  --run_name dcpdd_threshold_gptneo


