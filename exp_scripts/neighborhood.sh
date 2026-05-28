CUDA_VISIBLE_DEVICES=5 python baseline_method/src/labelshift/run_neighborhood_threshold_pythia.py \
  --local_samples_dir data_samples/pile \
  --target_model EleutherAI/gpt-neo-2.7B  \
  --n_perturbations 10 \
  --pct_words_masked 0.3 \
  --span_length 2 \
  --fill_strategy global \
  --max_fill_texts 5000 \
  --max_tokens 512 \
  --half \
  --trust_remote_code \
  --output_dir out \
  --run_name neighborhood_threshold_gptneo


