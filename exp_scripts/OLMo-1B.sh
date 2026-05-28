export CUDA_VISIBLE_DEVICES=2

python baseline_method/src/labelshift/run_labelshift.py \
  --local_samples_dir ./data_samples/ \
  --merge_web \
  --classifier distilbert \
  --target_model allenai/OLMo-1B \
  --num_prompts 300 \
  --max_new_tokens 512 \
  --max_per_class 5000 \
  --prompts_style neutral \
  --simulate_label_shift \
  --acc_log_file results_plot_olmo1b.csv \
  --output_dir out \
  --run_name visualization_OLMo-1B_sample5000