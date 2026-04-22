export CUDA_VISIBLE_DEVICES=4

python baseline_method/src/labelshift/run_labelshift.py \
  --local_samples_dir ./data_samples/ \
  --classifier distilbert \
  --merge_web \
  --max_per_class 500 \
  --target_model LLM360/Amber \
  --num_prompts 300 \
  --max_new_tokens 512 \
  --simulate_label_shift \
  --acc_log_file results_plot_amber.csv \
  --output_dir out \
  --prompts_style neutral \
  --run_name visualization_amber_sample500