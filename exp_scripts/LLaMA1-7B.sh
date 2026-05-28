export CUDA_VISIBLE_DEVICES=1

python baseline_method/src/labelshift/run_labelshift.py \
  --local_samples_dir ./data_samples/ \
  --classifier distilbert \
  --merge_web \
  --target_model huggyllama/llama-7b \
  --num_prompts 300 \
  --max_new_tokens 512 \
  --max_per_class 2 \
  --output_dir out \
  --prompts_style neutral \
  --run_name visualization_llama-7b_acc_plot_sample2
