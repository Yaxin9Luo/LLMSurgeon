export CUDA_VISIBLE_DEVICES=2

python baseline_method/src/labelshift/run_labelshift_olmo3.py \
  --local_samples_dir /data/hulk/yaxin/data_anatomy/data_samples/olmo3 \
  --classifier distilbert \
  --target_model allenai/Olmo-3-1025-7B \
  --num_prompts 300 \
  --max_new_tokens 512 \
  --max_per_class 5000 \
  --prompts_style neutral \
  --output_dir out \
  --run_name ablation_OLMo3-7B_neutral