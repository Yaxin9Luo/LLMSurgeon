export CUDA_VISIBLE_DEVICES=3

python baseline_method/src/labelshift/run_labelshift_pythia.py \
  --local_samples_dir ./data_samples/pile\
  --classifier distilbert \
  --naive  \
  --target_model EleutherAI/pythia-2.8b \
  --num_prompts 300 \
  --max_new_tokens 512 \
  --max_per_class 5000 \
  --prompts_style neutral \
  --output_dir out \
  --run_name ablation_gptneo_naive_baseline