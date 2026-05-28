#!/usr/bin/env bash
# ===========================================================================
#  Predict training data mixtures for closed-source LLMs (GPT & Gemini)
#
#  Prerequisites:
#    pip install openai google-genai
#    export OPENAI_API_KEY="sk-..."
#    export GEMINI_API_KEY="..."       # or GOOGLE_API_KEY
#
#  Usage:
#    bash exp_scripts/closedapi.sh
# ===========================================================================

set -euo pipefail

SCRIPT="baseline_method/src/labelshift/run_labelshift_closedapi.py"
DATA_DIR="./data_samples"
NUM_PROMPTS=300
MAX_TOKENS=512
MAX_PER_CLASS=5000
PROMPTS_STYLE="neutral"

# ----------------------------- OpenAI Models ------------------------------

echo "===== Running: GPT-4o ====="
python "$SCRIPT" \
  --generator openai \
  --target_model gpt-4o \
  --local_samples_dir "$DATA_DIR" \
  --classifier distilbert \
  --num_prompts "$NUM_PROMPTS" \
  --max_new_tokens "$MAX_TOKENS" \
  --max_per_class "$MAX_PER_CLASS" \
  --prompts_style "$PROMPTS_STYLE" \
  --bootstrap \
  --output_dir out \
  --run_name closedapi_gpt4o

echo "===== Running: GPT-4o-mini ====="
python "$SCRIPT" \
  --generator openai \
  --target_model gpt-4o-mini \
  --local_samples_dir "$DATA_DIR" \
  --classifier distilbert \
  --num_prompts "$NUM_PROMPTS" \
  --max_new_tokens "$MAX_TOKENS" \
  --max_per_class "$MAX_PER_CLASS" \
  --prompts_style "$PROMPTS_STYLE" \
  --bootstrap \
  --output_dir out \
  --run_name closedapi_gpt4o_mini

echo "===== Running: GPT-3.5-turbo ====="
python "$SCRIPT" \
  --generator openai \
  --target_model gpt-3.5-turbo \
  --local_samples_dir "$DATA_DIR" \
  --classifier distilbert \
  --num_prompts "$NUM_PROMPTS" \
  --max_new_tokens "$MAX_TOKENS" \
  --max_per_class "$MAX_PER_CLASS" \
  --prompts_style "$PROMPTS_STYLE" \
  --bootstrap \
  --output_dir out \
  --run_name closedapi_gpt35_turbo

# ----------------------------- Google Gemini Models -----------------------

echo "===== Running: Gemini 2.0 Flash ====="
python "$SCRIPT" \
  --generator google \
  --target_model gemini-2.0-flash \
  --local_samples_dir "$DATA_DIR" \
  --classifier distilbert \
  --num_prompts "$NUM_PROMPTS" \
  --max_new_tokens "$MAX_TOKENS" \
  --max_per_class "$MAX_PER_CLASS" \
  --prompts_style "$PROMPTS_STYLE" \
  --bootstrap \
  --output_dir out \
  --run_name closedapi_gemini_2_0_flash

echo "===== Running: Gemini 2.5 Flash ====="
python "$SCRIPT" \
  --generator google \
  --target_model gemini-2.5-flash-preview-04-17 \
  --local_samples_dir "$DATA_DIR" \
  --classifier distilbert \
  --num_prompts "$NUM_PROMPTS" \
  --max_new_tokens "$MAX_TOKENS" \
  --max_per_class "$MAX_PER_CLASS" \
  --prompts_style "$PROMPTS_STYLE" \
  --bootstrap \
  --output_dir out \
  --run_name closedapi_gemini_2_5_flash

echo "===== Running: Gemini 2.5 Pro ====="
python "$SCRIPT" \
  --generator google \
  --target_model gemini-2.5-pro-preview-05-06 \
  --local_samples_dir "$DATA_DIR" \
  --classifier distilbert \
  --num_prompts "$NUM_PROMPTS" \
  --max_new_tokens "$MAX_TOKENS" \
  --max_per_class "$MAX_PER_CLASS" \
  --prompts_style "$PROMPTS_STYLE" \
  --bootstrap \
  --output_dir out \
  --run_name closedapi_gemini_2_5_pro

echo ""
echo "===== All closed-API experiments completed! ====="
echo "Check results in out/closedapi_*/"
