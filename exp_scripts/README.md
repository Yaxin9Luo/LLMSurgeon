# Reproduction Scripts

Shell wrappers that reproduce the LLMScan results reported in the paper. All
scripts are launched from the **repo root**, e.g.:

```bash
bash exp_scripts/OLMo-1B.sh
```

Each script sets `CUDA_VISIBLE_DEVICES` — edit for your hardware. A few of the
multi-GPU DDP scripts contain an absolute `cd` path from our cluster; replace
it with your own checkout path before running.

Expect generation + scoring to run from minutes (1–7B, coarse) to several
hours (65B or StarCoder fine-grained). See the paper appendix for exact
runtime numbers.

## LLMSurgeon (main method — Table 2)

| Script | Target model | Paper table / figure |
| --- | --- | --- |
| `OLMo-1B.sh` | `allenai/OLMo-1B` | Table 2 (coarse) |
| `LLaMA1-7B.sh` | `huggyllama/llama-7b` | Table 2 (coarse) |
| `Amber-13B.sh` | `LLM360/Amber` | Table 2 (coarse) |
| `pythia.sh` | `EleutherAI/pythia-2.8b` / `gpt-neo-2.7B` | Table 2 (mid) |
| `olmo3.sh` | `allenai/Olmo-3-1025-7B` | Held-out generalization table |
| `starcoder.sh` | `bigcode/starcoder` | Table 2 (fine) |
| `closedapi.sh` | OpenAI + Gemini APIs | Closed-source models appendix |

## Baselines (Table 2 competing methods)

All baselines operate on Pythia / GPT-Neo unless otherwise noted.

| Script | Baseline | Paper table |
| --- | --- | --- |
| `mink.sh` | Min-K% | Table 2 |
| `minkpp.sh` | Min-K%++ | Table 2 |
| `zlib.sh` | zlib log-prob ratio | Table 2 |
| `recall.sh` | ReCaLL | Table 2 |
| `neighborhood.sh` | Neighborhood attack | Table 2 |
| `dcpdd.sh` | DC-PDD | Table 2 |
| `duci_olmo.sh` | DUCI on OLMo-1B | Table 2 |
| `duci_llama.sh` | DUCI on LLaMA-7B | Table 2 |
| `duci_amber.sh` | DUCI on Amber-13B | Table 2 |
| `duci_pythia.sh` | DUCI on Pythia / GPT-Neo | Table 2 |
| `duci_starcoder.sh` | DUCI on StarCoder | Table 2 (fine) |
| `starcoder_mink.sh` / `starcoder_mink_ddp.sh` | Min-K% on StarCoder (single / multi-GPU) | Table 2 (fine) |
| `starcoder_minkpp.sh` / `starcoder_minkpp_ddp.sh` | Min-K%++ on StarCoder (single / multi-GPU) | Table 2 (fine) |

## Evaluation

`evaluation.sh` is a minimal example of scoring a single `out/<run_name>/`
directory against a YAML ground-truth spec under `bench/specs/`. See the
top-level README for the `benchmark_evaluation.py` CLI.
