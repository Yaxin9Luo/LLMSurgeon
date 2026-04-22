"""
Min-K% with Binary Thresholding for StarCoder - Multi-GPU DDP Version

This script uses PyTorch DDP (Distributed Data Parallel) to run Min-K%
scoring across multiple GPUs in parallel for faster inference.

Usage:
    torchrun --nproc_per_node=NUM_GPUS run_mink_threshold_starcoder_ddp.py [args]
    
Example:
    torchrun --nproc_per_node=4 run_mink_threshold_starcoder_ddp.py \
        --target_model bigcode/starcoder --half
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_utils_starcoder import load_starcoder_categories, detect_language_files, _read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate StarCoder category mixture via Min-K%% with DDP (multi-GPU)"
    )
    # Data
    p.add_argument(
        "--samples_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "data_samples" / "starcoder"),
        help="Directory containing <language>.jsonl files",
    )
    p.add_argument("--spec_path", type=str, default=None)
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_missing", action="store_true")

    # Target model
    p.add_argument("--target_model", type=str, required=True)
    p.add_argument("--hf_revision", type=str, default=None)
    p.add_argument("--half", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--trust_remote_code", action="store_true")

    # Min-K parameters
    p.add_argument("--mink_ratio", type=float, default=0.2)
    p.add_argument("--threshold", type=float, default=None)

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="mink_threshold_starcoder_ddp")
    
    return p.parse_args()


def setup_distributed():
    """Initialize distributed training environment."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
    else:
        local_rank = 0
        world_size = 1
        rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return local_rank, world_size, rank


def cleanup_distributed():
    """Clean up distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def load_model_for_rank(
    name: str, 
    local_rank: int,
    revision: str = None, 
    half: bool = False, 
    int8: bool = False, 
    trust_remote_code: bool = False
):
    """Load model on specific GPU for this rank."""
    int8_kwargs = {}
    half_kwargs = {}
    if int8:
        int8_kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
    elif half:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    
    revision_kwargs = {"revision": revision} if revision else {}
    
    # Load model on specific device
    device = f"cuda:{local_rank}"
    model = AutoModelForCausalLM.from_pretrained(
        name, 
        return_dict=True, 
        device_map={"": device},  # Put entire model on this GPU
        trust_remote_code=trust_remote_code,
        **int8_kwargs, 
        **half_kwargs,
        **revision_kwargs
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code, **revision_kwargs)
    return model, tok, device


@torch.no_grad()
def mink_score(text: str, model, tok, device: str, max_tokens: int, ratio: float) -> float:
    """Compute Min-K% score for a text sample."""
    ids = tok.encode(text, truncation=True, max_length=max_tokens)
    if len(ids) < 2:
        return float("nan")
    
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    out = model(input_ids, labels=input_ids)
    logits = out.logits[0, :-1]
    target_ids = input_ids[0, 1:].unsqueeze(-1)

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)
    
    k = max(1, int(token_log_probs.numel() * ratio))
    vals, _ = torch.sort(token_log_probs)
    return float(vals[:k].mean().item())


def load_language_samples(
    samples_dir: str,
    spec_path: str = None,
    max_per_class: int = None,
    seed: int = 0,
    require_all: bool = True
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Load samples from each language JSONL file."""
    import random
    rng = random.Random(seed)
    
    languages = load_starcoder_categories(spec_path)
    lang_to_file = detect_language_files(samples_dir, languages)
    
    if require_all and len(lang_to_file) != len(languages):
        missing = [c for c in languages if c not in lang_to_file]
        raise FileNotFoundError(
            f"Missing samples for {len(missing)} languages. E.g., {missing[:5]}"
        )
    
    languages = [c for c in languages if c in lang_to_file]
    if not languages:
        raise FileNotFoundError(f"No language files detected under {samples_dir}")
    
    lang_to_texts: Dict[str, List[str]] = {}
    for lang in languages:
        path = lang_to_file[lang]
        texts = _read_jsonl(path)
        rng.shuffle(texts)
        if max_per_class is not None:
            texts = texts[:max_per_class]
        lang_to_texts[lang] = texts
    
    languages = [lang for lang in languages if lang_to_texts.get(lang)]
    lang_to_texts = {lang: lang_to_texts[lang] for lang in languages}
    
    return languages, lang_to_texts


def distribute_data(
    languages: List[str], 
    lang_to_texts: Dict[str, List[str]], 
    rank: int, 
    world_size: int
) -> Tuple[List[Tuple[str, str, int]], int]:
    """
    Distribute samples across ranks.
    Returns list of (language, text, global_idx) for this rank.
    """
    # Flatten all samples with their language and global index
    all_samples = []
    for lang in languages:
        for text in lang_to_texts[lang]:
            all_samples.append((lang, text))
    
    total = len(all_samples)
    
    # Distribute samples across ranks
    samples_per_rank = (total + world_size - 1) // world_size
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, total)
    
    my_samples = [(lang, text, i) for i, (lang, text) in enumerate(all_samples[start_idx:end_idx], start=start_idx)]
    
    return my_samples, total


def gather_results(
    local_results: List[Tuple[str, int, float]], 
    world_size: int
) -> Optional[List[Tuple[str, int, float]]]:
    """Gather results from all ranks to rank 0."""
    if world_size == 1:
        return local_results
    
    # Serialize local results
    local_data = [(lang, idx, score) for lang, idx, score in local_results]
    
    # Gather sizes first
    local_size = torch.tensor([len(local_data)], dtype=torch.long, device="cuda")
    all_sizes = [torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    
    # Gather all results to rank 0 using all_gather_object
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_data)
    
    if dist.get_rank() == 0:
        # Flatten gathered results
        all_results = []
        for rank_results in gathered:
            all_results.extend(rank_results)
        return all_results
    return None


def main() -> None:
    args = parse_args()
    
    # Setup distributed
    local_rank, world_size, rank = setup_distributed()
    is_main = (rank == 0)
    
    if is_main:
        print(f"Running with {world_size} GPU(s)")
    
    # Load data (all ranks need to know the full structure)
    if is_main:
        print(f"Loading StarCoder samples from: {args.samples_dir}")
    
    languages, lang_to_texts = load_language_samples(
        args.samples_dir,
        spec_path=args.spec_path,
        max_per_class=args.max_per_class,
        seed=args.seed,
        require_all=not args.allow_missing,
    )
    
    total_samples = sum(len(texts) for texts in lang_to_texts.values())
    if is_main:
        print(f"Languages: {len(languages)}, Total samples: {total_samples}")
    
    # Distribute data across ranks
    my_samples, _ = distribute_data(languages, lang_to_texts, rank, world_size)
    if is_main:
        print(f"Samples per GPU: ~{len(my_samples)}")
    
    # Load model on this GPU
    if is_main:
        print(f"\nLoading model: {args.target_model}")
    
    model, tok, device = load_model_for_rank(
        args.target_model, 
        local_rank,
        revision=args.hf_revision,
        half=args.half, 
        int8=args.int8,
        trust_remote_code=args.trust_remote_code
    )
    
    # Synchronize before scoring
    if world_size > 1:
        dist.barrier()
    
    # Score this rank's samples
    local_results: List[Tuple[str, int, float]] = []
    
    pbar = tqdm(my_samples, desc=f"GPU {rank}", disable=not is_main)
    for lang, text, global_idx in pbar:
        score = mink_score(text, model, tok, device, args.max_tokens, args.mink_ratio)
        local_results.append((lang, global_idx, score))
    
    # Synchronize before gathering
    if world_size > 1:
        dist.barrier()
    
    # Gather all results to rank 0
    all_results = gather_results(local_results, world_size)
    
    # Only rank 0 processes and saves results
    if is_main and all_results is not None:
        # Reconstruct per-language scores
        lang_scores: Dict[str, List[float]] = {lang: [] for lang in languages}
        all_scores: List[float] = []
        
        # Sort by global index to maintain order
        all_results.sort(key=lambda x: x[1])
        
        for lang, _, score in all_results:
            lang_scores[lang].append(score)
            if np.isfinite(score):
                all_scores.append(score)
        
        # Determine threshold
        if args.threshold is not None:
            threshold = args.threshold
            print(f"\nUsing specified threshold: {threshold}")
        else:
            threshold = float(np.median(all_scores))
            print(f"\nUsing adaptive threshold (median): {threshold:.6f}")
        
        # Apply threshold
        lang_members: Dict[str, int] = {lang: 0 for lang in languages}
        for lang in languages:
            for score in lang_scores[lang]:
                if np.isfinite(score) and score > threshold:
                    lang_members[lang] += 1
        
        # Compute statistics
        per_lang_results = []
        for lang in languages:
            scores = np.array(lang_scores[lang])
            n_total = len(scores)
            n_valid = int(np.isfinite(scores).sum())
            n_members = lang_members[lang]
            proportion = n_members / n_valid if n_valid > 0 else 0.0
            
            per_lang_results.append({
                "language": lang,
                "n_total": n_total,
                "n_valid": n_valid,
                "n_members": n_members,
                "member_proportion": proportion,
                "score_mean": float(np.nanmean(scores)) if n_valid > 0 else None,
                "score_std": float(np.nanstd(scores)) if n_valid > 1 else None,
                "score_median": float(np.nanmedian(scores)) if n_valid > 0 else None,
            })
        
        # Global mixture
        total_members = sum(lang_members.values())
        if total_members > 0:
            global_mixture = {lang: lang_members[lang] / total_members for lang in languages}
        else:
            global_mixture = {lang: 0.0 for lang in languages}
        
        # Print top results
        print("\n" + "=" * 70)
        print("Top 15 Languages by Predicted Proportion:")
        print("=" * 70)
        sorted_langs = sorted(languages, key=lambda l: global_mixture[l], reverse=True)
        for lang in sorted_langs[:15]:
            r = next(x for x in per_lang_results if x["language"] == lang)
            print(f"  {lang:20s}: {r['n_members']:5d}/{r['n_valid']:5d} → {global_mixture[lang]:.4f}")
        
        # Save outputs
        out_dir = Path(args.output_dir) / args.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        payload = {
            "config": {
                "samples_dir": args.samples_dir,
                "target_model": args.target_model,
                "mink_ratio": args.mink_ratio,
                "threshold": threshold,
                "world_size": world_size,
            },
            "languages": languages,
            "per_language": per_lang_results,
            "global_mixture": global_mixture,
            "summary_stats": {
                "total_languages": len(languages),
                "total_samples": total_samples,
                "total_members": total_members,
                "threshold_used": threshold,
            }
        }
        
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        with open(out_dir / "per_language.csv", "w", encoding="utf-8") as f:
            f.write("language,n_total,n_valid,n_members,member_proportion,score_mean\n")
            for r in per_lang_results:
                f.write(f"{r['language']},{r['n_total']},{r['n_valid']},{r['n_members']},"
                        f"{r['member_proportion']:.6f},{r['score_mean'] or ''}\n")
        
        with open(out_dir / "global_mixture.csv", "w", encoding="utf-8") as f:
            f.write("language,predicted_proportion\n")
            for lang in languages:
                f.write(f"{lang},{global_mixture[lang]:.6f}\n")
        
        print(f"\nWrote outputs to {out_dir}")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()

