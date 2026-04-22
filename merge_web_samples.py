#!/usr/bin/env python3
"""
Create a merged web samples file by combining CommonCrawl and C4.

By default, samples 2,500 examples from each of data_samples/commoncrawl.jsonl
and data_samples/c4.jsonl (random without replacement), shuffles, and writes to
data_samples/web.jsonl as JSONL with {"text": ...} lines.

Usage:
    python merge_web_samples.py --data_dir data_samples --total 5000 --seed 42

Notes:
    - If either source has fewer lines than requested per-source, it will use all
      available from that source and fill the remainder from the other, up to total.
    - Deduplicates simple exact matches after whitespace normalization.
"""

import argparse
import json
import os
import random
from typing import List


def read_jsonl_texts(path: str) -> List[str]:
    arr: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text")
            if isinstance(t, str) and t.strip():
                arr.append(t)
    return arr


def write_jsonl_texts(path: str, texts: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Merge CommonCrawl + C4 into web.jsonl")
    ap.add_argument("--data_dir", type=str, default="data_samples")
    ap.add_argument("--total", type=int, default=5000, help="Total merged samples to write")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cc_path = os.path.join(args.data_dir, "commoncrawl.jsonl")
    c4_path = os.path.join(args.data_dir, "c4.jsonl")
    out_path = os.path.join(args.data_dir, "web.jsonl")

    if not os.path.exists(cc_path) or not os.path.exists(c4_path):
        raise FileNotFoundError(f"Expected both files present: {cc_path} and {c4_path}")

    print(f"Reading CommonCrawl from {cc_path}")
    cc = read_jsonl_texts(cc_path)
    print(f"Reading C4 from {c4_path}")
    c4 = read_jsonl_texts(c4_path)

    rng = random.Random(args.seed)

    # Target per-source split (50/50)
    half = args.total // 2
    n_cc = min(half, len(cc))
    n_c4 = min(args.total - n_cc, len(c4))
    # If one source is short, try to fill from the other
    if n_cc + n_c4 < args.total:
        extra_needed = args.total - (n_cc + n_c4)
        if n_cc < len(cc):
            add = min(extra_needed, len(cc) - n_cc)
            n_cc += add
            extra_needed -= add
        if extra_needed > 0 and n_c4 < len(c4):
            add = min(extra_needed, len(c4) - n_c4)
            n_c4 += add

    rng.shuffle(cc)
    rng.shuffle(c4)
    merged = cc[:n_cc] + c4[:n_c4]
    rng.shuffle(merged)

    # Deduplicate by normalized whitespace + lowercase
    seen = set()
    unique: List[str] = []
    for t in merged:
        key = " ".join(t.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)
        if len(unique) >= args.total:
            break

    print(f"Writing {len(unique)} merged samples to {out_path}")
    write_jsonl_texts(out_path, unique)
    print("Done.")


if __name__ == "__main__":
    main()

