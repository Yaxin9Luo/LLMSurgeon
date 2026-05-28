#!/usr/bin/env python3
"""
Fetch StarCoder language samples into local JSONL files.

This script mirrors `fetch_category_samples.py` but targets the 86 StarCoder
languages defined in `bench/specs/starcoder.yaml`. It expects a local root
directory containing per-language data (e.g., a mirror of The Stack/StarCoder
sources) and will efficiently sample texts and write them to
`data_samples/starcoder/<language>.jsonl` as {"text": ...} lines.

Usage examples:
  python fetch_starcoder_samples.py \
      --stack_root /path/to/the-stack-root \
      --n_per_language 5000 \
      --out_dir data_samples/starcoder

Notes:
  - The script searches for per-language files under
    `<stack_root>/<language>/**/*.{jsonl.zst,jsonl.gz,jsonl,parquet}`.
  - For .jsonl.zst, it uses fast streaming via the `zstd` CLI (and optional
    `zstandard` Python binding to detect support).
  - If multiple files are present, it samples across them.
  - Deduplicates within the fetched batch using a simple normalized key.
"""

import os
import glob
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Optional zstd availability (only used for capability hints)
try:
    import zstandard as _zstd  # type: ignore
    _ZSTD_AVAILABLE = True
except Exception:
    _ZSTD_AVAILABLE = False


def load_starcoder_languages(spec_path: Optional[str] = None) -> List[str]:
    import yaml
    if spec_path is None:
        spec_path = str(Path(__file__).resolve().parent / "bench/specs/starcoder.yaml")
        # If running from repo root, the relative above is wrong; try repo-root path
        if not os.path.exists(spec_path):
            spec_path = str(Path(__file__).resolve().parent / "bench/specs/starcoder.yaml")
            # ultimately, try two levels up (when executing from within scripts dir)
            if not os.path.exists(spec_path):
                spec_path = str(Path(__file__).resolve().parent / "bench/specs/starcoder.yaml")
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    cats = spec.get("categories", [])
    if not isinstance(cats, list) or not cats:
        raise ValueError("Invalid or empty categories in starcoder spec")
    return [str(c) for c in cats]


def _detect_data_files_for_language(root: str, language: str) -> Tuple[str, dict]:
    """
    Detect files under `<root>/<language>/**` and return (builder, kwargs) for datasets.load_dataset.
    If Parquet is present, prefer it; else accept JSONL(.zst/.gz) or JSON.
    """
    lang_dir = os.path.join(root, language)
    # Parquet
    parquet_files = glob.glob(os.path.join(lang_dir, "**", "*.parquet"), recursive=True)
    if parquet_files:
        return 'parquet', {'data_files': sorted(parquet_files)}
    # JSON patterns
    json_patterns: List[str] = []
    if _ZSTD_AVAILABLE:
        json_patterns.extend(['**/*.jsonl.zst', '**/*.json.zst'])
    json_patterns.extend(['**/*.jsonl.gz', '**/*.jsonl', '**/*.json.gz', '**/*.json'])
    json_files: List[str] = []
    for pat in json_patterns:
        json_files.extend(glob.glob(os.path.join(lang_dir, pat), recursive=True))
    if json_files:
        return 'json', {'data_files': sorted(json_files)}
    return 'parquet', {'data_files': []}


def write_jsonl(path: str, rows: List[dict], append: bool = True) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    mode = 'a' if append and os.path.exists(path) else 'w'
    with open(path, mode, encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def efficient_sample_from_jsonl_zst(
    file_paths: List[str],
    n_samples: int,
    seed: int,
    min_chars: int = 30,
    max_chars: Optional[int] = 4000,
    text_key: str = 'text'
) -> List[Dict[str, str]]:
    if not _ZSTD_AVAILABLE:
        raise ImportError("zstandard package required for .jsonl.zst files")
    rng = random.Random(seed)
    samples = []
    seen_texts = set()
    shuffled_files = file_paths.copy()
    rng.shuffle(shuffled_files)
    for file_path in shuffled_files:
        if len(samples) >= n_samples:
            break
        try:
            import subprocess
            result = subprocess.run(['zstd', '-d', '-c', file_path], capture_output=True, text=True, timeout=45)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if len(samples) >= n_samples:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    # Prefer 'content' if present (The Stack), else 'text'
                    raw = obj.get('content')
                    if not isinstance(raw, str):
                        raw = obj.get(text_key)
                    if not isinstance(raw, str):
                        continue
                    text = raw.strip()
                    if len(text) < min_chars:
                        continue
                    if max_chars and len(text) > max_chars:
                        text = text[:max_chars]
                    key = ' '.join(text.split()).lower()
                    if key not in seen_texts:
                        seen_texts.add(key)
                        samples.append({'text': text})
            else:
                continue
        except Exception:
            continue
    return samples


def fetch_language(
    language: str,
    n: int,
    seed: int,
    min_chars: int,
    max_chars: Optional[int],
    out_dir: str,
    stack_root: str,
) -> Tuple[str, int]:
    builder, kwargs = _detect_data_files_for_language(stack_root, language)
    data_files = kwargs.get('data_files', [])
    if not data_files:
        raise FileNotFoundError(f"No files found for language '{language}' under {stack_root}")

    # If jsonl.zst available, use efficient sampler; else fallback to datasets
    if any(f.endswith('.jsonl.zst') for f in data_files):
        rows = efficient_sample_from_jsonl_zst(
            file_paths=data_files,
            n_samples=n,
            seed=seed,
            min_chars=min_chars,
            max_chars=max_chars,
            text_key='text',
        )
    else:
        from datasets import load_dataset
        ds = load_dataset(builder, None, split='train', data_files=data_files)
        rng = random.Random(seed)
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        seen = set()
        rows: List[Dict[str, str]] = []
        for i in idxs:
            if len(rows) >= n:
                break
            ex = ds[int(i)]
            # Prefer 'content' (The Stack) then 'text'
            raw = ex.get('content')
            if not isinstance(raw, str):
                raw = ex.get('text')
            if not isinstance(raw, str):
                continue
            t = raw.strip()
            if len(t) < min_chars:
                continue
            if max_chars is not None and len(t) > max_chars:
                t = t[:max_chars]
            key = ' '.join(t.split()).lower()
            if key in seen:
                continue
            seen.add(key)
            rows.append({'text': t})

    out_file = os.path.join(out_dir, f"{language}.jsonl")
    write_jsonl(out_file, rows, append=True)
    return out_file, len(rows)


def main():
    ap = argparse.ArgumentParser(description='Fetch StarCoder language samples into local JSONL files.')
    ap.add_argument('--languages', nargs='+', default=None, help='Subset of languages; default loads all from spec.')
    ap.add_argument('--n_per_language', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min_chars', type=int, default=30)
    ap.add_argument('--max_chars', type=int, default=4000)
    ap.add_argument('--out_dir', type=str, default=str(Path(__file__).resolve().parent / 'data_samples' / 'starcoder'))
    ap.add_argument('--stack_root', type=str, required=True, help='Root directory with per-language data folders/files')

    args = ap.parse_args()

    languages = args.languages or load_starcoder_languages()

    results = []
    for lang in languages:
        try:
            out_file, m = fetch_language(
                language=lang,
                n=args.n_per_language,
                seed=args.seed,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                out_dir=args.out_dir,
                stack_root=args.stack_root,
            )
            results.append((lang, out_file, m))
            print(f"{lang}: appended {m} samples to {out_file}")
        except Exception as e:
            print(f"Warning: failed to fetch {lang}: {e}")

    print("\nSummary:")
    for lang, path, m in results:
        print(f"  {lang}: +{m} -> {path}")


if __name__ == '__main__':
    main()
