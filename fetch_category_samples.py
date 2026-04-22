import os
import glob
import argparse
import json
import random
from typing import Dict, List, Tuple, Optional, Callable, Union

from datasets import load_dataset

# Detect zstd availability for .zst files
try:
    import zstandard as _zstd  # type: ignore
    _ZSTD_AVAILABLE = True
except Exception:
    _ZSTD_AVAILABLE = False


# Preset sources per category with (dataset, config, split, text_extractor)
# Text extractor returns a string given a row dict.
def _identity_text(col: str) -> Callable[[dict], Optional[str]]:
    def f(ex: dict) -> Optional[str]:
        t = ex.get(col)
        if isinstance(t, str):
            return t
        return None
    return f


def _daily_dialog_text(ex: dict) -> Optional[str]:
    # Flatten a dialogue into a single paragraph
    dial = ex.get('dialog')
    if isinstance(dial, list):
        return '\n'.join(str(u) for u in dial if isinstance(u, str))
    return None


def _detect_data_files(dirpath: Union[str, List[str]]) -> Tuple[str, dict]:
    """Detect Parquet or JSONL files and return (builder, loader_kwargs).

    Returns a builder name ('parquet' or 'json') and a dict with 'data_files'. If no files
    are found, returns ('parquet', {'data_files': []}) so the caller can surface a clear error.
    """
    dirs = [dirpath] if isinstance(dirpath, str) else list(dirpath)
    parquet_files: List[str] = []
    for d in dirs:
        parquet_files.extend(glob.glob(os.path.join(d, '**', '*.parquet'), recursive=True))
    if parquet_files:
        return 'parquet', {'data_files': sorted(parquet_files)}

    # Build JSON patterns; include zst only if supported
    json_patterns: List[str] = []
    if _ZSTD_AVAILABLE:
        json_patterns.extend(['**/*.jsonl.zst', '**/*.json.zst'])
    json_patterns.extend(['**/*.jsonl.gz', '**/*.jsonl', '**/*.json.gz', '**/*.json'])
    json_files: List[str] = []
    for d in dirs:
        for pat in json_patterns:
            json_files.extend(glob.glob(os.path.join(d, pat), recursive=True))
    if json_files:
        return 'json', {'data_files': sorted(json_files)}

    return 'parquet', {'data_files': []}


def build_presets(slimpajama_root: Optional[str] = None):
    """Return presets mapping.

    If slimpajama_root is provided, map categories to RedPajama splits under that root,
    detecting Parquet vs JSONL automatically. Otherwise, default to local Mink++ caches
    and DailyDialog for conversation.
    Each value is a tuple: (dataset, config, split, extractor, loader_kwargs)
    where loader_kwargs is passed through to datasets.load_dataset.
    """
    if slimpajama_root:
        rp = slimpajama_root
        mapping = {
            'C4': os.path.join(rp, 'RedPajamaC4'),
            'CommonCrawl': os.path.join(rp, 'RedPajamaCommonCrawl'),
            'GitHub': os.path.join(rp, 'RedPajamaGithub'),
            'Wikipedia': os.path.join(rp, 'RedPajamaWikipedia'),
            'Books': os.path.join(rp, 'RedPajamaBook'),
            'StackExchange': os.path.join(rp, 'RedPajamaStackExchange'),
            'Arxiv': os.path.join(rp, 'RedPajamaArXiv'),
        }
        presets = {}
        for cat, d in mapping.items():
            builder, kwargs = _detect_data_files(d)
            presets[cat] = (builder, None, 'train', _identity_text('text'), kwargs)
        return presets
    else:
        print("No slimpajama_root provided!!!!!!, exiting")
        exit(1)

FILENAME_MAP = {
    'CommonCrawl': 'commoncrawl.jsonl',
    'Wikipedia': 'wikipedia.jsonl',
    'Books': 'books.jsonl',
    'C4': 'c4.jsonl',
    'GitHub': 'github.jsonl',
    'Arxiv': 'arxiv.jsonl',
    'StackExchange': 'stackexchange.jsonl',
}


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
    min_chars: int = 300,
    max_chars: Optional[int] = 2000,
    text_key: str = 'text'
) -> List[Dict[str, str]]:
    """
    Efficiently sample from multiple .jsonl.zst files without loading everything into memory.
    """
    if not _ZSTD_AVAILABLE:
        raise ImportError("zstandard package required for .jsonl.zst files")
    
    rng = random.Random(seed)
    samples = []
    seen_texts = set()
    
    # Simple approach: randomly sample from files
    print(f"Sampling from {len(file_paths)} files...")
    
    # Shuffle files for random sampling
    shuffled_files = file_paths.copy()
    rng.shuffle(shuffled_files)
    
    for file_path in shuffled_files:
        if len(samples) >= n_samples:
            break
            
        try:
            # Use subprocess to decompress and read
            import subprocess
            result = subprocess.run(['zstd', '-d', '-c', file_path], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if len(samples) >= n_samples:
                        break
                        
                    if line.strip():
                        try:
                            data = json.loads(line)
                            text = data.get(text_key, '')
                            if isinstance(text, str) and len(text.strip()) >= min_chars:
                                if max_chars and len(text) > max_chars:
                                    text = text[:max_chars]
                                
                                # Deduplicate
                                normalized_text = ' '.join(text.split()).lower()
                                if normalized_text not in seen_texts:
                                    seen_texts.add(normalized_text)
                                    samples.append({'text': text})
                                    
                                    if len(samples) >= n_samples:
                                        break
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
        except Exception as e:
            print(f"Warning: Error reading {file_path}: {e}")
            continue
    
    print(f"Successfully sampled {len(samples)} texts")
    return samples


def fetch_category(
    category: str,
    n: int,
    seed: int,
    min_chars: int,
    max_chars: Optional[int],
    out_dir: str,
    dataset: Optional[str] = None,
    config: Optional[str] = None,
    split: str = 'train',
    text_col: Optional[str] = None,
    slimpajama_root: Optional[str] = None,
) -> Tuple[str, int]:
    print(f"Fetching category: {category}")
    print(f"SlimPajama root: {slimpajama_root}")
    
    # Resolve preset or overrides
    presets = build_presets(slimpajama_root)
    loader_kwargs = {}
    if dataset is None:
        if category not in presets:
            available_cats = list(presets.keys())
            raise ValueError(f'Unknown category {category}. Available categories: {available_cats}')
        ds_id, ds_cfg, ds_split, extractor, loader_kwargs = presets[category]
        print(f"Using preset for {category}: ds_id={ds_id}, loader_kwargs keys={list(loader_kwargs.keys())}")
    else:
        ds_id, ds_cfg, ds_split = dataset, config, split
        extractor = _identity_text(text_col) if text_col else (lambda ex: None)
        print(f"Using custom dataset: {dataset}")

    # Check if we're dealing with SlimPajama data (jsonl.zst files)
    if ds_id in ('parquet', 'json') and slimpajama_root:
        data_files = loader_kwargs.get('data_files', [])
        print(f"Found {len(data_files)} data files for {category}")
        if data_files:
            print(f"First few files: {data_files[:3]}")
            zst_files = [f for f in data_files if f.endswith('.jsonl.zst')]
            print(f"Found {len(zst_files)} .jsonl.zst files")
            
        if data_files and any(f.endswith('.jsonl.zst') for f in data_files):
            print(f"Using efficient sampling for {category} from {len(data_files)} .jsonl.zst files...")
            rows = efficient_sample_from_jsonl_zst(
                file_paths=data_files,
                n_samples=n,
                seed=seed,
                min_chars=min_chars,
                max_chars=max_chars,
                text_key='text'
            )
            out_file = os.path.join(out_dir, FILENAME_MAP[category])
            write_jsonl(out_file, rows, append=True)
            return out_file, len(rows)
        
        # Check if no files were found for SlimPajama categories
        elif isinstance(data_files, list) and len(data_files) == 0:
            hint = '' if _ZSTD_AVAILABLE else " Hint: .jsonl.zst requires the 'zstandard' package (pip install zstandard)."
            raise FileNotFoundError(
                f'No data files found for category {category}. Check --slimpajama_root path and contents.' + hint
            )

    # Fallback to original method for other data sources  
    # Guard: surface a clear error if no files were detected for parquet/json datasets
    if ds_id in ('parquet', 'json') and not slimpajama_root:
        data_files = loader_kwargs.get('data_files')
        if isinstance(data_files, list) and len(data_files) == 0:
            raise FileNotFoundError(f'No data files found for category {category}. Check dataset configuration.')

    ds = load_dataset(ds_id, ds_cfg, split=ds_split, **loader_kwargs)

    # Sample indices
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    # Deduplicate within this batch via simple normalization
    seen = set()
    rows = []
    for i in idxs:
        if len(rows) >= n:
            break
        ex = ds[int(i)]
        text = extractor(ex)
        if not isinstance(text, str):
            continue
        text = text.strip()
        if len(text) < min_chars:
            continue
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]
        key = ' '.join(text.split()).lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({'text': text})

    out_file = os.path.join(out_dir, FILENAME_MAP[category])
    write_jsonl(out_file, rows, append=True)
    return out_file, len(rows)


def main():
    parser = argparse.ArgumentParser(description='Fetch category samples into local_samples to scale up data.')
    parser.add_argument('--categories', nargs='+', default=None,
                        help='One or more categories; use names in PRESETS or "all".')
    parser.add_argument('--n_per_category', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min_chars', type=int, default=300)
    parser.add_argument('--max_chars', type=int, default=2000)
    parser.add_argument('--out_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'data_samples'))

    # Optional overrides for a single category run
    parser.add_argument('--dataset', type=str, default=None, help='HF dataset name to override preset (single category only).')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--text_col', type=str, default=None)
    parser.add_argument('--slimpajama_root', type=str, default='/data/yaxin/data/SlimPajama-627B-DC/train', help='Root of SlimPajama-627B-DC/train to map categories to RedPajama splits via Parquet.')

    args = parser.parse_args()

    # Set default categories after parsing args so we have access to slimpajama_root
    if args.categories is None:
        args.categories = list(build_presets(args.slimpajama_root).keys())

    categories = args.categories
    if categories == ['all']:
        categories = list(build_presets(args.slimpajama_root).keys())

    results = []
    for cat in categories:
        if args.dataset and len(categories) > 1:
            raise ValueError('Dataset override only supported for single-category runs.')
        out_file, m = fetch_category(
            category=cat,
            n=args.n_per_category,
            seed=args.seed,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            out_dir=args.out_dir,
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            text_col=args.text_col,
            slimpajama_root=args.slimpajama_root,
        )
        results.append((cat, out_file, m))

    for cat, path, m in results:
        print(f'{cat}: appended {m} samples to {path}')


if __name__ == '__main__':
    main()
