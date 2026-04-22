import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def _read_jsonl(path: str) -> List[str]:
    rows: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text")
            if isinstance(t, str) and t.strip():
                rows.append(t)
    return rows


def load_starcoder_categories(spec_path: Optional[str] = None) -> List[str]:
    """Load canonical StarCoder language list from YAML spec."""
    import yaml
    if spec_path is None:
        spec_path = str(Path(__file__).resolve().parents[3] / "bench/specs/starcoder.yaml")
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    cats = spec.get("categories", [])
    if not isinstance(cats, list) or not cats:
        raise ValueError("Invalid or empty categories in starcoder spec")
    # Preserve order as in spec
    return [str(c) for c in cats]


def detect_language_files(samples_dir: str, languages: List[str]) -> Dict[str, str]:
    """
    Map each language to a local samples file path. Expects `<language>.jsonl` files.
    Returns a dict {language: path}. Languages without files are omitted.
    """
    mapping: Dict[str, str] = {}
    for lang in languages:
        base = os.path.join(samples_dir, f"{lang}.jsonl")
        if os.path.exists(base):
            mapping[lang] = base
            continue
        # Accept upper/lower-case variants just in case
        alt = os.path.join(samples_dir, f"{lang.lower()}.jsonl")
        if os.path.exists(alt):
            mapping[lang] = alt
            continue
    return mapping


@dataclass
class Split:
    texts: List[str]
    labels: List[int]


@dataclass
class DatasetSplits:
    categories: List[str]
    train: Split
    val: Split


def build_balanced_splits_starcoder(
    samples_dir: str,
    *,
    spec_path: Optional[str] = None,
    max_per_class: Optional[int] = 2000,
    val_fraction: float = 0.2,
    seed: int = 0,
    require_all: bool = True,
) -> DatasetSplits:
    """
    Load per-language JSONL files for StarCoder and return balanced train/val splits.
    Expects files named `<language>.jsonl` under `samples_dir` for languages from the spec.
    """
    rng = random.Random(seed)
    languages = load_starcoder_categories(spec_path)
    lang_to_file = detect_language_files(samples_dir, languages)

    if require_all and len(lang_to_file) != len(languages):
        missing = [c for c in languages if c not in lang_to_file]
        raise FileNotFoundError(
            f"Missing samples for {len(missing)} languages under {samples_dir}. E.g., {missing[:5]}"
        )

    categories = [c for c in languages if c in lang_to_file]
    if not categories:
        raise FileNotFoundError(f"No language files detected under {samples_dir}")

    # Read and cap per-category
    cat_to_texts: Dict[str, List[str]] = {c: [] for c in categories}
    for c in categories:
        path = lang_to_file[c]
        texts = _read_jsonl(path)
        rng.shuffle(texts)
        if max_per_class is not None:
            texts = texts[:max_per_class]
        cat_to_texts[c].extend(texts)

    # Balance by truncating to min size
    sizes = [len(cat_to_texts[c]) for c in categories]
    min_size = min(sizes) if sizes else 0
    if min_size == 0:
        raise RuntimeError("At least one language has zero samples; ensure all selected languages are populated.")
    for c in categories:
        rng.shuffle(cat_to_texts[c])
        cat_to_texts[c] = cat_to_texts[c][:min_size]

    # Build splits
    train_texts: List[str] = []
    train_labels: List[int] = []
    val_texts: List[str] = []
    val_labels: List[int] = []

    for idx, c in enumerate(categories):
        texts = cat_to_texts[c]
        n = len(texts)
        n_val = max(1, int(n * val_fraction))
        val = texts[:n_val]
        train = texts[n_val:]
        train_texts.extend(train)
        train_labels.extend([idx] * len(train))
        val_texts.extend(val)
        val_labels.extend([idx] * len(val))

    # Shuffle within splits
    def _shuffle_pair(a: List[str], b: List[int]) -> Tuple[List[str], List[int]]:
        idxs = list(range(len(a)))
        rng.shuffle(idxs)
        return [a[i] for i in idxs], [b[i] for i in idxs]

    train_texts, train_labels = _shuffle_pair(train_texts, train_labels)
    val_texts, val_labels = _shuffle_pair(val_texts, val_labels)

    return DatasetSplits(
        categories=categories,
        train=Split(texts=train_texts, labels=train_labels),
        val=Split(texts=val_texts, labels=val_labels),
    )

