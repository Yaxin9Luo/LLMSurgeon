import os
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# Canonical categories aligned with LLaMA-1 spec
CATEGORIES_7 = [
    "CommonCrawl",
    "C4",
    "GitHub",
    "Wikipedia",
    "Books",
    "Arxiv",
    "StackExchange",
]

# Merged-web variant (6 classes)
CATEGORIES_6 = [
    "Web",  # CommonCrawl + C4 merged
    "GitHub",
    "Wikipedia",
    "Books",
    "Arxiv",
    "StackExchange",
]

# OLMo3 categories (6 classes)
CATEGORIES_OLMO3 = [
    "CommonCrawl (Web pages)",
    "olmOCR science PDFs (Academic documents)",
    "Stack-Edu (GitHub Code)",
    "FineMath 3+",
    "Wikipedia & Wikibooks (Encyclopedic)",
    "Arxiv",
]


# Default mapping from local sample filenames to categories (6-class, merged web)
DEFAULT_FILE_TO_CAT_6: Dict[str, str] = {
    # Web pool (prefer a pre-merged file if present; else, pool C4 + CommonCrawl)
    "web.jsonl": "Web",  # preferred explicit merged file
    "commoncrawl.jsonl": "Web",
    "c4.jsonl": "Web",

    # Wikipedia
    "wikipedia.jsonl": "Wikipedia",

    # Books
    "books.jsonl": "Books",

    # GitHub Code
    "github.jsonl": "GitHub",

    # StackExchange (conversation/Q&A)
    "stackexchange.jsonl": "StackExchange",

    # arXiv Papers
    "arxiv.jsonl": "Arxiv",
}

# Optional mapping if user provides split web files (7-class)
DEFAULT_FILE_TO_CAT_7: Dict[str, str] = {
    # Split Web
    "commoncrawl.jsonl": "CommonCrawl",
    "c4.jsonl": "C4",

    # Wikipedia
    "wikipedia.jsonl": "Wikipedia",

    # Books
    "books.jsonl": "Books",

    # GitHub Code
    "github.jsonl": "GitHub",

    # StackExchange (conversation/Q&A)
    "stackexchange.jsonl": "StackExchange",

    # arXiv Papers
    "arxiv.jsonl": "Arxiv",
}

# Mapping for OLMo3
DEFAULT_FILE_TO_CAT_OLMO3: Dict[str, str] = {
    "commoncrawl.jsonl": "CommonCrawl (Web pages)",
    "olmpdf.jsonl": "olmOCR science PDFs (Academic documents)",
    "github.jsonl": "Stack-Edu (GitHub Code)",
    "finemath.jsonl": "FineMath 3+",
    "wikipedia.jsonl": "Wikipedia & Wikibooks (Encyclopedic)",
    "arxiv.jsonl": "Arxiv",
}


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


def detect_available_categories(local_dir: str, merge_web: bool = True) -> Tuple[List[str], Dict[str, str]]:
    """
    Detect categories based on files present in local_dir and desired merge setting.

    Returns (categories, file_to_cat_map)
    """
    files = set(os.listdir(local_dir)) if os.path.isdir(local_dir) else set()

    # Detect OLMo3
    if "finemath.jsonl" in files or "olmpdf.jsonl" in files:
        mapping = {k: v for k, v in DEFAULT_FILE_TO_CAT_OLMO3.items() if k in files}
        cats = sorted(
            set(mapping.values()),
            key=lambda c: (0, CATEGORIES_OLMO3.index(c)) if c in CATEGORIES_OLMO3 else (1, c),
        )
        return cats, mapping

    def _generic_mapping(src_files: set[str]) -> Tuple[List[str], Dict[str, str]]:
        mapping: Dict[str, str] = {}
        for fname in sorted(src_files):
            if not fname.lower().endswith(".jsonl"):
                continue
            stem = os.path.splitext(fname)[0]
            if not stem:
                continue
            mapping[fname] = stem
        cats = list(dict.fromkeys(mapping.values()))
        return cats, mapping
    if merge_web:
        # Prefer a single explicit merged web file if available. Always include other categories.
        mapping: Dict[str, str] = {}
        if "web.jsonl" in files or "webpages.jsonl" in files:
            # Add non-web categories from defaults
            for fname, cat in DEFAULT_FILE_TO_CAT_6.items():
                if cat == "Web":
                    continue  # skip raw web parts
                if fname in files:
                    mapping[fname] = cat
            # Add exactly one merged web file
            if "web.jsonl" in files:
                mapping["web.jsonl"] = "Web"
            else:
                mapping["webpages.jsonl"] = "Web"
        else:
            # Fall back to pooling C4 + CommonCrawl implicitly by including both raw files
            mapping = {k: v for k, v in DEFAULT_FILE_TO_CAT_6.items() if k in files}
        if mapping:
            cats = sorted(
                set(mapping.values()),
                key=lambda c: (0, CATEGORIES_6.index(c)) if c in CATEGORIES_6 else (1, c),
            )
            return cats, mapping
        # Fallback: treat each *.jsonl file as its own category
        return _generic_mapping(files)
    else:
        # Prefer split mapping; if not fully available, fall back gracefully
        mapping7 = {k: v for k, v in DEFAULT_FILE_TO_CAT_7.items() if k in files}
        # If neither commoncrawl nor c4 present, attempt to use pooled web as shared source (optional)
        if ("commoncrawl.jsonl" not in mapping7) and ("c4.jsonl" not in mapping7) and (
            ("webpages.jsonl" in files) or ("commoncrawl.jsonl" in files) or ("c4.jsonl" in files)
        ):
            # We'll treat webpages.jsonl as web pool; splitting is handled downstream if requested.
            if "webpages.jsonl" in files:
                mapping7["webpages.jsonl"] = "Web"
            else:
                # If only one of c4/commoncrawl exists, we will still proceed with partial split
                pass
        # Compute final category order
        if mapping7:
            cats_present = sorted(
                set(mapping7.values()),
                key=lambda c: (0, (CATEGORIES_7 + ["Web"]).index(c))
                if c in (CATEGORIES_7 + ["Web"])
                else (1, c),
            )
            return cats_present, mapping7
        # Fallback for arbitrary categories
        return _generic_mapping(files)


@dataclass
class Split:
    texts: List[str]
    labels: List[int]


@dataclass
class DatasetSplits:
    categories: List[str]
    train: Split
    val: Split


def build_balanced_splits(
    local_dir: str,
    merge_web: bool = True,
    max_per_class: Optional[int] = 2000,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> DatasetSplits:
    """
    Load per-category JSONL files and return balanced train/val splits.
    """
    rng = random.Random(seed)
    categories, file_to_cat = detect_available_categories(local_dir, merge_web=merge_web)
    if not categories:
        raise FileNotFoundError(f"No category files detected under {local_dir}")

    # Aggregate texts per category
    cat_to_texts: Dict[str, List[str]] = {c: [] for c in categories}

    # Special handling if user requests 7-class but only has a pooled web file
    pooled_web: Optional[List[str]] = None
    if not merge_web and "Web" in categories:
        pooled_candidates = [
            os.path.join(local_dir, "webpages.jsonl"),  # legacy pooled
        ]
        for pp in pooled_candidates:
            if os.path.exists(pp):
                pooled_web = _read_jsonl(pp)
                break

    for fname, cat in file_to_cat.items():
        fpath = os.path.join(local_dir, fname)
        if not os.path.exists(fpath):
            continue
        if not merge_web and cat == "Web":
            # Defer pooled web handling
            continue
        texts = _read_jsonl(fpath)
        rng.shuffle(texts)
        if max_per_class is not None:
            texts = texts[:max_per_class]
        if cat not in cat_to_texts:
            # Skip unexpected categories
            continue
        cat_to_texts[cat].extend(texts)

    # If split web requested but only pooled web exists, split pooled into CommonCrawl/C4 halves
    if not merge_web and "Web" in categories and pooled_web:
        rng.shuffle(pooled_web)
        half = len(pooled_web) // 2
        cc = pooled_web[:half]
        c4 = pooled_web[half:]
        # Only add categories that are actually expected
        if "CommonCrawl" in categories:
            cat_to_texts.setdefault("CommonCrawl", []).extend(cc[:max_per_class] if max_per_class else cc)
        if "C4" in categories:
            cat_to_texts.setdefault("C4", []).extend(c4[:max_per_class] if max_per_class else c4)

    # If merge_web=True and no explicit merged file was read, optionally pool C4 + CommonCrawl
    if merge_web and not cat_to_texts.get("Web"):
        web_pool: List[str] = []
        for fname in ("commoncrawl.jsonl", "c4.jsonl"):
            fpath = os.path.join(local_dir, fname)
            if os.path.exists(fpath):
                web_pool.extend(_read_jsonl(fpath))
        if web_pool:
            rng.shuffle(web_pool)
            if max_per_class is not None:
                web_pool = web_pool[:max_per_class]
            cat_to_texts.setdefault("Web", [])
            cat_to_texts["Web"].extend(web_pool)

    # Balance by truncating each category to min size
    sizes = [len(cat_to_texts[c]) for c in categories]
    min_size = min(sizes) if sizes else 0
    if min_size == 0:
        raise RuntimeError("At least one category has zero samples; ensure your local_samples directory is populated.")
    for c in categories:
        rng.shuffle(cat_to_texts[c])
        cat_to_texts[c] = cat_to_texts[c][:min_size]

    # Build train/val splits
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
