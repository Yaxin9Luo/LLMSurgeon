import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from data_utils import _read_jsonl


# Category order from bench/specs/pythia.yaml (used for stable output ordering)
PYTHIA_CATEGORIES_ORDERED: List[str] = [
    "CommonCrawl",
    "GitHub",
    "Wikipedia",
    "Books3",
    "BookCorpus2",
    "Gutenberg (PG-19)",
    "Arxiv",
    "StackExchange",
    "PubMed Central",
    "OpenWebText2",
    "FreeLaw",
    "USPTO Backgrounds",
    "PubMed Abstracts",
    "OpenSubtitles",
    "DM Mathematics",
    "Ubuntu IRC",
    "EuroParl",
    "HackerNews",
    "YoutubeSubtitles",
    "PhilPapers",
    "NIH ExPorter",
    "Enron Emails",
]


# Mapping from `data_samples/pile/*.jsonl` filenames to Pythia taxonomy names.
# (Only a subset is present in this repo; missing categories are simply not loaded.)
PILE_FILE_TO_PYTHIA_CAT: Dict[str, str] = {
    "pile_cc.jsonl": "CommonCrawl",
    "github.jsonl": "GitHub",
    "wikipedia_en.jsonl": "Wikipedia",
    "gutenberg_pg_19.jsonl": "Gutenberg (PG-19)",
    "arxiv.jsonl": "Arxiv",
    "stackexchange.jsonl": "StackExchange",
    "pubmed_central.jsonl": "PubMed Central",
    "freelaw.jsonl": "FreeLaw",
    "uspto_backgrounds.jsonl": "USPTO Backgrounds",
    "pubmed_abstracts.jsonl": "PubMed Abstracts",
    "dm_mathematics.jsonl": "DM Mathematics",
    "ubuntu_irc.jsonl": "Ubuntu IRC",
    "europarl.jsonl": "EuroParl",
    "hackernews.jsonl": "HackerNews",
    "philpapers.jsonl": "PhilPapers",
    "nih_exporter.jsonl": "NIH ExPorter",
    "enron_emails.jsonl": "Enron Emails",
}


@dataclass
class Split:
    texts: List[str]
    labels: List[int]


@dataclass
class DatasetSplits:
    categories: List[str]
    train: Split
    val: Split


def detect_available_pythia_categories(local_dir: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Detect which Pythia categories are available under local_dir.

    Returns:
      categories: ordered list of Pythia category names present
      file_to_cat: filename -> category mapping for files present
    """
    files = set(os.listdir(local_dir)) if os.path.isdir(local_dir) else set()

    mapping: Dict[str, str] = {}
    for fname, cat in PILE_FILE_TO_PYTHIA_CAT.items():
        if fname in files:
            mapping[fname] = cat

    # Preserve Pythia spec order, but include only categories with files
    cats_present = [c for c in PYTHIA_CATEGORIES_ORDERED if c in set(mapping.values())]
    return cats_present, mapping


def build_balanced_splits_pythia(
    local_dir: str,
    max_per_class: Optional[int] = 2000,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> DatasetSplits:
    """
    Load per-category JSONL files (Pile sources) and return balanced train/val splits
    aligned to Pythia taxonomy.
    """
    rng = random.Random(seed)
    categories, file_to_cat = detect_available_pythia_categories(local_dir)
    if not categories:
        raise FileNotFoundError(
            f"No Pythia/Pile category files detected under {local_dir}. "
            f"Expected one of: {sorted(PILE_FILE_TO_PYTHIA_CAT.keys())}"
        )

    # Aggregate texts per category
    cat_to_texts: Dict[str, List[str]] = {c: [] for c in categories}
    for fname, cat in file_to_cat.items():
        fpath = os.path.join(local_dir, fname)
        if not os.path.exists(fpath):
            continue
        texts = _read_jsonl(fpath)
        rng.shuffle(texts)
        if max_per_class is not None:
            texts = texts[:max_per_class]
        cat_to_texts[cat].extend(texts)

    sizes = [len(cat_to_texts[c]) for c in categories]
    min_size = min(sizes) if sizes else 0
    if min_size == 0:
        missing = [c for c in categories if len(cat_to_texts[c]) == 0]
        raise RuntimeError(
            "At least one Pythia category has zero samples. "
            f"Empty: {missing}. Ensure your local_samples_dir is populated."
        )

    # Balance by truncating each category to min size (same behavior as data_utils.build_balanced_splits)
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


