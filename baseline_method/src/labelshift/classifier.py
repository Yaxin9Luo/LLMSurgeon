from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import random
import os
from transformers import AutoConfig
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Optional heavy deps for HF classifier
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )
    from tqdm import tqdm
    _HF_AVAILABLE = True
except Exception:  # pragma: no cover - handled gracefully at runtime
    _HF_AVAILABLE = False


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def _nll_from_logits(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    probs = _softmax(logits / T, axis=1)
    # Gather p(y_i|x_i)
    n = probs.shape[0]
    py = probs[np.arange(n), y]
    py = np.clip(py, 1e-12, 1.0)
    return float(-np.mean(np.log(py)))


def fit_temperature(logits: np.ndarray, y: np.ndarray, Ts: Optional[List[float]] = None) -> float:
    """
    Fit a single temperature T>0 on validation logits by minimizing NLL.
    Uses a simple grid over log-space for robustness (no SciPy dependency).
    """
    if Ts is None:
        # logT in [-2, 2] -> T in [~0.14, ~7.39]
        Ts = [float(np.exp(t)) for t in np.linspace(-2.0, 2.0, 41)]
    best_T = 1.0
    best_nll = float("inf")
    for T in Ts:
        nll = _nll_from_logits(logits, y, T)
        if nll < best_nll:
            best_nll = nll
            best_T = T
    return float(best_T)


@dataclass
class TfidfLogReg:
    vectorizer: FeatureUnion
    clf: LogisticRegression
    temperature: float = 1.0

    @classmethod
    def create(
        cls,
        max_word_features: int = 200_000,
        max_char_features: int = 200_000,
        C: float = 4.0,
        seed: int = 0,
        n_jobs: int = 4,
        verbose: int = 0,
    ) -> "TfidfLogReg":
        word = TfidfVectorizer(ngram_range=(1, 2), max_features=max_word_features, lowercase=True)
        char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=max_char_features, lowercase=True)
        vec = FeatureUnion([("word", word), ("char", char)])
        clf = LogisticRegression(
            max_iter=500,
            solver="saga",
            multi_class="multinomial",
            C=C,
            random_state=seed,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        return cls(vectorizer=vec, clf=clf, temperature=1.0)

    def fit(self, X: List[str], y: List[int]) -> None:
        Xv = self.vectorizer.fit_transform(X)
        self.clf.fit(Xv, y)

    def logits(self, X: List[str]) -> np.ndarray:
        Xv = self.vectorizer.transform(X)
        # decision_function returns shape [n, K]
        return self.clf.decision_function(Xv)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        logits = self.logits(X)
        probs = _softmax(logits / self.temperature, axis=1)
        return probs

    def predict(self, X: List[str]) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def train_tfidf_classifier(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    seed: int = 0,
    n_jobs: int = 4,
    verbose: int = 0,
) -> Tuple[TfidfLogReg, Dict[str, float], np.ndarray]:
    """
    Train TF-IDF + LogisticRegression classifier and calibrate temperature on validation set.
    Returns: (model, metrics, confusion_matrix)
    """
    model = TfidfLogReg.create(seed=seed, n_jobs=n_jobs, verbose=verbose)
    model.fit(train_texts, train_labels)

    # Pre-calibration metrics
    val_logits = model.logits(val_texts)
    y_val = np.array(val_labels)
    # Fit temperature
    T = fit_temperature(val_logits, y_val)
    model.temperature = T

    # Post-calibration evaluation
    val_probs = _softmax(val_logits / T, axis=1)
    y_pred = np.argmax(val_probs, axis=1)
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    # Normalize rows to get P(y_hat=j|y=i)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    C = cm / row_sums

    metrics = {
        "val_acc": float(acc),
        "temperature": float(T),
    }
    return model, metrics, C


# -------------------------------
# DistilBERT / HF text classifier
# -------------------------------

class _TextClsDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        enc = self.tok(
            text,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


@dataclass
class HFSequenceClassifier:
    tokenizer: any
    model: any
    temperature: float = 1.0
    device: Optional[torch.device] = None
    max_length: int = 256

    def logits(self, X: List[str], batch_size: int = 32) -> np.ndarray:
        assert _HF_AVAILABLE, "HuggingFace transformers/torch not available"
        mdl = self.model
        tok = self.tokenizer
        device = self.device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        mdl.eval()
        mdl.to(device)
        ds = _TextClsDataset(X, None, tok, self.max_length)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=self._collate)
        outs = []
        with torch.no_grad():
            for batch in dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = mdl(**batch).logits  # [B, K]
                outs.append(logits.detach().cpu())
        Z = torch.cat(outs, dim=0).numpy() if outs else np.zeros((0, mdl.config.num_labels), dtype=np.float32)
        return Z

    def predict_proba(self, X: List[str], batch_size: int = 32) -> np.ndarray:
        z = self.logits(X, batch_size=batch_size)
        probs = _softmax(z / float(self.temperature), axis=1)
        return probs

    def embeddings(self, X: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Return mean-pooled token embeddings from the last hidden layer for each input.
        Shape: [N, hidden_size]. Uses attention mask to average only over non-pad tokens.
        """
        assert _HF_AVAILABLE, "HuggingFace transformers/torch not available"
        mdl = self.model
        tok = self.tokenizer
        device = self.device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        mdl.eval()
        mdl.to(device)
        ds = _TextClsDataset(X, None, tok, self.max_length)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=self._collate)
        outs = []
        with torch.no_grad():
            for batch in dl:
                am = batch.get("attention_mask")
                batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                out = mdl(**batch, output_hidden_states=True)
                # last hidden state [B, T, H]
                if hasattr(out, "hidden_states") and out.hidden_states is not None:
                    h = out.hidden_states[-1]
                else:
                    # Fallback: some heads return tuple
                    h = out[1] if len(out) > 1 else out[0]
                # Mean-pool over tokens using attention mask
                if am is None:
                    am = torch.ones(h.shape[:2], dtype=torch.long, device=h.device)
                else:
                    am = am.to(h.device)
                am = am.float()
                am = am.unsqueeze(-1)  # [B, T, 1]
                masked = h * am
                denom = torch.clamp(am.sum(dim=1), min=1e-6)  # [B,1]
                pooled = masked.sum(dim=1) / denom
                outs.append(pooled.detach().cpu())
        E = torch.cat(outs, dim=0).numpy() if outs else np.zeros((0, getattr(mdl.config, 'hidden_size', 768)), dtype=np.float32)
        return E

    @staticmethod
    def _collate(batch: List[Dict]):
        # Dynamic padding collator
        keys = [k for k in batch[0].keys() if k != "labels"]
        max_len = max(x["input_ids"].shape[0] for x in batch)
        pad_id = 0
        if "token_type_ids" in batch[0]:
            have_type_ids = True
        else:
            have_type_ids = False
        input_ids, attn_mask, type_ids, labels = [], [], [], []
        for item in batch:
            L = item["input_ids"].shape[0]
            pad = max_len - L
            input_ids.append(torch.nn.functional.pad(item["input_ids"], (0, pad), value=pad_id))
            attn_mask.append(torch.nn.functional.pad(item["attention_mask"], (0, pad), value=0))
            if have_type_ids:
                type_ids.append(torch.nn.functional.pad(item["token_type_ids"], (0, pad), value=0))
            if "labels" in item:
                labels.append(item["labels"])
        batch_out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn_mask),
        }
        if have_type_ids:
            batch_out["token_type_ids"] = torch.stack(type_ids)
        if labels:
            batch_out["labels"] = torch.stack(labels)
        return batch_out


def train_distilbert_classifier(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    *,
    model_name: str = "distilbert/distilbert-base-uncased",
    pretrained_dir: Optional[str] = None, 
    train_from_scratch: bool = False,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    max_length: int = 256,
    seed: int = 0,
) -> Tuple[HFSequenceClassifier, Dict[str, float], np.ndarray]:
    """
    Fine-tune a HuggingFace sequence classifier (default DistilBERT) for domain classification.
    Returns: (model, metrics, row-normalized confusion matrix)
    """
    if not _HF_AVAILABLE:
        raise ImportError("PyTorch/transformers not available. Install torch and transformers to use distilbert classifier.")

    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # Torch CPU/CUDA
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # cuDNN & matmul 行为固定
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    rng = np.random.RandomState(seed)
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    use_pretrained = pretrained_dir is not None and os.path.isdir(pretrained_dir)
    
    if use_pretrained:
        # Load a fine-tuned classifier from the specified directory
        print(f"[DistilBERT] Loading pretrained classifier from: {pretrained_dir}")
        tok = AutoTokenizer.from_pretrained(pretrained_dir)
        mdl = AutoModelForSequenceClassification.from_pretrained(pretrained_dir)
        mdl.to(device)
        mdl.eval()
        # 这里先不建 train_dl，只需要 val_dl，下面一步来建
        num_labels = int(getattr(mdl.config, "num_labels", max(max(train_labels), max(val_labels)) + 1))

    else:
        num_labels = int(max(max(train_labels), max(val_labels)) + 1)
        tok = AutoTokenizer.from_pretrained(model_name)
                
        # Ensure pad token
        if tok.pad_token is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            else:
                tok.add_special_tokens({"pad_token": "[PAD]"})

        if train_from_scratch:
            print(f"[DistilBERT] Training from scratch (random init) with config from: {model_name}")
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = num_labels
            mdl = AutoModelForSequenceClassification.from_config(config)
        else:
            print(f"[DistilBERT] Fine-tuning from pretrained model: {model_name}")
            mdl = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
            )


        for module in mdl.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

        # Ensure correct loss type: single-label classification with CrossEntropy
        try:
            mdl.config.problem_type = "single_label_classification"
        except Exception:
            pass
        # If tokenizer added tokens, resize embeddings
        if hasattr(mdl, "resize_token_embeddings"):
            mdl.resize_token_embeddings(len(tok))
            
        mdl.to(device)

        train_ds = _TextClsDataset(train_texts, train_labels, tok, max_length)

        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=HFSequenceClassifier._collate)


        optimizer = torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=weight_decay)
        num_training_steps = epochs * max(1, len(train_dl))
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * num_training_steps), num_training_steps=num_training_steps)

        mdl.train()
        for epoch in range(epochs):
            pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch in pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = mdl(**batch)
                loss = out.loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                # show instantaneous loss and lr in the progress bar
                try:
                    lr_curr = scheduler.get_last_lr()[0]
                except Exception:
                    lr_curr = lr
                pbar.set_postfix({"loss": f"{loss.detach().cpu().item():.4f}", "lr": f"{lr_curr:.2e}"})
        mdl.eval()

        if train_from_scratch:
            save_dir = "./cache/classifier/distillbert_scratch"
        else:
            save_dir = "./cache/classifier/distillbert_finetuned"

        os.makedirs(save_dir, exist_ok=True)
        tok.save_pretrained(save_dir)
        mdl.save_pretrained(save_dir)
        print(f"[DistilBERT] Saved classifier to: {save_dir}")

    val_ds = _TextClsDataset(val_texts, val_labels, tok, max_length)
    val_dl = DataLoader(
        val_ds,
        batch_size=max(8, batch_size),
        shuffle=False,
        collate_fn=HFSequenceClassifier._collate,
    )

    # Collect validation logits for calibration and metrics
    mdl.eval()

    all_logits = []
    y_val = np.array(val_labels)
    with torch.no_grad():
        for batch in val_dl:
            features = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = mdl(**features).logits
            all_logits.append(logits.detach().cpu())
    val_logits = torch.cat(all_logits, dim=0).numpy() if all_logits else np.zeros((0, num_labels), dtype=np.float32)

    # Temperature scaling on validation
    T = fit_temperature(val_logits, y_val)

    # Metrics and confusion
    val_probs = _softmax(val_logits / T, axis=1)
    y_pred = np.argmax(val_probs, axis=1)
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    C = cm / row_sums

    model = HFSequenceClassifier(tokenizer=tok, model=mdl, temperature=float(T), device=device, max_length=max_length)
    metrics = {"val_acc": float(acc), "temperature": float(T), "model_name": model_name}

    return model, metrics, C
