"""
backend/train_transformer.py
─────────────────────────────────────────────────────────────────────────────
Trains TWO Transformer-encoder model variants:

    Variant A — TransformerEncoder (Finance Only)
    Variant B — TransformerEncoder (Finance + Sentiment)

ARCHITECTURE:
    Input → Linear projection → Sinusoidal positional encoding
          → TransformerEncoder (2 layers, self-attention) → Attention pooling
          → FC → Sigmoid

    The literature's most-hyped upgrade over plain LSTM (self-attention
    across the whole 10-day window instead of sequential recurrence). Kept
    lightweight (small d_model, few heads/layers) since SEQUENCE_LEN=10 is
    short — a large encoder would just overfit this dataset. Attention
    pooling on top of the encoder output (same convention as LSTMAttention
    and GRUAttention) keeps "which day mattered most" comparable across all
    three sequence models.

    NOTE — a real, documented risk going in: train_lstm.py's own comments
    record genuine training instability (seed-order sensitivity, collapse to
    all-BUY/all-SELL) on this exact dataset even with the simpler
    LSTM+Attention model. A full self-attention model has more parameters
    and no recurrence inductive bias, so it may be MORE prone to this, not
    less — if training collapses or badly overfits, the honest fix is
    smaller d_model / higher dropout / lower LR, not a bigger model. If this
    model doesn't beat LSTM+Attention or GRU, that is itself a legitimate,
    citable finding — several reviewed papers note simpler models
    outperforming Transformers on noisy financial series.

HOW TO RUN:
    python train_transformer.py

RUN THIS AFTER:  build_features.py
─────────────────────────────────────────────────────────────────────────────
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, classification_report,
    balanced_accuracy_score,
)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging

from cv_utils import (
    make_train_calib_split, pick_best_calibration, apply_calibrator,
    make_purged_folds, summarize_cv_metrics,
)

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("quantsight-transformer")

# ── Device ─────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

SEED = 42


def set_seed(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
PROCESSED_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR      = BASE_DIR / "data" / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"

FINANCE_CSV     = PROCESSED_DIR / "features_finance.csv"
SENTIMENT_CSV   = PROCESSED_DIR / "features_sentiment.csv"
METRICS_FILE    = PREDICTIONS_DIR / "model_metrics.json"

NON_FEATURE_COLS = {"ticker", "date", "signal", "Date", "Ticker"}

# ── Hyperparameters ────────────────────────────────────────────────────────────
SEQUENCE_LEN     = 10
D_MODEL          = 32       # small — SEQUENCE_LEN=10 is short, a big encoder just overfits
N_HEAD           = 4        # 32 / 4 = 8-dim per head
NUM_LAYERS       = 2
DIM_FEEDFORWARD  = 64
DROPOUT          = 0.3
BATCH_SIZE       = 256
EPOCHS           = 50
LR               = 5e-4
PATIENCE         = 8


# ── Dataset ────────────────────────────────────────────────────────────────────

class StockSequenceDataset(Dataset):
    """Converts flat feature DataFrame into overlapping N-day sequences,
    built per-ticker so sequences never cross stock boundaries."""

    def __init__(self, df: pd.DataFrame, feature_cols: list, seq_len: int):
        self.sequences = []
        self.labels    = []
        self.meta      = []

        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date").reset_index(drop=True)
            X     = group[feature_cols].values.astype(np.float32)
            y     = group["signal"].values.astype(np.float32)
            dates = group["date"].values

            for i in range(seq_len, len(group)):
                self.sequences.append(X[i - seq_len : i])
                self.labels.append(y[i])
                self.meta.append((ticker, str(dates[i])))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],    dtype=torch.float32),
        )


# ── Model Architecture ─────────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed (non-learned) positional encoding — no extra parameters, the
    simplest option for a sequence this short."""

    def __init__(self, d_model: int, max_len: int = SEQUENCE_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderModel(nn.Module):
    """
    Self-attention over the 10-day window. Flow:
        Input → Linear projection → +positional encoding
              → TransformerEncoder → Attention pooling → FC → Sigmoid

    Attention pooling on top (rather than e.g. taking only the last token)
    keeps the same "which day mattered most" explanation convention as
    LSTMAttention/GRUAttention, even though the encoder itself already uses
    self-attention internally — the pooling weights are what's exposed as
    this model's native, per-prediction explanation.
    """

    def __init__(self, n_features: int, d_model: int = D_MODEL,
                 nhead: int = N_HEAD, num_layers: int = NUM_LAYERS,
                 dim_feedforward: int = DIM_FEEDFORWARD, dropout: float = DROPOUT):
        super().__init__()

        self.input_proj   = nn.Linear(n_features, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=SEQUENCE_LEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor, return_logits: bool = False,
                return_attention: bool = False):
        # x: (batch, seq_len, features)
        x = self.input_proj(x)            # (B, T, d_model)
        x = self.pos_encoding(x)
        enc_out = self.transformer(x)     # (B, T, d_model)

        scores  = self.attention(enc_out)         # (B, T, 1)
        weights = torch.softmax(scores, dim=1)    # (B, T, 1)
        context = (weights * enc_out).sum(dim=1)  # (B, d_model)

        logits = self.classifier(context).squeeze(1)  # (B,)
        output = logits if return_logits else torch.sigmoid(logits)

        if return_attention:
            return output, weights.squeeze(-1)
        return output


class WeightedFocalLoss(nn.Module):
    """Identical to train_lstm.py's / train_gru.py's — same training dynamics
    comparison basis across all three sequence models."""
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal_term = (1 - p_t).clamp(min=1e-6) ** self.gamma
        return (focal_term * bce).mean()


# ── Training & Evaluation ──────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch, return_logits=True)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_probs  = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            logits  = model(X_batch, return_logits=True)
            probs   = torch.sigmoid(logits)
            loss    = criterion(logits, y_batch)
            total_loss += loss.item()
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return total_loss / len(loader), np.array(all_probs), np.array(all_labels)


def find_best_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    best_score     = 0.0
    best_threshold = 0.50
    for t in np.arange(0.30, 0.71, 0.05):
        preds = (probs >= t).astype(int)
        n_sell_predicted = (preds == 0).sum()
        n_buy_predicted  = (preds == 1).sum()
        if n_sell_predicted == 0 or n_buy_predicted == 0:
            continue
        score = balanced_accuracy_score(labels, preds)
        if score > best_score:
            best_score     = score
            best_threshold = round(t, 2)
    return best_threshold


# ── Metrics & Saving ───────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob, variant_name: str) -> dict:
    return {
        "model"     : "TransformerEncoder",
        "variant"   : variant_name,
        "accuracy"  : round(accuracy_score(y_true, y_pred) * 100, 2),
        "f1"        : round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "precision" : round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "recall"    : round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "auc_roc"   : round(roc_auc_score(y_true, y_prob) * 100, 2),
    }


def update_metrics_file(new_metrics: dict):
    all_metrics = {}
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as f:
            all_metrics = json.load(f)
    key = f"{new_metrics['model']}_{new_metrics['variant']}"
    all_metrics[key] = new_metrics
    with open(METRICS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"  Updated metrics → {METRICS_FILE.name}")


def save_predictions(meta: list, y_pred: np.ndarray,
                     y_prob: np.ndarray, out_path: Path):
    tickers = [m[0] for m in meta]
    dates   = [m[1] for m in meta]
    out_df  = pd.DataFrame({
        "ticker"           : tickers,
        "date"             : dates,
        "predicted_signal" : y_pred,
        "confidence"       : np.round(y_prob * 100, 2),
        "signal_label"     : ["BUY" if p == 1 else "SELL" for p in y_pred],
    })
    out_df.to_csv(out_path, index=False)
    logger.info(f"  Saved predictions → {out_path.name}")


def run_purged_cv_transformer(trainval_df: pd.DataFrame, feature_cols: list, n_folds: int = 3) -> dict:
    """Purged expanding-window walk-forward CV, training-period only —
    identical structure to train_lstm.py's / train_gru.py's."""
    CV_EPOCHS, CV_PATIENCE = 15, 4
    folds = make_purged_folds(trainval_df["date"], n_folds=n_folds, embargo_days=5)
    fold_metrics = []

    for i, (train_mask, val_mask) in enumerate(folds, start=1):
        fold_train_df = trainval_df.loc[train_mask].copy()
        fold_val_df   = trainval_df.loc[val_mask].copy()
        if len(fold_train_df) < SEQUENCE_LEN * 2 or len(fold_val_df) < SEQUENCE_LEN * 2:
            continue

        scaler = StandardScaler()
        fold_train_df[feature_cols] = scaler.fit_transform(fold_train_df[feature_cols])
        fold_val_df[feature_cols]   = scaler.transform(fold_val_df[feature_cols])

        train_ds = StockSequenceDataset(fold_train_df, feature_cols, SEQUENCE_LEN)
        val_ds   = StockSequenceDataset(fold_val_df,   feature_cols, SEQUENCE_LEN)
        if len(train_ds) == 0 or len(val_ds) == 0:
            continue
        if len(np.unique(train_ds.labels)) < 2 or len(np.unique(val_ds.labels)) < 2:
            continue

        set_seed(SEED + i)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = TransformerEncoderModel(n_features=len(feature_cols)).to(DEVICE)
        train_labels_arr = np.array(train_ds.labels)
        n_sell = (train_labels_arr == 0).sum()
        n_buy  = (train_labels_arr == 1).sum()
        pos_weight = torch.tensor([n_sell / max(n_buy, 1)], dtype=torch.float32).to(DEVICE)
        criterion = WeightedFocalLoss(pos_weight=pos_weight, gamma=2.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_loss, best_state, patience_count = float("inf"), None, 0
        for epoch in range(1, CV_EPOCHS + 1):
            train_epoch(model, train_loader, optimizer, criterion)
            val_loss, _, _ = evaluate(model, val_loader, criterion)
            if val_loss < best_loss:
                best_loss, best_state, patience_count = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                patience_count += 1
                if patience_count >= CV_PATIENCE:
                    break
        if best_state:
            model.load_state_dict(best_state)

        _, probs, labels = evaluate(model, val_loader, criterion)
        preds = (probs >= 0.5).astype(int)

        fold_metrics.append({
            "accuracy": round(accuracy_score(labels, preds) * 100, 2),
            "f1":       round(f1_score(labels, preds, zero_division=0) * 100, 2),
            "auc_roc":  round(roc_auc_score(labels, probs) * 100, 2),
        })
        logger.info(f"    CV fold {i}: acc={fold_metrics[-1]['accuracy']}% "
                    f"f1={fold_metrics[-1]['f1']}% auc={fold_metrics[-1]['auc_roc']}% "
                    f"(train={len(train_ds):,}, val={len(val_ds):,})")

    return summarize_cv_metrics(fold_metrics)


# ── Train One Variant ──────────────────────────────────────────────────────────

def train_variant(csv_path: Path, variant_name: str,
                  model_out: Path, pred_out: Path):
    logger.info(f"\n{'='*60}")
    logger.info(f"Training TransformerEncoder — {variant_name}")
    logger.info(f"{'='*60}")

    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    logger.info(f"  Features : {len(feature_cols)}")
    logger.info(f"  Rows     : {len(df):,}")
    logger.info(f"  Tickers  : {df['ticker'].nunique()}")

    split_test    = df["date"].quantile(0.80)
    trainval_df   = df[df["date"] <= split_test].copy()
    test_df       = df[df["date"] >  split_test].copy()

    logger.info("Running purged walk-forward CV (robustness check, training period only) ...")
    cv_summary = run_purged_cv_transformer(trainval_df, feature_cols, n_folds=3)
    if cv_summary:
        logger.info(
            f"  CV accuracy: {cv_summary['accuracy']['mean']}% ± {cv_summary['accuracy']['std']}%  |  "
            f"CV AUC: {cv_summary['auc_roc']['mean']}% ± {cv_summary['auc_roc']['std']}%"
        )

    train_mask, val_mask = make_train_calib_split(trainval_df["date"], calib_frac=0.15, embargo_days=5)
    train_df = trainval_df.loc[train_mask].copy()
    val_df   = trainval_df.loc[val_mask].copy()

    logger.info(
        f"  Train: {train_df['date'].min().date()} → "
        f"{train_df['date'].max().date()} ({len(train_df):,} rows)"
    )
    logger.info(
        f"  Val  : {val_df['date'].min().date()} → "
        f"{val_df['date'].max().date()} ({len(val_df):,} rows)"
    )
    logger.info(
        f"  Test : {test_df['date'].min().date()} → "
        f"{test_df['date'].max().date()} ({len(test_df):,} rows)"
    )

    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

    logger.info(f"Building {SEQUENCE_LEN}-day sequence windows ...")
    train_dataset = StockSequenceDataset(train_df, feature_cols, SEQUENCE_LEN)
    val_dataset   = StockSequenceDataset(val_df,   feature_cols, SEQUENCE_LEN)
    test_dataset  = StockSequenceDataset(test_df,  feature_cols, SEQUENCE_LEN)
    logger.info(f"  Train sequences : {len(train_dataset):,}")
    logger.info(f"  Val   sequences : {len(val_dataset):,}")
    logger.info(f"  Test  sequences : {len(test_dataset):,}")

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        logger.error("Not enough sequences. Need at least SEQUENCE_LEN rows per ticker.")
        return None

    set_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    n_features = len(feature_cols)
    model      = TransformerEncoderModel(n_features=n_features).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Model parameters: {total_params:,}")

    train_labels_arr = np.array(train_dataset.labels)
    n_sell = (train_labels_arr == 0).sum()
    n_buy  = (train_labels_arr == 1).sum()
    pos_weight_val = n_sell / max(n_buy, 1)
    pos_weight     = torch.tensor([pos_weight_val], dtype=torch.float32).to(DEVICE)
    logger.info(f"  SELL: {n_sell:,} | BUY: {n_buy:,} | pos_weight: {pos_weight_val:.3f}")
    criterion = WeightedFocalLoss(pos_weight=pos_weight, gamma=2.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    logger.info(f"Training for up to {EPOCHS} epochs (patience={PATIENCE}) ...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        thresh     = find_best_threshold(val_probs, val_labels)
        val_preds  = (val_probs >= thresh).astype(int)
        val_acc    = accuracy_score(val_labels, val_preds) * 100
        sell_count = (val_preds == 0).sum()

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                f"  Epoch {epoch:>3}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"Threshold: {thresh:.2f} | "
                f"SELL predicted: {sell_count:,}"
            )

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    _, best_val_probs, best_val_labels = evaluate(model, val_loader, criterion)
    best_threshold = find_best_threshold(best_val_probs, best_val_labels)

    calib_method, calibrator, brier_scores = pick_best_calibration(best_val_probs, best_val_labels)
    logger.info(f"  Calibration method chosen: {calib_method} (Brier scores: {brier_scores})")

    _, final_probs, final_labels = evaluate(model, test_loader, criterion)
    final_preds = (final_probs >= best_threshold).astype(int)
    calibrated_probs = apply_calibrator(calibrator, final_probs, calib_method)

    logger.info(f"  Best threshold: {best_threshold}")
    logger.info(f"  SELL predicted: {(final_preds==0).sum():,} / {len(final_preds):,}")
    logger.info(f"  BUY  predicted: {(final_preds==1).sum():,} / {len(final_preds):,}")

    metrics = compute_metrics(final_labels, final_preds, final_probs, variant_name)
    metrics["calibration_method"] = calib_method
    metrics["cv_robustness"] = cv_summary

    raw_conf   = np.where(final_preds == 1, final_probs, 1 - final_probs)
    calib_conf = np.where(final_preds == 1, calibrated_probs, 1 - calibrated_probs)
    correct    = (final_preds == final_labels).astype(int)
    metrics["calibration_check"] = {
        "raw_confidence_vs_correctness_corr":       round(float(np.corrcoef(raw_conf, correct)[0, 1]), 3),
        "calibrated_confidence_vs_correctness_corr": round(float(np.corrcoef(calib_conf, correct)[0, 1]), 3),
    }

    logger.info(f"\n  Final Results — TransformerEncoder ({variant_name}):")
    logger.info(f"  Accuracy  : {metrics['accuracy']}%")
    logger.info(f"  F1 Score  : {metrics['f1']}%")
    logger.info(f"  Precision : {metrics['precision']}%")
    logger.info(f"  Recall    : {metrics['recall']}%")
    logger.info(f"  AUC-ROC   : {metrics['auc_roc']}%")

    print(f"\nClassification Report — TransformerEncoder ({variant_name}):")
    print(classification_report(
        final_labels, final_preds,
        target_names=["SELL", "BUY"],
        zero_division=0
    ))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state"  : best_state,
        "scaler"       : scaler,
        "feature_cols" : feature_cols,
        "n_features"   : n_features,
        "threshold"    : best_threshold,
        "calibrator"   : calibrator,
        "calibration_method": calib_method,
        "hyperparams"  : {
            "seq_len"         : SEQUENCE_LEN,
            "d_model"         : D_MODEL,
            "n_head"          : N_HEAD,
            "num_layers"      : NUM_LAYERS,
            "dim_feedforward" : DIM_FEEDFORWARD,
            "dropout"         : DROPOUT,
        },
    }, model_out)
    logger.info(f"  Saved model → {model_out.name}")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    save_predictions(test_dataset.meta, final_preds, calibrated_probs, pred_out)
    update_metrics_file(metrics)

    return metrics


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if not FINANCE_CSV.exists():
        logger.error(f"Missing {FINANCE_CSV} — run build_features.py first")
        return

    metrics_a = train_variant(
        csv_path     = FINANCE_CSV,
        variant_name = "Finance Only",
        model_out    = MODELS_DIR / "transformer_finance.pt",
        pred_out     = PREDICTIONS_DIR / "transformer_finance_predictions.csv",
    )

    if not SENTIMENT_CSV.exists():
        logger.error(f"Missing {SENTIMENT_CSV} — run build_features.py first")
        return

    metrics_b = train_variant(
        csv_path     = SENTIMENT_CSV,
        variant_name = "Finance + Sentiment",
        model_out    = MODELS_DIR / "transformer_sentiment.pt",
        pred_out     = PREDICTIONS_DIR / "transformer_sentiment_predictions.csv",
    )

    if not metrics_a or not metrics_b:
        return

    print("\n-- TransformerEncoder Model Comparison ------------------------------")
    print(f"  {'Metric':<12} {'Finance Only':>16} {'Finance+Sentiment':>18} {'Delta':>10}")
    print(f"  {'-'*60}")
    for key in ["accuracy", "f1", "precision", "recall", "auc_roc"]:
        a_val = metrics_a[key]
        b_val = metrics_b[key]
        delta = b_val - a_val
        arrow = "UP" if delta > 0 else "DOWN"
        print(f"  {key:<12} {a_val:>15.2f}% {b_val:>17.2f}% {arrow}{abs(delta):>8.2f}%")
    print("--------------------------------------------------------------------")


if __name__ == "__main__":
    main()
