"""
notebooks/17_goemotions_validation.py
─────────────────────────────────────────────────────────────────────────────
Validates the pretrained emotion classifier (SamLowe/roberta-base-go_emotions)
against the official GoEmotions test split BEFORE using it on WSB text.

This is a model-choice sanity check, not something computed on our own data —
we have no human emotion labels for WSB posts, so the only way to know "does
this classifier behave sensibly" is to check it against the benchmark it was
actually trained on (proposal §6.6: "a sample of scored comments is
spot-checked against their assigned sentiment and emotion labels").

Reports macro-F1 across all 28 GoEmotions labels, plus per-label F1 for the
6 finance-relevant emotions QuantSight actually uses downstream
(fear, optimism, anger, excitement, confusion, disappointment).
"""

import json
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import pipeline
from sklearn.metrics import f1_score, classification_report

OUT_PATH = Path(__file__).resolve().parent.parent / "backend" / "data" / "predictions" / "goemotions_validation.json"

MODEL_NAME = "SamLowe/roberta-base-go_emotions"
FINANCE_EMOTIONS = ["fear", "optimism", "anger", "excitement", "confusion", "disappointment"]
THRESHOLD = 0.3  # per-label decision threshold for multi-label F1


def main():
    print(f"Loading GoEmotions test split ...")
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split="test")
    label_names = ds.features["labels"].feature.names  # 28 labels incl. neutral
    print(f"  {len(ds)} test examples, {len(label_names)} labels")

    print(f"\nLoading {MODEL_NAME} ...")
    clf = pipeline("text-classification", model=MODEL_NAME, top_k=None, truncation=True)

    texts = ds["text"]
    true_label_sets = ds["labels"]  # list of lists of label indices

    print(f"Scoring {len(texts)} test examples ...")
    batch_size = 32
    all_probs = np.zeros((len(texts), len(label_names)), dtype=np.float32)
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        results = clf(batch)
        for row_i, row_scores in enumerate(results):
            for item in row_scores:
                idx = label_to_idx.get(item["label"])
                if idx is not None:
                    all_probs[start + row_i, idx] = item["score"]
        if start % (batch_size * 20) == 0:
            print(f"  {start}/{len(texts)}")

    y_true = np.zeros((len(texts), len(label_names)), dtype=int)
    for i, labels in enumerate(true_label_sets):
        for l in labels:
            y_true[i, l] = 1

    y_pred = (all_probs >= THRESHOLD).astype(int)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    finance_f1 = {}
    for emo in FINANCE_EMOTIONS:
        idx = label_to_idx.get(emo)
        if idx is not None:
            finance_f1[emo] = round(float(per_label_f1[idx]), 3)

    print("\n" + "=" * 70)
    print("GOEMOTIONS VALIDATION RESULTS")
    print("=" * 70)
    print(f"Macro-F1 (28 labels): {macro_f1:.3f}")
    print(f"Micro-F1 (28 labels): {micro_f1:.3f}")
    print("\nFinance-relevant emotion F1 scores:")
    for emo, f1 in finance_f1.items():
        print(f"  {emo:<15}: {f1}")

    results = {
        "model": MODEL_NAME,
        "test_examples": len(texts),
        "threshold": THRESHOLD,
        "macro_f1": round(float(macro_f1), 3),
        "micro_f1": round(float(micro_f1), 3),
        "finance_emotion_f1": finance_f1,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
