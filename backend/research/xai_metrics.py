"""Offline explanation diagnostics. None are prediction accuracy metrics."""
import numpy as np


def top_overlap(a, b, k=3):
    left = set(np.argsort(-np.abs(a), kind="stable")[:k].tolist())
    right = set(np.argsort(-np.abs(b), kind="stable")[:k].tolist())
    return len(left & right) / len(left | right) if left | right else 1.0


def removal_response(predict, original, reference, importance, temporal=False):
    """Mean |f(x)-f(masked)| for cumulative masks; synthetic inputs."""
    order = np.argsort(-np.abs(importance), kind="stable")
    masked = original.copy()
    points = []
    for index in order:
        if temporal:
            masked[index, :] = reference[index, :]
        else:
            masked[index] = reference[index]
        points.append(masked.copy())
    logits = np.asarray(predict(np.asarray(points)), dtype=float).reshape(-1)
    initial = float(np.asarray(predict(original[None])).reshape(-1)[0])
    return float(np.mean(np.abs(logits - initial)))


def random_response(predict, original, reference, temporal=False):
    rng = np.random.default_rng(42)
    n = original.shape[0]
    return float(np.mean([removal_response(predict, original, reference, rng.random(n), temporal)
                          for _ in range(20)]))


def day_occlusion(predict, original, reference):
    n = len(original)
    masked = np.repeat(original[None], n, axis=0)
    for i in range(n):
        masked[i, i, :] = reference[i, :]
    return float(predict(original[None])[0]) - np.asarray(predict(masked))
