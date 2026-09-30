"""Agreement metrics between predicted rewards and human sync ratings."""

import numpy as np


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or len(y) != len(x):
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denom) if denom > 0 else 0.0


def spearman(x, y):
    return pearson(average_ranks(x), average_ranks(y))


def kendall_tau_b(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    concordant = discordant = tied_x_only = tied_y_only = 0
    for i in range(len(x) - 1):
        dx = np.sign(x[i] - x[i + 1:])
        dy = np.sign(y[i] - y[i + 1:])
        s = dx * dy
        concordant += int((s > 0).sum())
        discordant += int((s < 0).sum())
        tied_x_only += int(((dx == 0) & (dy != 0)).sum())
        tied_y_only += int(((dy == 0) & (dx != 0)).sum())
    denom = np.sqrt(float(concordant + discordant + tied_x_only) * float(concordant + discordant + tied_y_only))
    return float((concordant - discordant) / denom) if denom > 0 else 0.0


def pairwise_accuracy(target, prediction, min_label_gap=0.75):
    """Fraction of clip pairs with |target gap| >= min_label_gap ordered correctly; prediction ties count 0.5."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    correct, count = 0.0, 0
    for i in range(len(target)):
        gaps = target[i] - target[i + 1:]
        mask = np.abs(gaps) >= min_label_gap
        if not mask.any():
            continue
        products = np.sign(gaps[mask]) * np.sign(prediction[i] - prediction[i + 1:][mask])
        correct += float((products > 0).sum()) + 0.5 * float((products == 0).sum())
        count += int(mask.sum())
    return (correct / count if count else 0.0), count


METRICS = {
    "spearman": lambda t, p: spearman(p, t),
    "pearson": lambda t, p: pearson(p, t),
    "kendall": lambda t, p: kendall_tau_b(p, t),
    "pairwise_accuracy": lambda t, p: pairwise_accuracy(t, p)[0],
    "mae": lambda t, p: float(np.abs(np.asarray(p, dtype=np.float64) - np.asarray(t, dtype=np.float64)).mean()),
}


def compute_metrics(target, prediction):
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if len(target) == 0 or len(target) != len(prediction):
        raise ValueError("target and prediction must have equal non-zero length")
    out = {"n": int(len(target))}
    out.update({name: fn(target, prediction) for name, fn in METRICS.items()})
    out["pairwise_pairs"] = pairwise_accuracy(target, prediction)[1]
    return out


def bootstrap_ci(target, prediction, metrics=("spearman", "pairwise_accuracy"), num_samples=2000, seed=20260722):
    """Percentile bootstrap (resampling clips with replacement): {metric: {low, median, high}} at 95%."""
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in metrics}
    for _ in range(num_samples):
        idx = rng.integers(0, len(target), len(target))
        for name in values:
            values[name].append(METRICS[name](target[idx], prediction[idx]))
    return {name: {"low": float(np.percentile(v, 2.5)), "median": float(np.percentile(v, 50)),
                   "high": float(np.percentile(v, 97.5))} for name, v in values.items()}
