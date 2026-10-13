"""Calibration and selective prediction for the verifier (numpy only).

* Temperature scaling: one scalar T fitted on validation logits to minimise NLL.
* ECE: expected calibration error, 15 equal-width bins.
* Risk-coverage: answer only when confidence >= tau; the threshold tau is chosen on validation
  data as the lowest value whose selective error stays under a target (e.g. 2%), which
  maximises how many checks the small model handles without sending them to the LLM.
"""

from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = logits / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, y: np.ndarray, t: float) -> float:
    p = softmax(logits, t)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    grid = np.concatenate([np.linspace(0.25, 1.0, 16), np.linspace(1.05, 10.0, 180)])
    losses = [nll(logits, y, t) for t in grid]
    return float(grid[int(np.argmin(losses))])


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(total)


def reliability(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> list[dict]:
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "confidence": float(conf[m].mean()),
                        "accuracy": float((pred[m] == y[m]).mean())})
    return out


def risk_coverage(probs: np.ndarray, y: np.ndarray) -> list[dict]:
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    out = []
    for tau in np.round(np.arange(0.30, 1.0, 0.025), 3):
        m = conf >= tau
        cov = float(m.mean())
        risk = float((pred[m] != y[m]).mean()) if m.any() else 0.0
        out.append({"tau": float(tau), "coverage": cov, "risk": risk})
    return out


def choose_tau(probs: np.ndarray, y: np.ndarray, target_risk: float = 0.02, min_n: int = 30) -> float:
    for row in risk_coverage(probs, y):
        if row["coverage"] * len(y) >= min_n and row["risk"] <= target_risk:
            return row["tau"]
    return 0.99


def metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    f1s = []
    for k in range(probs.shape[1]):
        tp = int(((pred == k) & (y == k)).sum())
        fp = int(((pred == k) & (y != k)).sum())
        fn = int(((pred != k) & (y == k)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"n": int(len(y)), "accuracy": float((pred == y).mean()), "macro_f1": float(np.mean(f1s)),
            "ece": ece(probs, y)}
