from __future__ import annotations
import numpy as np

def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return y_true - y_pred

def rmse(r: np.ndarray) -> float:
    r = np.asarray(r, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean(r**2)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R^2 = 1 - SS_res / SS_total

    If SS_total == 0 (all y_true identical), R^2 is undefined.
    We return 0.0 in that edge case (prevents division by zero).
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    ss_total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))

    if ss_total == 0.0:
        return 0.0

    return float(1.0 - (ss_res / ss_total))