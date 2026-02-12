from __future__ import annotations
import numpy as np

def fit_closed_form_ls(X: np.ndarray, y: np.ndarray):
    """
    Closed-form Least Squares via normal equations:

        w_hat = (X^T X)^(-1) X^T y

    Implementation uses `solve` instead of explicit inversion.
    Also returns (X^T X)^(-1) for uncertainty estimates.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    XtX = X.T @ X
    XtY = X.T @ y

    # Solve XtX * w = XtY
    w_hat = np.linalg.solve(XtX, XtY).reshape(-1)

    # Compute XtX_inv via solve: XtX * A = I
    I = np.eye(XtX.shape[0], dtype=float)
    XtX_inv = np.linalg.solve(XtX, I)

    return w_hat, XtX_inv

def fit_numpy_lstsq(X: np.ndarray, y: np.ndarray):
    """Reference solver for verification."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    w, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
    return w.reshape(-1), residuals, rank, s

def predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float).reshape(-1, 1)
    return (X @ w).reshape(-1)