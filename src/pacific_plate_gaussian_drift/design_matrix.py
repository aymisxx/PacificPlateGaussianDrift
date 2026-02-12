from __future__ import annotations
import numpy as np

def make_design_matrix(age: np.ndarray) -> np.ndarray:
    """
    Create design matrix for affine model:
        distance = b + v * age + noise

    X = [1, age]
    """
    age = np.asarray(age, dtype=float).reshape(-1)
    return np.column_stack([np.ones_like(age), age])