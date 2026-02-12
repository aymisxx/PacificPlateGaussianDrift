from __future__ import annotations
import pandas as pd
import numpy as np

def load_volcanoes_csv(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def extract_age_distance(df: pd.DataFrame, age_col: str, dist_col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract numeric age and distance arrays, coercing non-numeric values to NaN and dropping them.
    Returns:
        age (float ndarray), distance (float ndarray)
    """
    age = pd.to_numeric(df[age_col], errors="coerce")
    dist = pd.to_numeric(df[dist_col], errors="coerce")

    clean = pd.concat([age, dist], axis=1).dropna()

    age_np = clean[age_col].to_numpy(dtype=float)
    dist_np = clean[dist_col].to_numpy(dtype=float)
    return age_np, dist_np