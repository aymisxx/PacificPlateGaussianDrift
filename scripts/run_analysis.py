from __future__ import annotations
import json
import os
import numpy as np

from pacific_plate_gaussian_drift.io import load_volcanoes_csv, extract_age_distance
from pacific_plate_gaussian_drift.design_matrix import make_design_matrix
from pacific_plate_gaussian_drift.fit import fit_closed_form_ls, fit_numpy_lstsq, predict
from pacific_plate_gaussian_drift.metrics import residuals, rmse, r2_score
from pacific_plate_gaussian_drift.uncertainty import (
    sigma2_hat_from_residuals,
    covariance_of_w,
    standard_errors,
    ci95,
    kmmyr_to_cmyr,
)

def main():
    csv_path = os.path.join("data", "volcanoes_data.csv")

    # Dataset-specific column names (your CSV headers)
    age_col = "0.4"
    dist_col = "0"

    df = load_volcanoes_csv(csv_path)
    age, distance = extract_age_distance(df, age_col=age_col, dist_col=dist_col)

    X = make_design_matrix(age)
    w_hat, XtX_inv = fit_closed_form_ls(X, distance)
    b_hat, v_hat = float(w_hat[0]), float(w_hat[1])

    # Verify with lstsq
    w_np, *_ = fit_numpy_lstsq(X, distance)

    y_pred = predict(X, w_hat)
    r = residuals(distance, y_pred)

    rmse_val = rmse(r)
    r2_val = r2_score(distance, y_pred)

    sigma2_hat = sigma2_hat_from_residuals(r, num_params=2)
    cov_w = covariance_of_w(sigma2_hat, XtX_inv)
    se_b, se_v = standard_errors(cov_w)

    v_ci = ci95(v_hat, se_v)
    v_ci_cmyr = (kmmyr_to_cmyr(v_ci[0]), kmmyr_to_cmyr(v_ci[1]))

    summary = {
        "b_hat_km": b_hat,
        "v_hat_km_per_myr": v_hat,
        "v_hat_cm_per_year": kmmyr_to_cmyr(v_hat),
        "v_95ci_km_per_myr": [float(v_ci[0]), float(v_ci[1])],
        "v_95ci_cm_per_year": [float(v_ci_cmyr[0]), float(v_ci_cmyr[1])],
        "rmse_km": rmse_val,
        "r2": r2_val,
        "sigma2_hat": sigma2_hat,
        "se_b_km": float(se_b),
        "se_v_km_per_myr": float(se_v),
        "numpy_lstsq_matches": bool(np.allclose(w_np, w_hat, atol=1e-10)),
    }

    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()