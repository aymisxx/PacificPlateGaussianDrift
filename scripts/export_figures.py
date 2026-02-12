from __future__ import annotations
import os
import argparse

from pacific_plate_gaussian_drift.io import load_volcanoes_csv, extract_age_distance
from pacific_plate_gaussian_drift.design_matrix import make_design_matrix
from pacific_plate_gaussian_drift.fit import fit_closed_form_ls, predict
from pacific_plate_gaussian_drift.metrics import residuals
from pacific_plate_gaussian_drift.plots import (
    ensure_dir,
    scatter_age_distance,
    fit_plot,
    residuals_vs_age,
    residual_hist,
)

def parse_args():
    p = argparse.ArgumentParser(description="Export figures for PacificPlateGaussianDrift.")
    p.add_argument("--csv", default=os.path.join("data", "volcanoes_data.csv"), help="Path to CSV file.")
    p.add_argument("--age_col", default="0.4", help="Age column name in CSV.")
    p.add_argument("--dist_col", default="0", help="Distance column name in CSV.")
    p.add_argument("--outdir", default=os.path.join("results", "figures"), help="Output directory for figures.")
    p.add_argument("--bins", type=int, default=10, help="Number of bins for residual histogram.")
    return p.parse_args()

def main():
    args = parse_args()

    df = load_volcanoes_csv(args.csv)
    age, distance = extract_age_distance(df, age_col=args.age_col, dist_col=args.dist_col)

    X = make_design_matrix(age)
    w_hat, _ = fit_closed_form_ls(X, distance)
    y_pred = predict(X, w_hat)
    r = residuals(distance, y_pred)

    ensure_dir(args.outdir)

    scatter_age_distance(
        age, distance,
        outpath=os.path.join(args.outdir, "age_vs_distance.png"),
        title="Age vs Distance — Pacific Plate Drift Data",
    )
    fit_plot(
        age, distance, y_pred,
        outpath=os.path.join(args.outdir, "least_squares_fit.png"),
        title="Least Squares Fit",
    )
    residuals_vs_age(
        age, r,
        outpath=os.path.join(args.outdir, "residuals_vs_age.png"),
        title="Residuals vs Age",
    )
    residual_hist(
        r,
        bins=args.bins,
        outpath=os.path.join(args.outdir, "residual_hist.png"),
        title="Residual Histogram",
    )

if __name__ == "__main__":
    main()