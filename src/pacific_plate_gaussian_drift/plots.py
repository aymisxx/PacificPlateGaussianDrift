from __future__ import annotations
import os
import matplotlib.pyplot as plt

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def scatter_age_distance(age, distance, outpath=None, title="Age vs Distance"):
    plt.figure()
    plt.scatter(age, distance)
    plt.xlabel("Age (Myr)")
    plt.ylabel("Distance (km)")
    plt.title(title)
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()

def fit_plot(age, distance, y_pred, outpath=None, title="Least Squares Fit"):
    plt.figure()
    plt.scatter(age, distance)
    plt.plot(age, y_pred)
    plt.xlabel("Age (Myr)")
    plt.ylabel("Distance (km)")
    plt.title(title)
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()

def residuals_vs_age(age, residuals, outpath=None, title="Residuals vs Age"):
    plt.figure()
    plt.scatter(age, residuals)
    plt.axhline(0)
    plt.xlabel("Age (Myr)")
    plt.ylabel("Residual (km)")
    plt.title(title)
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()

def residual_hist(residuals, bins=10, outpath=None, title="Residual Histogram"):
    plt.figure()
    plt.hist(residuals, bins=bins)
    plt.xlabel("Residual (km)")
    plt.ylabel("Count")
    plt.title(title)
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=200)
    plt.show()
    plt.close()