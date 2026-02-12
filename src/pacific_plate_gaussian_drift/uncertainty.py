from __future__ import annotations
import numpy as np

def sigma2_hat_from_residuals(r: np.ndarray, num_params: int) -> float:
    """
    Unbiased estimate of noise variance under i.i.d. Gaussian residuals:

        sigma^2_hat = (1 / (N - p)) * sum(r_i^2)

    where:
        N = number of samples
        p = number of parameters

    Raises:
        ValueError if degrees of freedom <= 0.
    """
    r = np.asarray(r, dtype=float).reshape(-1)
    dof = len(r) - int(num_params)
    if dof <= 0:
        raise ValueError(
            f"Degrees of freedom must be positive, got dof={dof}. Need N > num_params."
        )
    return float(np.sum(r**2) / dof)

def covariance_of_w(sigma2_hat: float, XtX_inv: np.ndarray) -> np.ndarray:
    """
    Cov(w_hat) = sigma^2_hat * (X^T X)^(-1)
    """
    XtX_inv = np.asarray(XtX_inv, dtype=float)
    return float(sigma2_hat) * XtX_inv

def standard_errors(cov_w: np.ndarray) -> np.ndarray:
    """
    Standard errors are sqrt of diagonal of covariance matrix.
    """
    cov_w = np.asarray(cov_w, dtype=float)
    return np.sqrt(np.diag(cov_w))

def ci95(value: float, se: float, z: float = 1.96):
    """
    95% CI under normal approximation: value ± z * se.
    """
    return float(value - z * se), float(value + z * se)

def kmmyr_to_cmyr(v_kmmyr: float) -> float:
    """
    Convert km/Myr to cm/year.

    1 km = 100000 cm
    1 Myr = 1,000,000 years
    => 1 km/Myr = 0.1 cm/year
    """
    return float(v_kmmyr * 0.1)