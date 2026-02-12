import numpy as np
from pacific_plate_gaussian_drift.design_matrix import make_design_matrix
from pacific_plate_gaussian_drift.fit import fit_closed_form_ls, fit_numpy_lstsq

def test_closed_form_matches_lstsq():
    rng = np.random.default_rng(0)
    age = rng.uniform(0, 50, size=50)
    X = make_design_matrix(age)
    true_w = np.array([10.0, 2.0])
    y = (X @ true_w.reshape(-1, 1)).reshape(-1) + rng.normal(0, 0.1, size=50)

    w_cf, _ = fit_closed_form_ls(X, y)
    w_np, *_ = fit_numpy_lstsq(X, y)

    assert np.allclose(w_cf, w_np, atol=1e-8)