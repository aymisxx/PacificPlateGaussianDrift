# **PacificPlateGaussianDrift**
### Linear Regression & Gaussian Uncertainty Estimation of Pacific Plate Velocity

---

## Concept

The motion of tectonic plates can be reconstructed using **hotspot tracks**, volcanic chains formed as a plate moves over a relatively stationary mantle plume.

As the plate drifts:

- Older volcanoes lie farther from the active hotspot.  
- Distance grows approximately linearly with age.  

This project reconstructs that physical insight using:

- Linear algebra.  
- Least squares estimation.  
- Gaussian noise modeling.  
- Statistical uncertainty quantification.  
- Clean reproducible software engineering.  

It bridges **geophysics × probability × numerical methods × robotics-grade code structure**.

## Mathematical Model

We assume a linear drift model:

$$distance = b + v · Age + ε$$

Where:

- $b$ → intercept (km).  
- $v$ → plate velocity (km/Myr).  
- $ε$ ~ $N(0, σ²)$ → Gaussian measurement noise.  

### Least Squares Solution

$$ŵ = (XᵀX)⁻¹ Xᵀy$$

Implemented using:

$$np.linalg.solve(X_tX, X_tY)$$

to ensure numerical stability.

### Noise Variance Estimate

$$σ̂² = (1 / (N − p)) Σ rᵢ²$$

### Parameter Covariance

$$Cov(ŵ) = σ̂² (XᵀX)⁻¹$$

### 95% Confidence Interval

$$v̂ ± 1.96 · SE(v)$$

## Results

Estimated Pacific Plate velocity:

- $75.87 km/Myr$  
- $7.59 cm/year$  

95% CI: $[7.26, 7.92] cm/year$

Model Quality:

- $R²$ = 0.9835.
- $RMSE$ ≈ 201 km.
- Closed-form solution matches $NumPy$ solver.

## Discussion

- The linear model explains 98% of variance.
- Residuals are centered near zero.
- Linear drift assumption holds.
- Estimated velocity aligns with published Pacific Plate motion values (~7–11 cm/year).

## Repository Structure

```
PacificPlateGaussianDrift/
│
├── data/                      # Raw volcano dataset
│
├── notebook/                  # Exploratory analysis
│   └── pacific_plate_gaussian_drift.pdf               
│
├── results/
│   ├── summary.json           # Final computed metrics
│   └── figures/               # Generated plots
│
├── scripts/
│   ├── run_analysis.py        # Full regression + statistics
│   └── export_figures.py      # Plot generation CLI
│
├── src/pacific_plate_gaussian_drift/
│   ├── design_matrix.py
│   ├── fit.py
│   ├── metrics.py
│   ├── uncertainty.py
│   ├── io.py
│   ├── plots.py
│   └── __init__.py
│
├── tests/
│   └── test_fit_matches_lstsq.py
│
├── pyproject.toml
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

## How To Run

```cmd
python -m venv venv  
venv\Scripts\activate  
pip install -e .  
python -m scripts.run_analysis  
python -m scripts.export_figures  
```

### **Academic Context & Acknowledgment**

This micro-project was completed as part of:

**SES 598: Space Robotics & AI**  
Arizona State University  

**Instructor:** Prof. Jnaneshwar Das  
**GitHub:** https://github.com/darknight-007  

The course is affiliated with the  
**Distributed Robotic Exploration and Mapping Systems (DREAMS) Laboratory**  
**GitHub:** https://github.com/DREAMS-lab  
**Website:** https://deepgis.org/dreamslab  

The assignment/micro-project structure, evaluation methodology, and coverage-control framework were inspired by course material and lab research themes in autonomous systems and robotic exploration.

### **License**

Creative Commons Attribution 4.0 International License (CC-BY-4.0)

> see **LICENSE** file.

## Author

Ayushman M. (https://github.com/aymisxx)  
M.S. Robotics & Autonomous Systems (MAE)  
Arizona State University  

---