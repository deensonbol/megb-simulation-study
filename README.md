# Evaluating Mixed-Effects Gradient Boosting for Longitudinal Data Prediction

**Deen Sonbol**  
MS Thesis: Master of Science in Biostatistics  
Rutgers School of Public Health, May 2026  
Advisor: Dr. Shou-En Lu

---

## Overview

Longitudinal data, where the same individuals are measured repeatedly 
over time, introduce within-subject correlation that standard machine 
learning methods are not designed to handle. This thesis evaluates 
whether explicitly accounting for that correlation through 
**Mixed-Effects Gradient Boosting (MEGB)** leads to better predictive 
performance compared to a standard **Gradient Boosting Machine (GBM)**.

---

## Key Findings

- MEGB consistently outperformed GBM across all three prediction scenarios
- The largest gains occurred under **time-split forecasting**, where MEGB 
  achieved mean RMSE of ~1.21 vs ~2.79 for GBM at n = 20, with the gap 
  widening at larger sample sizes
- Under subject-level holdout and independent dataset evaluation, MEGB 
  still outperformed GBM, though with smaller margins — as expected, since 
  subject-specific random effects cannot be estimated for new individuals
- Performance advantages became more stable as sample size increased

---

## Simulation Design

| Parameter | Value |
|-----------|-------|
| Subjects (n) | 20, 50, 100 |
| Predictors (p) | 170 (6 truly relevant) |
| Time points | 10 per subject |
| Simulation replicates | 200 per condition |
| Signal type | Linear mixed-effects |
| Random intercept variance | 0.5 |
| Random slope variance | 3.0 |
| Residual noise SD | 0.5 |
| Predictor correlation (ρ) | 0.6 |

---

## Evaluation Scenarios

| Scenario | Description |
|----------|-------------|
| Subject-level holdout | 20% of subjects held out; MEGB uses fixed-effects only for new subjects |
| Time-split forecasting | Train on first 8 time points, predict last 2 within same subjects |
| Independent datasets | Train and test on entirely separate simulated datasets |

---

## Script Structure

| Part | Description |
|------|-------------|
| A | Single-run sanity check (training fit) |
| B | Subject-level holdout — 200 replicates × 3 sample sizes |
| C | Time-split evaluation — 200 replicates × 3 sample sizes |
| D | Visualizations (Figures 1–3) |
| E | Independent dataset evaluation |
| F | Hyperparameter tuning via 5-fold subject-level cross-validation |

---

## Output

Produces 5 figures saved as 300 dpi PNGs:
- `figure1_subject_holdout_rmse.png`
- `figure2_time_split_rmse.png`
- `figure3_time_split_difference_boxplot.png`
- `figure4_independent_dataset_rmse.png`
- `figure5_tuned_test_rmse.png`

---

## Requirements

```r
install.packages(c("gbm", "ggplot2"))
```

## How to Run

Open `simulation_study.R` in R and run Parts A through F sequentially.
Parts B, C, and E each run 200 replicates across 3 sample sizes and 
may take several minutes.

---

## Reference

Olaniran, O. R., et al. (2025). Mixed effect gradient boosting for 
high-dimensional longitudinal data. *Scientific Reports*, 15(1), 30927.
https://doi.org/10.1038/s41598-025-16526-z
