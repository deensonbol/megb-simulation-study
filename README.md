# MEGB Simulation Study

Simulation study comparing Mixed Effects Gradient Boosting (MEGB) against standard GBM for longitudinal data.

## Structure

- **Part A** – Single-run sanity check (training fit)
- **Part B** – Subject-level holdout (generalization to new subjects)
- **Part C** – Time-split evaluation (predicting future time points)
- **Part D** – Plots
- **Part E** – Independent training/test datasets
- **Part F** – Hyperparameter tuning with 5-fold cross-validation

## Requirements

- R packages: `MEGB`, `gbm`, `ggplot2`
