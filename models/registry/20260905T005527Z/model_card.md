# Churn model card — `20260905T005527Z`

- **Created:** 2026-09-05T00:55:27+00:00
- **Selected model:** `logistic_regression`  (`{'C': 1.0}`)
- **Selection metric:** roc_auc (stratified 5-fold CV)
- **Calibration:** sigmoid
- **git:** `None`  **python:** 3.14.7  **sklearn:** 1.9.0
- **Training rows:** 5634  **Test rows:** 1409  **data sha256:** `16320c9c1ec7…`

## Decision threshold

Tuned to **0.35** (maximises churn-class F1 on out-of-fold predictions).

| metric | CV @ tuned threshold |
| --- | --- |
| precision | 0.569 |
| recall | 0.718 |
| f1 | 0.635 |

The fixed business risk bands (High >= 0.70, Moderate 0.40-0.69, Low < 0.40) are separate from this threshold and drive the recommended retention action.

## Held-out test performance

- ROC-AUC: **0.8417**   PR-AUC: **0.6349**   Brier: **0.1378** (lower is better; calibration quality)

| threshold | precision | recall | f1 | accuracy |
| --- | --- | --- | --- | --- |
| tuned (0.35) | 0.555 | 0.717 | 0.625 | 0.772 |
| default (0.50) | 0.648 | 0.508 | 0.570 | 0.796 |

## Top 15 feature weights

| feature | weight |
| --- | --- |
| tenure | -0.9833 |
| Contract = Two year | -0.8506 |
| InternetService = Fiber optic | +0.7060 |
| Contract = Month-to-month | +0.6983 |
| InternetService = DSL | -0.6590 |
| MonthlyCharges | -0.6148 |
| tenure_group = 12-24m | -0.3681 |
| tenure_group = 48m+ | +0.3066 |
| PaperlessBilling = No | -0.2712 |
| InternetService = No | -0.2430 |
| OnlineSecurity = No internet service | -0.2430 |
| OnlineBackup = No internet service | -0.2430 |
| DeviceProtection = No internet service | -0.2430 |
| TechSupport = No internet service | -0.2430 |
| StreamingTV = No internet service | -0.2430 |
