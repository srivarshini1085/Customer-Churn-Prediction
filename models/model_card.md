# Churn model card

- **Generated:** 2026-09-05T05:43:06+05:30
- **Selected model:** `logistic_regression`
- **Best hyper-parameters:** `{'C': 1.0}`
- **Selection metric:** roc_auc (stratified 5-fold CV)

## Decision threshold

Tuned to **0.60** (maximises churn-class F1 on out-of-fold predictions).

| metric | CV @ tuned threshold |
| --- | --- |
| precision | 0.571 |
| recall | 0.716 |
| f1 | 0.635 |

The fixed business risk bands (High >= 0.70, Moderate 0.40-0.69, Low < 0.40) are separate from this threshold and drive the recommended retention action, not the yes/no label.

## Held-out test performance

- ROC-AUC: **0.8417**
- PR-AUC: **0.6349**

| threshold | precision | recall | f1 | accuracy |
| --- | --- | --- | --- | --- |
| tuned (0.60) | 0.560 | 0.714 | 0.627 | 0.775 |
| default (0.50) | 0.499 | 0.794 | 0.613 | 0.734 |

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
