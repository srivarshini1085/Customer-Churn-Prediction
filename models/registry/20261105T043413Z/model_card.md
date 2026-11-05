# Churn model card — `20261105T043413Z`

- **Created:** 2026-11-05T04:34:13+00:00
- **Selected model:** `logistic_regression`  (`{'C': 1.0}`)
- **Selection metric:** roc_auc (stratified 5-fold CV)
- **Calibration:** sigmoid
- **git:** `c3922c9`  **python:** 3.13.7  **sklearn:** 1.9.1
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

## Fairness

Held-out test set, tuned threshold 0.35. Gaps above 0.10 are flagged for review.

| attribute | group | n | churn rate | recall | precision | FPR |
| --- | --- | --- | --- | --- | --- | --- |
| gender | Female | 687 | 0.281 | 0.684 | 0.567 | 0.204 |
| gender | Male | 722 | 0.251 | 0.751 | 0.544 | 0.211 |
| SeniorCitizen | 0 | 1187 | 0.233 | 0.670 | 0.535 | 0.177 |
| SeniorCitizen | 1 | 222 | 0.441 | 0.847 | 0.606 | 0.435 |
| Partner | No | 736 | 0.337 | 0.774 | 0.578 | 0.287 |
| Partner | Yes | 673 | 0.187 | 0.603 | 0.503 | 0.137 |

**Interpretation**

- gender: recall ranges from 0.68 (gender=Female) to 0.75 (gender=Male); all gaps are within 0.10. The smallest group (gender=Male) has 181 churners in the test set, so its recall is uncertain by about ±0.06 (95%).
- SeniorCitizen: recall ranges from 0.67 (SeniorCitizen=0) to 0.85 (SeniorCitizen=1); the gap in recall, false-positive rate exceeds 0.10 and is worth reviewing. Actual churn rates differ (0: 23%, 1: 44%), which by itself moves precision and FPR. The smallest group (SeniorCitizen=1) has 98 churners in the test set, so its recall is uncertain by about ±0.07 (95%).
- Partner: recall ranges from 0.60 (Partner=Yes) to 0.77 (Partner=No); the gap in recall, false-positive rate exceeds 0.10 and is worth reviewing. Actual churn rates differ (No: 34%, Yes: 19%), which by itself moves precision and FPR. The smallest group (Partner=Yes) has 126 churners in the test set, so its recall is uncertain by about ±0.09 (95%).

Recall is the most important metric here: a missed churner never receives a retention offer. When churn rates differ between groups, equal precision and equal FPR cannot both hold, so gaps there are expected and need context. Gender and SeniorCitizen are model inputs; removing them would not remove gaps on its own, because correlated features remain.

## Time to churn (survival analysis)

Cox proportional-hazards model with `tenure` as the duration and `Churn` as the event; `tenure`, `TotalCharges` are excluded because they encode the duration itself.

- Concordance index: **0.858** (train), **0.845** (test)
- Proportional-hazards violations (p < 0.05): `Contract=One year`, `InternetService=Fiber optic`, `MonthlyCharges_per_10`, `Partner=Yes`, `PaymentMethod=Credit card (automatic)`, `TechSupport=Yes`

| covariate | hazard ratio | 95% CI | p |
| --- | --- | --- | --- |
| InternetService=Fiber optic | 2.26 | 1.85-2.75 | 1.14e-15 |
| PaperlessBilling=Yes | 1.18 | 1.04-1.33 | 0.0081 |
| PaymentMethod=Mailed check | 1.08 | 0.92-1.25 | 0.345 |
| SeniorCitizen | 0.89 | 0.79-1.00 | 0.0565 |
| Dependents=Yes | 0.88 | 0.76-1.02 | 0.0888 |
| MonthlyCharges_per_10 | 0.85 | 0.82-0.89 | 3.91e-14 |
| TechSupport=Yes | 0.69 | 0.60-0.79 | 2.59e-07 |
| Partner=Yes | 0.60 | 0.53-0.67 | 5.89e-18 |
| OnlineSecurity=Yes | 0.56 | 0.49-0.64 | 7.26e-16 |
| PaymentMethod=Bank transfer (automatic) | 0.53 | 0.45-0.61 | 1.1e-16 |
| PaymentMethod=Credit card (automatic) | 0.50 | 0.43-0.59 | 5.11e-18 |
| InternetService=No | 0.32 | 0.25-0.41 | 1.64e-19 |
| Contract=One year | 0.21 | 0.18-0.25 | 3.21e-66 |
| Contract=Two year | 0.07 | 0.06-0.10 | 1.21e-85 |

Hazard ratio > 1 means customers with that attribute tend to churn sooner. A violated assumption means the effect changes with tenure, so its single hazard ratio is an average over time rather than a constant.

Limitations: one snapshot of data (tenure reconstructed from sign-up, one exit window), tenure observed only up to 72 months, covariates are current values, and churn rises again for customers past ~60 months, so long-tenure estimates deserve extra caution.
