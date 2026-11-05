# Interview notes

For each major component: what it does, why it was chosen over the alternatives,
and three questions an interviewer is likely to ask, with answers. Numbers are
from model version `20261105T043413Z` (see the README's Results section).

**The 30-second pitch:** "It's an end-to-end churn system. A calibrated model
predicts who will churn; signed explanations and a what-if search show why and
what would help; a survival model estimates when; and a budget optimiser decides
who gets which retention offer. It's served through FastAPI and Streamlit, every
model version ships with a model card, a fairness report and drift checks, and
it's typed, tested and runs in CI."

---

## 1. Data and the feature pipeline

**What it does.** `churn/data.py` loads the Telco CSV, drops `customerID`, turns
the mistyped `TotalCharges` into numbers (11 blanks, all customers with tenure 0)
and decodes `Churn` to 0/1. `churn/features.py` builds one scikit-learn
`Pipeline`: engineered features (`tenure_group`, `has_internet`, `num_addons`),
imputation, scaling and one-hot encoding with `handle_unknown="ignore"`.

**Why this design.** Putting every transformation inside the saved pipeline
removes train/serve skew: the API, CLI and dashboard all call the same object.
It also stops leakage during cross-validation, because imputation and scaling are
refit inside every fold. Engineered features are deliberately row-local (no
ratios like total/tenure) so a single API request with partial input still works.

**Likely questions**

- *Why not do the cleaning in pandas before training?*
  Then the serving code would have to repeat it exactly, and the CV would leak
  statistics (for example, the median used for imputation) from validation folds.
  Inside the pipeline both problems disappear.
- *What happens if a new category shows up in production?*
  `OneHotEncoder(handle_unknown="ignore")` encodes it as all zeros, so the model
  still scores; separately, input validation adds a warning to the response
  ("never seen in training").
- *How do you handle class imbalance (26.5% churn)?*
  `class_weight="balanced"` in every candidate, ROC-AUC and PR-AUC instead of
  accuracy for selection, and a tuned decision threshold. I did not use SMOTE
  because reweighting achieves the same goal without inventing customers.

---

## 2. Model selection, threshold and calibration

**What it does.** Logistic regression, random forest and histogram gradient
boosting are grid-searched with stratified 5-fold CV. The best ROC-AUC wins, with
ties (to 3 decimals) broken by recall; logistic regression won (CV ROC-AUC 0.8463
vs. 0.8464 for the forest, but recall 0.798 vs. 0.737). The winner is calibrated
with `CalibratedClassifierCV` (sigmoid, chosen by CV Brier score), and the
decision threshold is tuned on out-of-fold predictions to 0.35. Test: ROC-AUC
0.842, PR-AUC 0.635, Brier 0.138, recall 0.717 at precision 0.555.

**Why this design.** Accuracy rewards predicting "stays" for everyone. ROC-AUC
measures ranking, which is what targeting needs, and recall reflects the business
cost (a missed churner is lost revenue; a spare offer is cheap). Calibration
matters because the risk tiers and the retention optimiser's expected saving use
P(churn) directly, so a predicted 30% must really mean 30%.

**Likely questions**

- *Why did a simple logistic regression beat the tree models?*
  The signal in this data is mostly additive (contract, tenure, payment method,
  internet type), so the trees cannot exploit interactions much; all three are
  within 0.0014 ROC-AUC. Given a tie, the linear model is also cheaper and easier
  to explain, and the tie-break on recall chose it.
- *Why tune the threshold instead of using 0.5?*
  At 0.5 the model catches 190 of 374 test churners (recall 0.508); at 0.35 it
  catches 268 (0.717) while precision drops from 0.648 to 0.555. The threshold is
  chosen on out-of-fold training predictions, never on the test set.
- *How do you know the probabilities are calibrated?*
  The Brier score (0.138) and the reliability curve saved with the model: in each
  tenth of test customers, the observed churn rate is close to the predicted one.
  The dashboard's Model Performance page plots it against the diagonal.

---

## 3. Model registry and model card

**What it does.** Each training run is saved as `models/registry/<version>/`
with `model.joblib`, `survival.joblib`, `manifest.json` (git SHA, data SHA-256,
library versions, all metrics, fairness, survival results) and `model_card.md`.
`latest.json` points to the promoted version; `churn registry promote <v>` rolls
back.

**Why this design.** A file-based registry is enough for one model and keeps the
project dependency-free; the interface (save, load, list, promote) is the same
one MLflow exposes, so it could be swapped later. `latest.json` is a pointer file
rather than a symlink so it works on Windows.

**Likely questions**

- *Why not MLflow?*
  It would add a server and a dependency for features I don't need yet. The
  manifest already records what MLflow tracking would: params, metrics, data
  hash, code version. Moving to MLflow would only change `registry.py`.
- *How do you roll back a bad model?*
  `python -m churn registry promote <older version>` rewrites `latest.json`; the
  API loads the new pointer on restart. Every version stays on disk.
- *Why aren't the model files in git?*
  They are pickles: they can execute code when loaded and only load reliably with
  the same library versions. Manifests and model cards are committed; binaries
  are regenerated with `python -m churn train`, and the hosted demo trains on
  first start.

---

## 4. REST API

**What it does.** FastAPI app (`churn/service/`): `/predict`, `/predict/batch`,
`/explain/whatif`, `/retention/plan`, `/model`, `/health`, `/metrics`. The
request model is generated from `churn/schema.py`; unknown fields return 422,
too-large batches 413, a missing model 503, unexpected errors a generic 500.
Every response has a request id; Prometheus counts predictions and latency.
Measured latency: median 46 ms end to end (model call 22 ms), in-process.

**Why this design.** FastAPI gives validation and OpenAPI docs from type hints.
Generating the schema from one definition means the API, CLI flags and dashboard
form can never disagree about the 19 inputs.

**Likely questions**

- *How does the API load the model?*
  Once, in the FastAPI lifespan handler at startup, into app state; a dependency
  (`get_model`) returns it or raises 503 if loading failed, so `/health` reports
  "degraded" instead of the process crashing.
- *How did you test the API?*
  Contract tests for every endpoint check the exact response keys and types and
  every error body (422, 413, 503, 500), including that a 500 does not leak the
  exception text.
- *How would you scale it?*
  Scoring is CPU-bound and stateless, so run several uvicorn workers or replicas
  behind a load balancer; `/predict/batch` already vectorises. The dashboard can
  point at the API with `CHURN_API_URL` so the model is served in one place.

---

## 5. Per-customer explanations (coefficients and SHAP)

**What it does.** `ChurnModel.top_factors` returns signed contributions: for the
logistic regression, `scaled value x coefficient` per encoded feature; for tree
models, SHAP values from `TreeExplainer`. Both are ranked by magnitude and shown
as a red/green bar chart.

**Why this design.** Coefficient contributions are exact for a linear model and
free to compute, so SHAP would add cost without changing the answer. Trees have
no coefficients, and global feature importance is unsigned and not per-customer,
so SHAP is the standard choice; it is lazily imported and the explainer is not
pickled.

**Likely questions**

- *What does a contribution of +0.82 for tenure mean?*
  In log-odds units, this customer's short tenure pushes the score up by 0.82
  relative to an average customer. Positive raises risk, negative lowers it.
- *Explanations are computed on the uncalibrated model. Is that a problem?*
  Calibration (sigmoid) is a monotonic mapping, so the ranking and direction of
  factors are unchanged; only the probability scale differs.
- *How do you know the SHAP values are right?*
  A test checks SHAP's defining property: expected value plus the contributions
  equals the model's raw output for that customer.

---

## 6. What-if search (counterfactuals)

**What it does.** `churn/counterfactual.py` tries every combination of up to
three realistic changes (longer contract, automatic payment, add OnlineSecurity
or TechSupport, change paperless billing), scores them in one batch and returns
those that lower P(churn) most. Fixed attributes (gender, SeniorCitizen, Partner,
Dependents, tenure, TotalCharges) are never changed; contracts only get longer;
add-ons need internet; and in a multi-change suggestion every change must lower
the risk by at least 0.5 points.

**Why this design.** With five actionable features and at most two target values
each, three simultaneous changes is about 130 candidates, so exhaustive search is
exact and takes about 0.2 s. Optimisation-based counterfactual libraries (e.g.
DiCE) are built for large search spaces and would be harder to explain.

**Likely questions**

- *Does this mean switching the contract will keep the customer?*
  No. The model learned associations; the suggestion is a hypothesis to test with
  an experiment. The UI says so.
- *Why the "every change must pay its way" rule?*
  Without it, the top results are "the best single change plus something
  irrelevant", which is noise for an account manager.
- *How do you make sure gender is never changed?*
  Immutable and actionable features are separate lists in config, a test asserts
  they never overlap, the search skips immutable features even if misconfigured,
  and a Hypothesis property test checks random customers.

---

## 7. Retention budget optimiser

**What it does.** For each customer and offer: expected saving =
P(churn) x success rate x (MonthlyCharges x 12) − cost. Given a budget, choose at
most one offer per customer to maximise the total. Results (full file, assumed
offers): at a 20,000 budget the optimiser expects 108,600, greedy 99,975 and the
naive top-N-by-risk 51,507.

**Why this design.** It is a multiple-choice knapsack. I first built an exact
dynamic programme over the budget, but it needs costs rounded to budget steps and
that distorted the $10 offer by up to 40%. Lagrangian relaxation prices each
dollar at λ, lets every customer choose independently, and binary-searches λ to
fit the budget; it needs no rounding, runs in milliseconds, and the same λ gives
an upper bound, so each plan reports its share of the best possible (~100% on
the real data).

**Likely questions**

- *Why does greedy do so badly at a 60,000 budget (50% of the bound)?*
  It ranks offers by saving per dollar, so it gives most customers the cheap call
  and never upgrades high-value customers to the bigger bundle; once the cheap
  options run out it stops, leaving about 20,000 unspent.
- *Where do the offer costs and success rates come from?*
  They are labelled placeholders in `churn/config.py`. In practice costs come from
  finance and success rates from A/B tests of past campaigns; ideally an uplift
  model per customer.
- *Is the optimiser exact?*
  No: the gap is at most about one customer's saving. On thousands of customers
  that is invisible, but on a toy problem with 7 customers it can be large. A
  test compares it with brute force on small problems and checks the bound.

---

## 8. Time to churn (survival analysis)

**What it does.** `churn/survival.py` fits Kaplan-Meier curves by contract and a
Cox proportional-hazards model with `tenure` as duration and `Churn` as event.
Test concordance 0.845. Per customer it reports the median remaining months and
the chance of churning in the next 12 months, conditional on their current
tenure: S(t+u)/S(t).

**Why this design.** The classifier says *whether*, survival says *when*, and it
uses customers who haven't churned yet (censored) instead of discarding them.
Kaplan-Meier shows the raw data; Cox gives interpretable hazard ratios and
per-customer curves. `tenure` and `TotalCharges` are excluded as covariates
because they encode the duration.

**Likely questions**

- *What is the concordance index?*
  For pairs of customers, how often the one who churned first got the higher risk
  score: 0.5 random, 1.0 perfect. It is the survival analogue of ROC-AUC.
- *What does a hazard ratio of 0.075 for a two-year contract mean?*
  At any point in time, two-year customers have about 7.5% of the monthly churn
  risk of month-to-month customers, other covariates equal.
- *The proportional-hazards test fails for 6 covariates. Is the model useless?*
  No: the ranking (concordance) is still valid, but those hazard ratios are
  averages over time rather than constants. The fix would be stratifying by
  contract or a time-varying model. I also found churn is U-shaped in tenure,
  rising again past ~60 months in the raw data, which I report as a limitation.

---

## 9. Fairness report

**What it does.** `churn/fairness.py` computes recall, precision and
false-positive rate for each group of gender, SeniorCitizen and Partner on the
test set, flags gaps above 0.10, and writes a generated interpretation into the
model card. Result: gender within tolerance; SeniorCitizen and Partner flagged
(senior recall 0.847 vs. 0.670).

**Why this design.** Recall per group answers "are this group's churners being
missed?", the main harm here. Showing base rates matters because when churn rates
differ (seniors 44% vs. 23%), equal precision and equal FPR are mathematically
impossible together, so the report flags gaps for review instead of declaring
the model unfair.

**Likely questions**

- *The model is better at catching senior churners. Is that unfair?*
  The direction favours the group with the higher churn rate, which partly
  follows from the base rates. Whether it is acceptable is a business and
  ethical decision; my job is to measure and surface it.
- *Why not just drop gender and SeniorCitizen as features?*
  Other features are correlated with them (fiber, payment method), so dropping
  them would not remove the gap and could make it harder to measure.
- *How reliable are these numbers?*
  The senior group has only 98 churners in the test set, so its recall has a
  margin of about ±0.07; the report states this next to each attribute.

---

## 10. Monitoring and drift

**What it does.** At training, the model stores a snapshot of every feature's
distribution. The Monitoring page and `churn drift` compute the Population
Stability Index per feature for a new batch: green < 0.1, amber 0.1–0.2,
red ≥ 0.2. Retraining is recommended when any feature is red or three or more are
amber. Each prediction also carries input-validation warnings.

**Why this design.** PSI is a standard, label-free, easy-to-explain drift measure
with accepted rule-of-thumb thresholds. Thresholds and the retrain rule live in
config so they can be tuned.

**Likely questions**

- *What does PSI not catch?*
  Concept drift: the inputs can look the same while their relationship with churn
  changes. That needs fresh labels and monitoring of recall/calibration over time.
- *Why warn below 100 rows?*
  With few rows, the binned frequencies are noisy and PSI jumps around, so a red
  signal could be chance.
- *Would you retrain automatically on a red signal?*
  No: drift says the inputs changed, not that the model got worse. I would alert,
  check recent outcomes, then retrain through the normal pipeline and compare the
  new model card before promoting it.

---

## 11. Dashboard

**What it does.** Seven Streamlit pages in `churn/ui/` (one module each, shared
`components.py` and `theme.py`): Prediction (form, KPI cards, factors, time to
churn, what-if), Churn Analysis, Model Performance, Batch Prediction, Retention
Planner, Time to Churn and Monitoring. It uses the local model, or the API when
`CHURN_API_URL` is set.

**Why this design.** Streamlit keeps the UI in Python, close to the model code.
The original single 1,460-line file was split so each page can be read and tested
on its own; expensive work (model loading, scoring the sample file) is cached.

**Likely questions**

- *How do you test a Streamlit app?*
  With Streamlit's headless `AppTest`: a test opens every page and fails on any
  exception, and checks the prediction flow and the empty states.
- *How is the hosted demo given a model?*
  With `CHURN_AUTO_TRAIN=true` it trains once on first start (about 40 seconds
  locally), cached per server process, instead of committing pickles.
- *Why does the prediction form use `st.form`?*
  Without it, every widget change reruns the page; the form submits all 19 inputs
  at once.

---

## 12. Engineering quality

**What it does.** mypy `--strict` on `churn/`, ruff lint and format, 203 pytest
tests (unit, Hypothesis property tests, API contract tests, dashboard smoke
tests), 94% coverage with an 85% gate, pre-commit hooks, CI on Python 3.10–3.12
with separate lint, type-check, coverage and Docker-build jobs, Dependabot.

**Why this design.** Each tool catches a different class of bug cheaply: types
catch wrong shapes and missing cases before running, property tests find edge
cases I would not think of, contract tests protect API clients, and CI makes
every check automatic.

**Likely questions**

- *Give an example of a bug the tests caught.*
  The retention optimiser bought a $10 offer with a $0 budget: at exactly the best
  saving-per-cost price, floating-point rounding left one offer "worthwhile". A
  zero-budget test failed; the fix starts the price search safely above that.
  Another: a survival test assumed risk falls with tenure; it failed, and the data
  showed the U-shape, which became a documented limitation.
- *Why property-based tests?*
  For invariants that must hold for any input: engineered features stay in range,
  higher probability never gives a lower risk tier, the what-if search never
  changes a fixed attribute, and no retention plan ever overspends.
- *What is not covered by the 94%?*
  The Streamlit rendering code is excluded from the gate (it is smoke-tested
  instead), plus a few CLI branches such as `serve` and interactive prompts.
