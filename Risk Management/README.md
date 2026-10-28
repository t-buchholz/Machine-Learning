# credit-migration-early-warning

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%E2%89%A51.2-orange)
![Data](https://img.shields.io/badge/data-synthetic-lightgrey)
![Regulatory model](https://img.shields.io/badge/regulatory%20model-no-red)

A machine learning **Early Warning System (EWS)** for corporate bond rating downgrades and credit migration risk. The project is aimed at institutional investment risk management and **Solvency II-style spread-risk analysis**.

> **Important:** All data is synthetic. The project is not investment advice. It is not a regulatory model. It does not implement the Solvency II Standard Formula.

---

## 1. Project Overview

The system answers one question: *which bonds in a portfolio are likely to be downgraded by at least one notch within 12 months, and what is the financial impact?*

The workflow has six steps:

1. **Generate** a synthetic portfolio of bonds with financial ratios, market signals, and a downgrade target.
2. **Split** the data by issuer in time order. One issuer appears in one split only.
3. **Train** a gradient boosting classifier with class weights. Compare it with a logistic regression baseline.
4. **Calibrate** the probabilities (isotonic regression) and **tune two decision thresholds** on validation data.
5. **Quantify risk**: Expected Loss, duration-based price impact, and portfolio VaR / ES from a Vasicek one-factor Monte Carlo model.
6. **Report**: metrics, tables, a saved model, and publication-quality figures.

The whole project is in **three files**:

| File | Content |
|---|---|
| `main.py` | Configuration, data generator, risk metrics, ML pipeline, visualizations, CLI, self-checks |
| `requirements.txt` | Python dependencies |
| `README.md` | This document |

---

## 2. Financial and Mathematical Background

### 2.1 Downgrade probability and default probability

The project keeps two concepts apart:

* **Downgrade probability** $p_i^{\downarrow}$: the model output. It is a calibrated probability of a downgrade of at least one notch within 12 months.
* **Default probability** $PD_i$: a fixed lookup by rating class (documented in `PD_BY_RATING`).

### 2.2 Expected Loss

$$EL_i = EAD_i \times PD_i \times LGD_i$$

* $EAD_i$ = market value + accrued interest. Accrued interest = annual coupon amount × `accrued_fraction`.
* $LGD_i$ = assumed loss given default by sector.

The **migration expected loss** adds the cost of spread widening after a downgrade:

$$EL^{mig}_i = p_i^{\downarrow} \times L^{\downarrow}_i , \qquad L^{\downarrow}_i = -\tfrac{N_i}{100}\,\Delta P_i$$

### 2.3 Modified duration and credit spread sensitivity

$$\Delta P \approx -D_{mod} \times P \times \Delta y \;+\; \tfrac{1}{2}\, C \times P \times \Delta y^{2}$$

$D_{mod}$ is the modified duration, $P$ the price, $\Delta y$ the yield or spread change (decimal), and $C$ the optional convexity. The downgrade shock $\Delta y$ depends on the rating class (`SHOCK_BPS_BY_RATING`).

### 2.4 Vasicek one-factor model (Gaussian copula)

Each bond $i$ has an asset return

$$X_i = \sqrt{\rho}\, Z + \sqrt{1-\rho}\, \varepsilon_i , \qquad Z, \varepsilon_i \sim \mathcal{N}(0,1)$$

A bond defaults if $X_i < \Phi^{-1}(PD_i)$. A bond is downgraded without default if $\Phi^{-1}(PD_i) \le X_i < \Phi^{-1}(p_i^{\downarrow})$. The conditional probability given the systematic factor is

$$p_i(z) = \Phi\!\left(\frac{\Phi^{-1}(p_i) - \sqrt{\rho}\, z}{\sqrt{1-\rho}}\right)$$

The simulation uses 50,000 runs, a 1-year horizon, and $\rho = 0.20$ by default.

### 2.5 Value at Risk and Expected Shortfall

$$VaR_{\alpha}(L) = \inf\{\ell : \mathbb{P}(L \le \ell) \ge \alpha\}, \qquad ES_{\alpha}(L) = \mathbb{E}\left[\,L \mid L \ge VaR_{\alpha}\,\right]$$

The code reports $\alpha \in \{95\%, 99\%, 99.5\%\}$.

### 2.6 Cost-sensitive classification

Errors have asymmetric costs:

$$C = c_{FN} \cdot FN + c_{FP} \cdot FP$$

$$c_{FN,i} = EAD_i \times LGD_i \times s , \qquad c_{FP,i} = c_{review} + b \times MV_i$$

* $s$ = `fn_loss_share` (share of the loss that a missed downgrade causes).
* $c_{review}$ = fixed review or de-risking cost. $b$ = bid-ask cost in basis points. $MV_i$ = market value.

### 2.7 Threshold optimization

The project selects two thresholds on **validation data only**:

$$t_{recall} = \arg\max_{t}\ \text{Precision}(t)\quad \text{subject to}\quad \text{Recall}(t) \ge 0.80$$

$$t_{cost} = \arg\min_{t}\ C(t)$$

### 2.8 Solvency II context

Under Solvency II, insurers hold capital for *spread risk* on corporate bonds. The risk depends on credit quality and duration. A downgrade raises the spread and the capital charge. An early warning on downgrades supports earlier de-risking and better capital planning. This project is **Solvency II-inspired** only. It does not compute the SCR.

---

## 3. Methodological Choices

| Topic | Choice |
|---|---|
| Target | `downgrade_12m` = 1 if the rating falls by at least one notch within 12 months |
| Split | Issuers are ordered by their first observation date: train, validation, test. No issuer in two splits |
| Class imbalance | `class_weight="balanced"` for both models |
| Calibration | Isotonic regression fitted on validation data |
| Hyperparameters | Small grid (learning rate, tree depth) scored on validation PR-AUC |
| Importance | Permutation importance (always). SHAP summary (if `shap` is installed) |
| Portfolio for risk | The test set, with out-of-sample calibrated probabilities |

---

## 4. Folder Structure

```
credit-migration-early-warning/
├── main.py            # complete project (see section map inside the file)
├── requirements.txt
├── README.md
└── outputs/           # created by main.py
    ├── metrics.json
    ├── risk_report.json
    ├── predictions.csv
    ├── rating_summary.csv
    ├── transition_matrix.csv
    ├── model.joblib
    └── *.png          # figures
```

---

## 5. Installation and Quickstart

```bash
git clone <your-repo-url>
cd credit-migration-early-warning
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install "shap>=0.43"           # optional: SHAP summary plot
```

Run the full workflow:

```bash
python main.py
```

Options:

| Option | Default | Meaning |
|---|---|---|
| `--seed` | 42 | Seed for all random generators |
| `--theme` | light | `light` or `dark` plots |
| `--output-dir` | outputs | Folder for all artifacts |
| `--n-simulations` | 50000 | Monte Carlo runs for VaR / ES |
| `--no-shap` | off | Skip the SHAP plot |
| `--skip-checks` | off | Skip the built-in self-checks |

The run executes built-in self-checks (EL identity, duration sign, VaR ≤ ES, seed reproducibility, base rate, no issuer leakage, recall target on validation). The exit code is `1` if a check fails.

Load the saved model:

```python
import joblib

bundle = joblib.load("outputs/model.joblib")
raw = bundle["pipeline"].predict_proba(new_bonds[bundle["features"]])[:, 1]
prob = bundle["calibrator"].predict(raw)
alert = prob >= bundle["threshold_recall"]
```

---

## 6. Output Artifacts and Sample Visualizations

Run `python main.py` to reproduce all results. Numbers differ by seed. This README does not quote metrics.

| Artifact | Content |
|---|---|
| `metrics.json` | AUC, PR-AUC, Brier scores, thresholds, precision / recall / cost per threshold, baseline |
| `risk_report.json` | EL, VaR and ES per confidence level |
| `rating_summary.csv` | Exposure and expected loss per rating class |
| `transition_matrix.csv` | Empirical 8 × 8 + Default transition matrix |
| `predictions.csv` | Test-set probabilities and decisions |
| `model.joblib` | Pipeline, calibrator, thresholds, feature list |

**Confusion matrices** (normalized and cost-weighted)

![Confusion matrices](outputs/confusion_matrices.png)

**Precision-Recall and ROC curves** with the chosen thresholds

![PR and ROC](outputs/pr_roc_curves.png)

**Permutation feature importance**

![Feature importance](outputs/feature_importance.png)

**SHAP summary** (only if `shap` is installed)

![SHAP summary](outputs/shap_summary.png)

**Exposure vs. expected loss and EL waterfall per rating class**

![Rating risk](outputs/rating_risk.png)

**Calibration curve**

![Calibration](outputs/calibration.png)

**Simulated loss distribution** with VaR and ES

![Loss distribution](outputs/loss_distribution.png)

---

## 7. Limitations and Disclaimer

* **Synthetic data.** The relations in the data come from the generator. Real credit data has different noise, regimes, and rating dynamics.
* **Small test set.** The test set has about 60 issuers and about 100 downgrade events. Test metrics change strongly with the seed. In development runs, test recall at the validation-tuned threshold was often **below** the 80% target. Threshold tuning on one validation set does not guarantee the recall target out of sample.
* **Regime drift.** The systematic factor changes the downgrade rate over time. A model trained in calm periods can underestimate risk later.
* **Simplified risk model.** The Vasicek model uses one factor, a constant correlation, fixed LGD, and a fixed spread shock per rating. It ignores multi-notch migration and spread dynamics in the simulation.
* **Not a regulatory model.** The project does not compute the Solvency II SCR. Do not use it for regulatory reporting.
* **Not investment advice.** Do not use it for real investment decisions.
