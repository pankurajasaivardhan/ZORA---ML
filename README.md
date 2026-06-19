# SENTINEL-ML

## Unified Multi-Domain Intelligent Risk Detection Platform

SENTINEL-ML is a production-grade machine learning platform that unifies six independent risk detection systems, each modeled on a real-world use case from a major technology or financial company. The platform demonstrates an end-to-end ML engineering workflow: research notebooks, production Python packages, a REST API, and a live web dashboard, all built around the same architecture across every module.

This project was built as a portfolio piece for the Google Pre-Doctoral Researcher Program application, with an emphasis on production-quality engineering practices rather than isolated notebook experiments.


## Project Philosophy

Most student ML projects stop at a single Jupyter notebook with a trained model. SENTINEL-ML instead follows the same lifecycle used by real engineering teams:

1. **Research layer** — exploratory data analysis and feature engineering in Jupyter notebooks
2. **Production layer** — the same logic rewritten as clean, modular, testable Python packages
3. **Serving layer** — a FastAPI REST API exposing every model as a versioned HTTP endpoint
4. **Application layer** — a React dashboard that calls the API and visualizes predictions in real time

Every module in this repository follows this same four-layer structure, so the architecture is consistent regardless of which domain (finance, healthcare, manufacturing, cybersecurity) the module belongs to.


## Architecture

```
SENTINEL-ML/
├── notebooks/              Research layer: EDA, feature engineering, model experiments
│   ├── fraud/
│   ├── loan_default/
│   ├── health_risk/
│   ├── equipment_failure/
│   ├── network_anomaly/
│   └── customer_churn/
│
├── modules/                Production layer: one Python package per domain
│   ├── fraud/
│   │   ├── config.py            All paths, hyperparameters, and constants
│   │   ├── logger.py            Structured logging
│   │   ├── data_loader.py       Loading and validating raw data
│   │   ├── preprocessor.py      Splitting, scaling, resampling
│   │   ├── feature_engineering.py
│   │   ├── models.py            Model class wrappers
│   │   ├── trainer.py           Cross-validation and training orchestration
│   │   ├── evaluator.py         Metrics and business cost optimization
│   │   ├── predictor.py         Single-record inference for production
│   │   └── pipeline.py          End-to-end runnable pipeline
│   ├── loan_default/        (same structure)
│   ├── health_risk/         (same structure)
│   ├── equipment_failure/   (same structure)
│   ├── network_anomaly/     (same structure)
│   └── customer_churn/      (same structure)
│
├── api/                    Serving layer: FastAPI application
│   ├── main.py              App entrypoint, lifespan startup, predictor loading
│   ├── routes.py            All REST endpoints
│   ├── schemas.py           Pydantic request/response models
│   └── dependencies.py      Predictor singletons, dependency injection
│
├── dashboard/               Application layer: React (CDN, no build step)
│   ├── index.html
│   ├── app.js                One component per module
│   └── styles.css            Production light-theme design system
│
├── data/                    Raw and processed datasets (not committed to git)
├── models/                  Trained model artifacts (not committed to git)
├── reports/                 SHAP plots and other generated reports
└── requirements.txt
```

Data files and trained model binaries are excluded from version control via `.gitignore` because of GitHub's file size limits; only code, notebooks, and small metadata/report JSON files are tracked.

---

## Modules

### 1. Fraud Detection — COMPLETE
**Modeled on:** JPMorgan, Goldman Sachs
**Dataset:** Kaggle credit card fraud dataset (284,807 transactions, 473 confirmed fraud after deduplication)
**Models:** XGBoost, LightGBM, Isolation Forest
**Best model:** XGBoost — AUC-ROC 0.9845, Recall 0.8243
**Key engineering decisions:**
- Temporal train/test split (not random) to prevent data leakage, mirroring how a bank would train on past transactions and evaluate on future ones
- Business cost matrix (false negative = -$500, false positive = -$10, true positive = +$200) used to select the operating threshold instead of defaulting to 0.5
- SHAP explainability included because flagged transactions must be explainable to regulators
- Engineered feature `V14_V17_euclidean` became the single most important feature, outperforming all 28 original PCA components

### 2. Loan Default Prediction — COMPLETE
**Modeled on:** Morgan Stanley, Citi
**Dataset:** Lending Club accepted loans, 2007–2018 (2.26 million records)
**Models:** XGBoost, LightGBM
**Best model:** LightGBM — AUC-ROC 0.6864, business value $315,249,500 on the test set
**Key engineering decisions:**
- Composite `risk_score` feature combining interest rate, grade, FICO score, and DTI became the top predictor
- Threshold deliberately tuned for high recall given the asymmetric cost of missing a defaulting loan (-$15,000) versus a false rejection (-$500)
- AUC is lower than other modules because the dataset contains loans that have not yet reached maturity, a known limitation of this public dataset that is disclosed openly rather than hidden

### 3. Health Risk Assessment — COMPLETE
**Modeled on:** Google Health, DeepMind
**Datasets:** UCI Heart Disease (1,025 patients), Pima Indians Diabetes (768 patients)
**Models:** SVM, Gradient Boosting, Naive Bayes (trained separately per condition, combined into a unified score)
**Best models:** SVM for heart disease (AUC 1.0 on a small 205-patient test set — disclosed as a small-dataset limitation rather than presented as a definitive result), Gradient Boosting for diabetes (AUC 0.819)
**Key engineering decisions:**
- Two independently trained models combined via a weighted score (60% heart, 40% diabetes) into a single household health risk output
- Zero-value imputation applied to physiologically impossible zeros in the diabetes dataset (e.g., zero blood pressure) before training

### 4. Equipment Failure Prediction — COMPLETE
**Modeled on:** Tesla, SpaceX
**Dataset:** NASA C-MAPSS turbofan engine degradation simulation (100 engines, run-to-failure)
**Models:** XGBoost Regressor, LightGBM Regressor, Random Forest Regressor
**Best model:** XGBoost — RMSE 76.5 cycles, business value $875,000 (all failures in the test set correctly flagged for maintenance)
**Key engineering decisions:**
- Predicts Remaining Useful Life (RUL) as a regression target, clipped at 125 cycles per the standard C-MAPSS convention
- Rolling-window statistics (mean, std, min, max over 5/10/20/30-cycle windows) and lag/diff features capture degradation trends rather than single-point sensor readings
- A composite `health_index` and `failure_risk` score were engineered and became the two most predictive features

### 5. Network Anomaly Detection — COMPLETE
**Modeled on:** Oracle, Microsoft
**Dataset:** NSL-KDD intrusion detection dataset (125,973 training connections, 22,544 test connections including attack types never seen in training)
**Models:** XGBoost, LightGBM, Random Forest
**Best model:** Random Forest — AUC-ROC 0.9644, F1 0.9334, business value $602,032,500
**Key engineering decisions:**
- Engineered `network_risk_score`, a weighted composite of error rates, service rarity, protocol type, and access-pattern anomalies, became the single most important feature by SHAP value
- Explicitly evaluated against attack types absent from the training set, the realistic scenario for any production intrusion detection system

### 6. Customer Churn Prediction — IN PROGRESS
**Modeled on:** Google, Salesforce
**Dataset:** Telco Customer Churn (7,043 customers)
**Models:** XGBoost, LightGBM, Logistic Regression
**Status:** Exploratory analysis and feature engineering complete. A data leakage issue was identified and corrected during feature engineering (an encoded version of the target column was briefly present among the candidate features and has been excluded before model training). Final model training, evaluation, production module, API integration, and dashboard page are the remaining steps.
**Findings so far:** Month-to-month contracts churn at 42.7% versus 2.8% for two-year contracts; customers in their first year churn at 47.7%; approximately $139,000 in monthly recurring revenue is currently at risk.



## Coming Soon

- Completion of the Customer Churn module (production code, API endpoint, dashboard page)
- Deeper analysis pass on the Fraud Detection module: learning curves and model calibration, to demonstrate depth alongside the platform's breadth

A seventh module, Stock Risk, was considered and intentionally excluded. Public market price prediction is a well-known, largely unsolved problem even for well-resourced quantitative funds, and including a weak module would undermine the credibility of the other six.



## Technology Stack

**Machine learning:** XGBoost, LightGBM, scikit-learn, imbalanced-learn (SMOTE), SHAP
**Backend:** Python, FastAPI, Pydantic, uvicorn
**Frontend:** React (via CDN, no build tooling), vanilla CSS
**Data processing:** pandas, NumPy

---

## Running the Platform Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train a module's models (example: fraud)
cd modules/fraud
python pipeline.py

# Start the API (serves all trained modules)
cd ../..
python api/main.py

# Open the dashboard
open dashboard/index.html
```

The API will be available at `http://localhost:8000`, with interactive documentation at `http://localhost:8000/docs`.



## Design Decisions Worth Noting

**Temporal splits over random splits.** Wherever the data has a time dimension (fraud, equipment sensors), the train/test split respects chronological order. A model is trained on the past and evaluated on the future, the same constraint a production system would face.

**Business cost optimization over accuracy.** Every module defines an explicit cost matrix (false negative cost, false positive cost, true positive reward) and selects its operating threshold by maximizing business value on the test set, not by defaulting to a 0.5 probability cutoff.

**Honest reporting of weaknesses.** Where a result looks too good (Heart Disease AUC of 1.0 on a small test set) or weaker than expected (Loan Default AUC of 0.69), the limitation is stated directly rather than omitted. A small, clean dataset producing a perfect score is a sign to investigate further, not a result to present uncritically.

**Shared architecture across domains.** Every module, regardless of industry, follows the identical eleven-file production structure. This was a deliberate constraint to demonstrate that the underlying engineering pattern generalizes, rather than building six unrelated one-off scripts.
