# Customer Churn Probability Prediction System

## Abstract

This project presents a machine learning pipeline for predicting customer churn probability in subscription-based business environments. Rather than producing binary churn classifications, the system outputs calibrated churn probabilities, enabling nuanced risk segmentation and data-driven retention strategies. The pipeline employs XGBoost with Bayesian hyperparameter optimisation via Optuna, isotonic probability calibration, and SHAP-based explainability to produce interpretable, actionable predictions. Using Ollama's `llama3.2` LLM model, churn risks factors are derived from the prediction phase through SHAP and used as input prompts to generate a reponse. The system is designed to be dataset-agnostic, supporting multiple companies and datasets through a schema-driven feature validation architecture.

---

## Table of Contents

- [Background](#background)
- [Datasets](#dataset)
- [System Architecture](#system-architecture)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Model Evaluation](#model-evaluation)
- [Prediction Simulation](#prediction-simulation)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Usage](#usage)
- [Results](#results)
- [Limitations and Future Work](#limitations-and-future-work)

---

## Background

Customer churn - the rate at which customers discontinue their relationship with a service provider - represents a significant challenge in subscription-based industries such as telecommunications, SaaS, and financial services. Acquiring a new customer is estimated to cost five to seven times more than retaining an existing one, making proactive churn identification a high-value business problem.

Existing approaches frequently frame churn prediction as a binary classification task, producing a hard yes/no label. This framing discards the probabilistic uncertainty inherent in customer behaviour and limits the ability to prioritise retention interventions. This system instead models churn as a probability estimation problem, producing a continuous risk score per customer that supports ranked intervention strategies and campaign targeting by risk tier.

---

## Dataset

To avoid any breaching of licenses, the datasets used in the prediction are not included in this repo but can be downloaded below:

Telco Customer Churn: 
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Customer Churn: 
https://www.kaggle.com/datasets/royjafari/customer-churn

However, the resampled datasets created during my initial test run are available for use.

---

## System Architecture

The system is composed of two independent pipelines:

### Training Pipeline (`train_pipeline.py`)

Responsible for data ingestion, feature engineering, model training, hyperparameter optimisation, and artifact persistence. Implements a two-pass training strategy:

- **Pass 1** - Trains on the full engineered feature set to measure XGBoost feature importances
- **Pass 2** - Retrains on the reduced feature set after pruning low-importance features (importance < 0.01), producing the final deployable model

### Prediction Pipeline (`predict_pipeline.py`)

Responsible for loading trained artifacts, applying identical feature engineering, generating churn probability scores, and producing SHAP-based explanations for each prediction. Outputs a CSV containing churn probability, risk tier, and top contributing risk factors per customer.

### Artifact Schema

Each trained model persists three artifacts to `models/<company>/`:

| File | Contents |
|---|---|
| `pipeline.pkl` | Fitted sklearn/imblearn pipeline including preprocessor and calibrated XGBoost model |
| `schema.json` | Feature list, ignore columns, and risk segmentation thresholds |
| `model_metrics.csv` | Probability-focused evaluation metrics from the held-out test set |

---

## Methodology

### Preprocessing

- Median imputation for numeric features
- Mode imputation for categorical features
- Standard scaling for numeric features
- One-hot encoding for categorical features

All preprocessing is encapsulated within the sklearn `Pipeline` object to prevent data leakage - fit parameters are derived exclusively from training data and applied statically to the test set.

### Model

`XGBClassifier` (XGBoost) is used as the base estimator. XGBoost's gradient boosting framework iteratively corrects residuals from previous trees, producing sharper decision boundaries than ensemble averaging methods such as Random Forest on tabular data.

### Probability Calibration

The XGBoost model is wrapped in `CalibratedClassifierCV` with isotonic regression calibration (`cv=3`). Raw XGBoost probability outputs are frequently overconfident. Isotonic calibration corrects this by fitting a monotonic function from predicted probabilities to observed frequencies, ensuring that a predicted probability of 0.7 genuinely reflects approximately 70% churn likelihood.

### Hyperparameter Optimisation

Hyperparameter search is performed using Optuna with the Tree-structured Parzen Estimator (TPE) sampler over 50 trials. Unlike random search, TPE builds a probabilistic model of the objective landscape and preferentially samples hyperparameter regions that have yielded strong results in prior trials. The objective function minimises out-of-fold Brier Score via `StratifiedKFold` cross-validation, directly optimising for probability quality rather than classification accuracy.

The search space is defined as follows:

| Hyperparameter | Range |
|---|---|
| `n_estimators` | 100 – 500 |
| `max_depth` | 3 – 8 |
| `learning_rate` | 0.01 – 0.2 (log scale) |
| `subsample` | 0.6 – 1.0 |
| `colsample_bytree` | 0.6 – 1.0 |
| `min_child_weight` | 1 – 10 |
| `gamma` | 0.0 – 1.0 |

---

## Feature Engineering

The following features are derived from raw input columns prior to model training:

| Feature | Description | Rationale |
|---|---|---|
| `charges_ratio` | Monthly charges / (Total charges + 1) | Captures billing consistency; high ratio indicates short tenure relative to spend |
| `tenure_x_monthly` | Tenure × Monthly charges | Interaction term; high cost with short tenure signals elevated early churn risk |
| `tenure_cohort` | Binned tenure: new (0–12), developing (12–36), established (36+) | Non-linear tenure effects; new customers churn for different reasons than established ones |
| `high_risk_contract` | Binary flag: month-to-month contract AND above-median monthly charge | Strongest single churn predictor combination in telecom datasets |
| `paperless_flag` | Binary encoding of paperless billing status | Strong standalone churn correlate in subscription services |
| `active_services` | Count of active subscribed services | Fewer services indicate lower switching cost and higher churn propensity |

All feature engineering is applied identically in both training and prediction pipelines. The prediction pipeline inherits the correct feature set automatically via schema validation.

---

## Model Evaluation

The system is evaluated exclusively on probability-quality metrics, consistent with the probability estimation objective:

| Metric | Description |
|---|---|
| **ROC-AUC** | Probability that the model ranks a random churner above a random non-churner. Primary ranking metric. |
| **PR-AUC** | Area under the precision-recall curve. More informative than ROC-AUC under class imbalance as it focuses on minority class retrieval. |
| **Log Loss** | Penalises confident incorrect probability estimates. Measures sharpness of the probability distribution. |
| **Brier Score** | Mean squared error between predicted probabilities and true binary outcomes. Direct measure of probability accuracy. |
| **Brier Skill Score** | Brier Score normalised against the no-skill baseline (predicting the mean churn rate for all customers). Represents percentage improvement over naive prediction. |

Binary classification metrics (accuracy, precision, recall, F1) are intentionally omitted as they require an arbitrary threshold decision that is not relevant to the probability estimation objective.

---

## Prediction Simulation

### Resampling

For simulation purposes, Scikit Learn's Resampling function can be used to generate synthetic datasets from the existing datasets. Using Scikit Learn's Resampling library ensures that the generated dataset represents real value distributions from the original dataset.

### Actionable Insights

Actionable insights: Response to generated churn risks inferred from SHAP are implemented in two separate methods. 

Rule-Based Response - A deterministic rule-based approach that maps customer feature patterns to predefined retention strategies using regular expression matching within tuples.

LLM-Generation - Churn risk factors prompt the local Ollama `llama3.2` model to generate a response.

| Factor | Rule-Based | LLM-Generation |
|---|---|---|
| **Simplicity** | Simpler | More Complex and reliant on locally hosted LLMs |
| **Flexibility** | Rigid and limited to manually set reponses | Several times more flexible |
| **Performance** | Faster in execution | Latency from LLM processing and generation |

---

## Project Structure

```
churn_project/
│
├── src/
│   ├── train_pipeline.py       # Model training, feature engineering, artifact persistence
│   └── predict_pipeline.py              # Probability prediction and SHAP explanation
│
├── models/
│   └── <company>/
│       ├── pipeline.pkl        # Trained model, Encoder, Imputer
│       ├── schema.json         # Feature schema and risk thresholds
│       └── model_metrics.csv   # Evaluation metrics
│
├── data/                       # Input datasets
├── results/                    # Prediction output CSVs
└── requirements.txt
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
optuna
shap
joblib
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python src/train_pipeline.py \
  --company <company_name> \
  --data <path_to_training_csv> \
  --test-size 0.2 \
  --n-trials 50
```

| Argument | Required | Description |
|---|---|---|
| `--company` | Yes | Company identifier, used to name the model output directory |
| `--data` | Yes | Path to labelled training CSV containing a Churn column |
| `--test-size` | No | Proportion of data held out for evaluation (default: 0.2) |
| `--n-trials` | No | Number of Optuna hyperparameter search trials (default: 50) |

### Prediction

```bash
python src/predict.py \
  --company <company_name> \
  --data <path_to_input_csv>
```

| Argument | Required | Description |
|---|---|---|
| `--company` | Yes | Company identifier, used to locate trained model artifacts |
| `--data` | Yes | Path to input CSV for scoring |
| `--demo` | No | Bootstrap-resample input data for demonstration purposes |
| `--sample-size` | No | Number of resampled rows when `--demo` is set (default: 500) |

### Output

Predictions are saved to `results/<company>_predictions_<timestamp>.csv` with the following columns:

| Column | Description |
|---|---|
| `customerID` | Customer identifier (if present in input data) |
| `Churn Probability` | Calibrated churn probability in range [0, 1] |
| `Churn Risk Level` | Risk tier: Low Risk (< 0.3), Medium Risk (0.3–0.7), High Risk (> 0.7) |
| `Key Risk Factors` | Top 3 SHAP-identified features driving the prediction |

---

## Results

The following results were obtained on the IBM Telco Customer Churn dataset:

| Metric | Score |
|---|---|
| ROC-AUC | 0.8465 |
| PR-AUC | 0.6537 |
| Log Loss | 0.4322 |
| Brier Score | 0.1354 |
| Brier Skill Score | 0.3057 |

The Brier Skill Score of 0.306 indicates the model produces churn probability estimates approximately 30.6% more accurate than a naive baseline of predicting the population churn rate for every customer. The ROC-AUC of 0.847 places the model within the "production-ready" range for telecom churn prediction tasks, consistent with published benchmarks on the same dataset (0.82–0.88).

---

## Limitations and Future Work

**Limitations**

- Feature engineering is partially domain-specific to telecom datasets. Columns such as `contract`, `paperlessBilling`, and service subscription flags are expected by name. Datasets from other industries may not trigger these engineered features, reducing predictive signal.
- The two-pass feature selection strategy uses importance scores from a single training run, which may be unstable across random seeds or folds. A more robust approach would aggregate importance across multiple runs before pruning.
- Model retraining overwrites existing artifacts, meaning historical model versions and their associated metrics are not preserved.

**Future Work**

- Generalise feature engineering to be fully schema-driven, allowing domain-specific features to be defined externally per company rather than hardcoded
- Implement model versioning to support performance tracking across retraining iterations
- Extend evaluation with a reliability diagram to visually assess calibration quality
- Explore cost-sensitive threshold selection using business-defined misclassification costs for downstream retention campaign targeting
- Implement GUI implementation to replace CLI reliance
