import argparse
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

# -----------------------------
# Configuration
# -----------------------------

SEED        = 42
TEST_SIZE   = 0.2
IGNORE_COLUMNS = ["customerID", "Churn"]

LOW_RISK_THRESHOLD    = 0.3
MEDIUM_RISK_THRESHOLD = 0.7
HIGH_RISK_THRESHOLD     = 0.89

N_TRIALS    = 50
CV_FOLDS    = 5
OPTUNA_PARAM_SPACE = {
    "n_estimators":     (100, 500),
    "max_depth":        (3, 8),
    "learning_rate":    (0.01, 0.2),
    "subsample":        (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "min_child_weight": (1, 10),
    "gamma":            (0.0, 1.0),
}


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Churn model training pipeline")
    parser.add_argument("--company",   required=True,                help="Company identifier (used to name model output directory)")
    parser.add_argument("--data",      required=True,                help="Path to labelled training CSV")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE, help="Fraction of data held out for evaluation (default: 0.2)")
    parser.add_argument("--n-trials",  type=int,   default=N_TRIALS,  help="Number of Optuna trials for hyperparameter search (default: 50)")
    return parser.parse_args()


# -----------------------------
# Data loading & cleaning
# -----------------------------

def load_and_clean(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df


# -----------------------------
# Column detection
# -----------------------------

def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def detect_columns(df: pd.DataFrame, targets: list[str]) -> dict[str, str | None]:
    norm_to_actual = {_normalise(col): col for col in df.columns}
    return {target: norm_to_actual.get(_normalise(target)) for target in targets}


# -----------------------------
# Feature engineering
# -----------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    col_map     = detect_columns(df, ["tenure", "MonthlyCharges", "TotalCharges"])
    tenure_col  = col_map["tenure"]
    monthly_col = col_map["MonthlyCharges"]
    total_col   = col_map["TotalCharges"]

    #--Generate New Customer Column, Tenure * Monthly col, Charges Ratio, Tenure Bins--

    for col in [tenure_col, monthly_col, total_col]:
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if total_col and tenure_col:
        new_customer_mask = df[tenure_col] == 0
        df.loc[new_customer_mask, total_col] = df.loc[new_customer_mask, total_col].fillna(0)

    if monthly_col and total_col:
        df["charges_ratio"] = df[monthly_col] / (df[total_col].replace(0, 1))

    if tenure_col and monthly_col:
        df["tenure_x_monthly"] = df[tenure_col] * df[monthly_col]

    if tenure_col:
        df["tenure_cohort"] = pd.cut(
            df[tenure_col],
            bins=[0, 12, 36, np.inf],
            labels=["new", "developing", "established"],
        ).astype(str)

    #--Generate High Risk Contract Column--

    contract_col = next(
        (c for c in df.columns if _normalise(c) == "contract"), None
    )
    if contract_col and monthly_col:
        median_charge = df[monthly_col].median()
        df["high_risk_contract"] = (
            (df[contract_col].str.lower() == "month-to-month") &
            (df[monthly_col] > median_charge)
        ).astype(int)

    service_cols = [
        c for c in df.columns
        if _normalise(c) in {
            "phoneservice", "multiplelines", "internetservice",
            "onlinesecurity", "onlinebackup", "deviceprotection",
            "techsupport", "streamingtv", "streamingmovies",
        }
    ]
    if service_cols:
        df["active_services"] = (
            df[service_cols]
            .apply(lambda col: col.str.lower().isin(["yes", "dsl", "fiber optic"]))
            .sum(axis=1)
        )

    return df


# -----------------------------
# Preprocessing
# -----------------------------

def split_target_independent_var(df: pd.DataFrame,
                        churn_col: str,
                        customer_col: str | None,) -> tuple[pd.DataFrame, pd.Series]:
    
    drop_cols = [c for c in [churn_col, customer_col] if c and c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[churn_col]
    
    y_list = y.tolist()
    
    for d in y_list:
        if d is not type(int):
            y.map({"No": 0, "Yes": 1})
            if y.isna().any():
                raise ValueError(
                    f"Churn column '{churn_col}' contains unexpected values: "
                    f"{df[churn_col].unique().tolist()}. Expected 'Yes'/'No'."
                )

    return X, y


# -----------------------------
# Schema
# -----------------------------

def build_schema(X: pd.DataFrame) -> dict:
    return {
        "features": list(X.columns),
        "ignore_columns": IGNORE_COLUMNS,
        "thresholds": {
            "low_risk":    LOW_RISK_THRESHOLD,
            "medium_risk": MEDIUM_RISK_THRESHOLD,
            "high_risk":   HIGH_RISK_THRESHOLD,
        },
    }

# -----------------------------
# Pipeline builder
# -----------------------------

def build_pipeline(X: pd.DataFrame, xgb_params: dict| None = None, calibrate: bool = True) -> Pipeline:
    numeric_cols     = X.select_dtypes(exclude="object").columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline,     numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )

    base_xgb = XGBClassifier(
        **(xgb_params or {}),
        random_state=SEED,
        eval_metric="logloss",
        use_label_encoder=False,
    )

    # Isotonic calibration ensures predicted probabilities reflect true churn likelihood 
    calibrated_xgb = CalibratedClassifierCV(
        base_xgb,
        method="isotonic",
        cv=3,
    ) if calibrate else base_xgb

    return Pipeline([
        ("preprocessing", preprocessor),
        ("model",         calibrated_xgb),
    ])


# -----------------------------
# Hyperparameter search (Optuna)
# -----------------------------

def _objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, cv: StratifiedKFold,) -> float:
    params = {
        "n_estimators":     trial.suggest_int(  "n_estimators",     *OPTUNA_PARAM_SPACE["n_estimators"]),
        "max_depth":        trial.suggest_int(  "max_depth",        *OPTUNA_PARAM_SPACE["max_depth"]),
        "learning_rate":    trial.suggest_float("learning_rate",    *OPTUNA_PARAM_SPACE["learning_rate"], log=True),
        "subsample":        trial.suggest_float("subsample",        *OPTUNA_PARAM_SPACE["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *OPTUNA_PARAM_SPACE["colsample_bytree"]),
        "min_child_weight": trial.suggest_int(  "min_child_weight", *OPTUNA_PARAM_SPACE["min_child_weight"]),
        "gamma":            trial.suggest_float("gamma",            *OPTUNA_PARAM_SPACE["gamma"]),
    }

    pipeline = build_pipeline(X_train, xgb_params=params, calibrate = False)

    oof_proba = cross_val_predict(
        pipeline, X_train, y_train,
        cv=cv, method="predict_proba", n_jobs=-1,
    )[:, 1]

    return float(average_precision_score(y_train, oof_proba))


def run_optuna_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int,
) -> dict:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )

    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, cv),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    print(f"Best Optuna CV Brier Score: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params

# -----------------------------
# Feature importance
# -----------------------------

def _get_grouped_importances(pipeline: Pipeline) -> dict[str, float]:
    preprocessor   = pipeline.named_steps["preprocessing"]
    base_estimator = pipeline.named_steps["model"].calibrated_classifiers_[0].estimator
    feature_names  = preprocessor.get_feature_names_out()
    importances    = base_estimator.feature_importances_

    grouped: dict[str, float] = {}
    for name, imp in zip(feature_names, importances):
        name = name.split("__")[-1]
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and not parts[1].isnumeric():
            name = parts[0]
        grouped[name] = grouped.get(name, 0.0) + imp

    return grouped


def print_feature_importance(pipeline: Pipeline, top_n: int = 15) -> None:
    try:
        grouped      = _get_grouped_importances(pipeline)
        sorted_feats = sorted(grouped.items(), key=lambda x: x[1], reverse=True)

        print(f"\n-- Top {top_n} Feature Importances ----------")
        for name, imp in sorted_feats[:top_n]:
            print(f"  {name:<45} {imp:.4f}")
        print("----------------------------------------\n")

    except Exception as e:
        print(f"[warn] Could not extract feature importances: {e}")


def select_important_features(pipeline: Pipeline, X: pd.DataFrame, importance_threshold: float) -> list[str]:
    grouped = _get_grouped_importances(pipeline)

    important = [name for name, imp in grouped.items() if imp >= importance_threshold]
    dropped   = [name for name, imp in grouped.items() if imp < importance_threshold]

    if dropped:
        print(f"[feature selection] Dropping {len(dropped)} low-importance features: {dropped}")
    print(f"[feature selection] Retaining {len(important)} features above threshold {importance_threshold}")

    return [col for col in X.columns if col in important]


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_model(pipeline, X_test, y_test) -> dict:
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    brier          = brier_score_loss(y_test, y_proba)
    brier_baseline = brier_score_loss(y_test, np.full_like(y_proba, y_test.mean()))

    return {
        "roc_auc":           roc_auc_score(y_test, y_proba),
        "pr_auc":            average_precision_score(y_test, y_proba),
        "log_loss":          log_loss(y_test, y_proba),
        "brier_score":       brier,
        "brier_skill_score": 1 - (brier / brier_baseline),
    }


def print_metrics(metrics: dict) -> None:
    print("\n-- Evaluation Metrics ------------------")
    for k, v in metrics.items():
        print(f"  {k:<18} {v:.4f}")
    print("----------------------------------------\n")


# -----------------------------
# Artifact persistence
# -----------------------------

def save_artifacts(pipeline: Pipeline, schema: dict, metrics: dict, company: str) -> Path:
    model_dir = Path(f"models/{company}")
    model_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_dir / "pipeline.pkl")

    with open(model_dir / "schema.json", "w") as f:
        json.dump(schema, f, indent=4)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(model_dir / "model_metrics.csv", index=False)

    return model_dir


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    if not (0 < args.test_size < 1):
        raise ValueError(f"--test-size must be between 0 and 1 exclusively, got {args.test_size}")

    print(f"Loading dataset from '{args.data}' ...")
    df = load_and_clean(args.data)

    col_map      = detect_columns(df, ["Churn", "CustomerID"])
    churn_col    = col_map["Churn"]
    customer_col = col_map["CustomerID"]

    if not churn_col:
        raise ValueError("Could not detect a 'Churn' column in the dataset.")

    print("Engineering features ...")
    df = engineer_features(df)

    X, y = split_target_independent_var(df, churn_col, customer_col)

    # Stratified train / test split — shared across both passes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=SEED, stratify=y
    )

    # ------------------------------------------------
    # Hyperparameter search (on full feature set)
    # ------------------------------------------------
    print(f"Running Optuna search ({args.n_trials} trials) ...")
    best_params = run_optuna_search(X_train, y_train, n_trials=args.n_trials)

    # ------------------------------------------------
    # Pass 1 — fit on all features to rank importance
    # ------------------------------------------------
    print("\n[Pass 1] Fitting on full feature set ...")
    first_pipeline = build_pipeline(X_train, xgb_params=best_params, calibrate = True)
    first_pipeline.fit(X_train, y_train)

    print_feature_importance(first_pipeline)

    important_cols  = select_important_features(first_pipeline, X_train, importance_threshold = 0.019)
    X_train_reduced = X_train[important_cols]
    X_test_reduced  = X_test[important_cols]

    # ------------------------------------------------
    # Pass 2 — refit on reduced feature set
    # ------------------------------------------------
    print("\n[Pass 2] Refitting on reduced feature set ...")
    final_pipeline = build_pipeline(X_train_reduced, xgb_params=best_params)
    final_pipeline.fit(X_train_reduced, y_train)

    schema = build_schema(X_train_reduced)

    # ------------------------------------------------
    # Evaluation
    # ------------------------------------------------
    print("Evaluating model ...")
    metrics = evaluate_model(final_pipeline, X_test_reduced, y_test)
    print_metrics(metrics)

    model_dir = save_artifacts(final_pipeline, schema, metrics, args.company)
    print(f"Artifacts saved to {model_dir}/")
    print("Training complete.")


if __name__ == "__main__":
    main()