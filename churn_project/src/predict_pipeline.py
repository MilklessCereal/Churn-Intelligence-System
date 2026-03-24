import argparse
import json
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import shap
from sklearn.utils import resample

warnings.filterwarnings("ignore")

# -----------------------------
# Configuration
# -----------------------------

SEED = 42
DEFAULT_SAMPLE_SIZE     = 200
TOP_N_FEATURES          = 3

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2" 


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Churn prediction pipeline")
    parser.add_argument("--company",     required=True,                         help="Company identifier (used to locate model artifacts)")
    parser.add_argument("--data",        required=True,                         help="Path to input CSV")
    parser.add_argument("--demo",        action="store_true",                   help="Bootstrap-resample input data for demo purposes")
    parser.add_argument("--sample_size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Number of samples when --demo is set")
    parser.add_argument(
        "--llm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use LLM-generated retention actions via Ollama (default, -no-llm for rule based recommendations)",
    )
    return parser.parse_args()


# -----------------------------
# Data loading & cleaning
# -----------------------------

def load_and_clean(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()
    df = df.replace(r"^\s*$", np.nan, regex=True)
    return df


def resample_for_demo(df: pd.DataFrame, n_samples: int, timestamp: str) -> pd.DataFrame:
    synthetic = resample(df, replace=True, n_samples=n_samples, random_state=SEED)
    out_path  = f"data/synthetic_prediction_data_{timestamp}.csv"
    Path("data").mkdir(parents=True, exist_ok=True)
    synthetic.to_csv(out_path, index=False)
    print(f"[demo] Synthetic dataset saved -> {out_path}")
    return synthetic


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

    for col in [tenure_col, monthly_col, total_col]:
        if col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if total_col and tenure_col:
        new_customer_mask = df[tenure_col] == 0
        df.loc[new_customer_mask, total_col] = df.loc[new_customer_mask, total_col].fillna(0)

    if monthly_col and total_col:
        df["charges_ratio"] = df[monthly_col] / (df[total_col] + 1)

    if tenure_col and monthly_col:
        df["tenure_x_monthly"] = df[tenure_col] * df[monthly_col]

    if tenure_col:
        df["tenure_cohort"] = pd.cut(
            df[tenure_col],
            bins=[0, 12, 36, np.inf],
            labels=["new", "developing", "established"],
        ).astype(str)

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
# Schema validation
# -----------------------------

def validate_schema(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    ignore  = schema.get("ignore_columns", [])
    df      = df.drop(columns=[c for c in ignore if c in df.columns], errors="ignore")

    required = set(schema["features"])
    present  = set(df.columns)

    missing = required - present
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")

    extra = present - required
    if extra:
        print(f" Dropping unexpected columns not in schema: {extra}")

    return df[schema["features"]]


# -----------------------------
# Model artifacts
# -----------------------------

def load_artifacts(model_dir: Path) -> tuple:
    pipeline_path = model_dir / "pipeline.pkl"
    schema_path   = model_dir / "schema.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"No trained pipeline found at '{pipeline_path}'. Run training first.")
    if not schema_path.exists():
        raise FileNotFoundError(f"No schema found at '{schema_path}'. Run training first.")
    pipeline = joblib.load(pipeline_path)

    with open(schema_path) as f:
        schema = json.load(f)

    thresholds = schema.get("thresholds", {})
    low        = thresholds.get("low_risk")
    medium     = thresholds.get("medium_risk")
    high   = thresholds.get("high_risk")    

    return pipeline, schema, low, medium, high   

# -----------------------------
# Prediction
# -----------------------------

def predict_proba(pipeline, df: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(df)[:, 1]


def assign_risk_segment(prob: float, low: float, medium: float, high: float) -> str:
    if prob < low:
        return "Low Risk"
    elif prob < medium:
        return "Medium Risk"
    elif prob < high:
        return "High Risk"
    return "Critical Risk"              

# -----------------------------
# SHAP explanation
# -----------------------------

def _parse_original_feature_name(encoded_name: str) -> str:
    return encoded_name.split("__")[-1]


def _get_display_name(encoded_name: str, df_original: pd.DataFrame, row_idx: int) -> str:
    raw = _parse_original_feature_name(encoded_name)

    for col in df_original.columns:
        prefix = f"{col}_"
        if raw.startswith(prefix):
            category_value = raw[len(prefix):]
            return f"{col} : {category_value}"

    if raw in df_original.columns:
        value = df_original.iloc[row_idx][raw]
        if pd.notna(value):
            display_value = round(value, 2) if isinstance(value, float) else value
            return f"{raw} : {display_value}"

    return raw.replace("_", " ")


def get_top_shap_features(
    shap_matrix:   np.ndarray,
    feature_names: np.ndarray,
    df_original:   pd.DataFrame,
    top_n:         int = TOP_N_FEATURES,
) -> list[str]:
    results = []

    for row_idx, row in enumerate(shap_matrix):
        grouped: dict[int, float] = {}
        for idx, val in enumerate(row):
            grouped[idx] = grouped.get(idx, 0.0) + abs(float(val))

        top_indices = sorted(grouped, key=grouped.__getitem__, reverse=True)[:top_n]

        labels = [
            _get_display_name(feature_names[i], df_original, row_idx)
            for i in top_indices
        ]
        results.append(", ".join(labels))

    return results


def _unwrap_xgb_model(pipeline):
    calibrated = pipeline.named_steps["model"]
    try:
        return calibrated.calibrated_classifiers_[0].estimator
    except AttributeError:
        return calibrated


def explain_predictions(pipeline, df: pd.DataFrame) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessing"]
    xgb_model    = _unwrap_xgb_model(pipeline)

    X_processed = preprocessor.transform(df)
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()
    X_processed = X_processed.astype(float)

    feature_names = preprocessor.get_feature_names_out()

    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_processed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return get_top_shap_features(shap_values, feature_names, df)


# -----------------------------
# Retention actions — rule-based
# -----------------------------

GENERIC_RULES: list[tuple[str, str]] = [
    (r"contract.*month",           "Promote switching to a long-term contract"),
    (r"tenure",                    "Provide a loyalty reward or retention incentive"),
    (r"charge|price|rate",         "Review pricing and offer a tailored discount"),
    (r"security.*no",              "Offer an online security add-on"),
    (r"support.*no",               "Provide a dedicated support package"),
    (r"fiber",                     "Check service quality and offer upgrade incentives"),
    (r"payment.*electronic|check", "Encourage switching to automatic payment"),
    (r"paperless.*yes",            "Provide billing transparency incentives"),
    (r"line.*no|multi.*no",        "Offer a bundled multi-line promotion"),
    (r"dependent.*no|family.*no",  "Offer family or bundled service plans"),
]

RISK_LEVEL_ACTIONS: dict[str, str] = {
    "High Risk":   "Escalate to retention team immediately",
    "Medium Risk": "Send a targeted retention promotion",
    "Low Risk": "If required: Send a generic retention promotion",
}


def generate_recommended_actions(
    risk_factors: list[str],
    risk_level:   str | None = None,
    top_n:        int = 3,
) -> str:
    seen:           set[str]  = set()
    factor_actions: list[str] = []

    for factor in risk_factors:
        normalised = factor.lower().replace(" ", "").replace("_", "")
        for pattern, action in GENERIC_RULES:
            if re.search(pattern, normalised) and action not in seen:
                seen.add(action)
                factor_actions.append(action)

    factor_actions = factor_actions[:top_n]

    risk_action = RISK_LEVEL_ACTIONS.get(risk_level, "") if risk_level else ""
    if risk_action and risk_action not in seen:
        all_actions = (
            [risk_action] + factor_actions
            if risk_level == "High Risk"
            else factor_actions + [risk_action]
        )
    else:
        all_actions = factor_actions

    return "; ".join(all_actions) if all_actions else "Monitor customer satisfaction and engagement"


# -----------------------------
# Retention actions — LLM (Ollama)
# -----------------------------

def generate_recommended_actions_llm(
    risk_factors: list[str],
    risk_level:   str | None = None,
    top_n:        int = 3,
) -> str:
    factor_text = "\n".join(f"  - {f}" for f in risk_factors[:top_n])
    level_text  = f"Churn risk level: {risk_level}\n" if risk_level else ""

    prompt = f"""You are a customer retention specialist.

A customer is at risk of churning. Based on the information below, suggest specific retention actions.

{level_text}Top contributing churn risk factors (from most to least important):
{factor_text}

Rules: 
- Output must return exactly {top_n} actions, separated by semicolons 
- Each action must be short (under 10 words), specific, and directly address one of the risk factors
- No bullet points, numbering, preamble, or explanation — output actions only"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json    = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout = 60,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.RequestException as e:
        print(f"[warn] Ollama request failed mid-run: {e}. Falling back to rule-based.")
        return generate_recommended_actions(risk_factors, risk_level, top_n)


# -----------------------------
# Retention actions — batch
# -----------------------------

def _check_ollama_available() -> bool:
    try:
        response = requests.get(OLLAMA_URL.replace("/api/generate", ""), timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def generate_actions_batch(
    risk_factors_col: list[str],
    risk_levels_col:  list[str],
    top_n:            int = 3,
    max_workers:      int = 10,
    use_llm:          bool = True,
) -> list[str]:

    if use_llm and not _check_ollama_available():
        print(f'[warn] Ollama is not running — falling back to rule-based for all rows.'
            'To run LLM response, start Ollama with: ollama serve')
        use_llm = False

    action_fn = generate_recommended_actions_llm if use_llm else generate_recommended_actions

    def _process_row(args):
        idx, factors_str, level = args
        factor_list = [f.strip() for f in factors_str.split(",")]
        return idx, action_fn(factor_list, level, top_n)

    results  = [None] * len(risk_factors_col)
    job_args = [
        (i, factors, level)
        for i, (factors, level) in enumerate(zip(risk_factors_col, risk_levels_col))
    ]

    print(f"Generating retention actions for {len(job_args)} customers "
        f"({'LLM via Ollama' if use_llm else 'rule-based'}, {max_workers} threads) ...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_row, args): args[0] for args in job_args}
        for future in as_completed(futures):
            idx, action = future.result()
            results[idx] = action

    return results


# -----------------------------
# Output
# -----------------------------

def build_results(
    customer_ids:     pd.Series | None,
    y_proba:          np.ndarray,
    risk_factors:     list[str],
    low_threshold:    float,
    medium_threshold: float,
    high_threshold:   float,
    recommended:      list[str] | None = None,
    df_original:      pd.DataFrame | None = None,  
) -> pd.DataFrame:
    risk_levels = [
        assign_risk_segment(p, low_threshold, medium_threshold, high_threshold) for p in y_proba]
    
    churn_prob = y_proba * 100
    churn_prob_percentage = churn_prob.round(2)

    results = pd.DataFrame({
        "Churn Probability":   churn_prob_percentage,
        "Churn Risk Level":    risk_levels,
        "Key Risk Factors":    risk_factors,
        "Recommended Actions": recommended if recommended is not None else [""] * len(y_proba),
    })

    if df_original is not None:
        col_map = detect_columns(df_original, ["MonthlyCharges", "Contract", "tenure"])
        for label, col in col_map.items():
            if col and col in df_original.columns:
                results[label] = df_original[col].reset_index(drop=True)

    if customer_ids is not None:
        results.insert(0, "customerID", customer_ids.reset_index(drop=True))

    return results


def save_results(results: pd.DataFrame, company: str, timestamp: str) -> str:
    Path("results").mkdir(parents=True, exist_ok=True)
    path = f"results/{company}_predictions_{timestamp}.csv"
    results.to_csv(path, index=False)
    return path


# -----------------------------
# Main
# -----------------------------

def main():
    args      = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = Path(f"models/{args.company}")

    df = load_and_clean(args.data)

    col_map      = detect_columns(df, ["Churn", "CustomerID"])
    churn_col    = col_map["Churn"]
    customer_col = col_map["CustomerID"]

    prediction_df = df.drop(columns=[churn_col] if churn_col else [])

    if args.demo:
        if args.sample_size != DEFAULT_SAMPLE_SIZE and not args.demo:
            print("warning: run --demo before inputting --sample_size")
        print(f"[demo] Resampling {args.sample_size} rows from '{args.data}'")
        prediction_df = resample_for_demo(prediction_df, args.sample_size, timestamp)

    print(f"Running predictions for '{args.company}' on {len(prediction_df)} rows ...")

    customer_ids = (
        prediction_df[customer_col].copy()
        if customer_col and customer_col in prediction_df.columns
        else None
    )

    print("Engineering features ...")
    prediction_df = engineer_features(prediction_df)

    pipeline, schema, low_threshold, medium_threshold, high_threshold = load_artifacts(model_dir) 

    X = validate_schema(prediction_df, schema)

    print("Running churn predictions ...")
    y_proba = predict_proba(pipeline, X)

    print("Computing SHAP explanations ...")
    risk_factors = explain_predictions(pipeline, X)

    risk_levels = [
        assign_risk_segment(p, low_threshold, medium_threshold, high_threshold)   
        for p in y_proba
    ]

    recommended = generate_actions_batch(
        risk_factors_col = risk_factors,
        risk_levels_col  = risk_levels,
        use_llm          = args.llm,
    )

    results = build_results(
        customer_ids, y_proba, risk_factors,
        low_threshold, medium_threshold, high_threshold,   
        recommended  = recommended,
        df_original  = prediction_df,
    )

    print(results.head())

    results_path = save_results(results, args.company, timestamp)
    print(f"Predictions saved to {results_path}")


if __name__ == "__main__":
    main()