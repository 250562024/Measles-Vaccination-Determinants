"""
logistic_regression.py

Multivariable logistic regression predicting measles vaccination status
from the significant/relevant predictors identified in chi-square testing.
Reports adjusted odds ratios (aOR), 95% CIs, and p-values — the standard
output format for a thesis discussion chapter.

Uses scikit-learn for the model fit and a bootstrap for confidence intervals
(no statsmodels dependency required). If statsmodels is installed, run
logistic_regression_statsmodels.py instead for classic Wald CIs/p-values.

Usage:
    python scripts/logistic_regression.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path("data/cleaned_survey_data.csv")
TABLES_DIR = Path("outputs/tables")

OUTCOME = "child_vaccinated_measles"

# Predictors to include in the multivariable model (informed by chi-square
# screening — in practice, include variables with p < 0.25 in bivariate
# analysis, per standard epidemiological model-building practice)
PREDICTORS = [
    "education_level",
    "distance_to_facility",
    "vaccine_stockout_experience",
    "schedule_info_provided",
    "facility_delivery",
    "knows_importance_of_vaccination",
    "received_health_education",
    "family_supports_immunization",
    "cultural_beliefs_influence",
]

N_BOOTSTRAP = 1000
SEED = 42


def build_design_matrix(df: pd.DataFrame):
    X_raw = df[PREDICTORS]
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    X_encoded = encoder.fit_transform(X_raw)
    feature_names = encoder.get_feature_names_out(PREDICTORS)
    X = pd.DataFrame(X_encoded, columns=feature_names, index=df.index)
    y = (df[OUTCOME] == "Yes").astype(int)
    return X, y, encoder


def bootstrap_ci(X: pd.DataFrame, y: pd.Series, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(X)
    coefs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        X_b, y_b = X.iloc[idx], y.iloc[idx]
        if y_b.nunique() < 2:
            continue
        model = LogisticRegression(max_iter=1000)
        model.fit(X_b, y_b)
        coefs.append(model.coef_[0])

    return np.array(coefs)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run scripts/clean_data.py first.")

    df = pd.read_csv(DATA_PATH)
    X, y, encoder = build_design_matrix(df)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    coefs = model.coef_[0]
    feature_names = X.columns.tolist()

    print("Bootstrapping confidence intervals "
          f"({N_BOOTSTRAP} resamples)... this may take a few seconds.")
    boot_coefs = bootstrap_ci(X, y, N_BOOTSTRAP, SEED)

    ci_lower = np.percentile(boot_coefs, 2.5, axis=0)
    ci_upper = np.percentile(boot_coefs, 97.5, axis=0)

    # Approximate two-sided p-value from the bootstrap distribution
    # (proportion of resampled coefficients crossing zero)
    p_values = []
    for i in range(len(feature_names)):
        col = boot_coefs[:, i]
        p_above = (col <= 0).mean() if coefs[i] > 0 else (col >= 0).mean()
        p_values.append(min(2 * p_above, 1.0))

    results = pd.DataFrame({
        "Predictor": feature_names,
        "Coefficient (log-odds)": coefs.round(3),
        "Adjusted OR": np.exp(coefs).round(3),
        "OR 95% CI lower": np.exp(ci_lower).round(3),
        "OR 95% CI upper": np.exp(ci_upper).round(3),
        "p-value (bootstrap)": np.round(p_values, 4),
    })
    results["Significant (p<0.05)"] = np.where(results["p-value (bootstrap)"] < 0.05, "Yes", "No")
    results = results.sort_values("p-value (bootstrap)")

    print("\nMultivariable logistic regression — adjusted odds ratios for measles vaccination\n")
    print(results.to_string(index=False))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "logistic_regression_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    # Quick model fit diagnostic
    train_acc = model.score(X, y)
    print(f"\nTraining accuracy: {train_acc:.3f} "
          f"(NB: with imbalanced classes, accuracy alone is not a reliable fit metric — "
          f"see README for caveats)")


if __name__ == "__main__":
    main()
