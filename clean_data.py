"""
clean_data.py

Loads the raw survey export, validates expected columns/categories,
handles missing values, and writes a cleaned analysis-ready CSV.

Usage:
    python scripts/clean_data.py
"""

import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw_survey_data.csv")
CLEAN_PATH = Path("data/cleaned_survey_data.csv")

EXPECTED_COLUMNS = [
    "age_group", "sex", "marital_status", "education_level", "occupation",
    "monthly_income", "distance_to_facility", "waiting_time",
    "vaccine_stockout_experience", "schedule_info_provided", "facility_delivery",
    "child_vaccinated_measles", "vaccination_confirmation_source",
    "knows_importance_of_vaccination", "received_health_education",
    "family_supports_immunization", "cultural_beliefs_influence",
    "number_of_children",
]

CATEGORICAL_LEVELS = {
    "age_group": ["15-24", "25-34", "35-44", "45+"],
    "sex": ["Male", "Female"],
    "marital_status": ["Married", "Single", "Widow", "Widower"],
    "education_level": ["No formal education", "Primary", "Secondary", "College", "University"],
    "monthly_income": ["Low", "Middle", "High"],
    "distance_to_facility": ["Less than 1 km", "1-5 km", "More than 5 km"],
    "waiting_time": ["Less than 30 minutes", "30-60 minutes", "More than 1 hour"],
    "vaccine_stockout_experience": ["Yes", "No"],
    "schedule_info_provided": ["Yes", "No"],
    "facility_delivery": ["Yes", "No"],
    "child_vaccinated_measles": ["Yes", "No"],
    "knows_importance_of_vaccination": ["Yes", "No"],
    "received_health_education": ["Yes", "No"],
    "family_supports_immunization": ["Yes", "No"],
    "cultural_beliefs_influence": ["Yes", "No"],
    "number_of_children": ["1-2", "3-4", "5+"],
}


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/generate_synthetic_data.py first, "
            f"or place your real cleaned export at this path with matching column names."
        )
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> None:
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Strip whitespace from string columns
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()

    # Drop fully duplicate rows (e.g. double data-entry)
    n_before = len(df)
    df = df.drop_duplicates()
    n_dupes = n_before - len(df)
    if n_dupes:
        print(f"Removed {n_dupes} duplicate row(s).")

    # Flag unexpected category values rather than silently dropping them
    for col, levels in CATEGORICAL_LEVELS.items():
        if col not in df.columns:
            continue
        bad_mask = ~df[col].isin(levels) & df[col].notna() & (df[col] != "")
        n_bad = bad_mask.sum()
        if n_bad:
            print(f"WARNING: {n_bad} unexpected value(s) in '{col}': "
                  f"{df.loc[bad_mask, col].unique().tolist()}")

    # Missing-data report (no imputation — flag only, since
    # over-imputing survey categorical data can distort thesis findings)
    missing_counts = df.replace("", pd.NA).isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if len(missing_counts):
        print("\nMissing values per column:")
        print(missing_counts)
    else:
        print("\nNo missing values detected.")

    return df


def main():
    df = load_raw(RAW_PATH)
    validate_columns(df)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {RAW_PATH}")

    df_clean = clean(df)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\nCleaned dataset written to {CLEAN_PATH} ({len(df_clean)} rows)")


if __name__ == "__main__":
    main()
