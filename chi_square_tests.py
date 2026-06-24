"""
chi_square_tests.py

Tests association between each categorical predictor and the outcome
(child_vaccinated_measles) using Pearson's chi-square test of independence.
This mirrors the standard "Table 4.x: Association between X and measles
vaccination status" tables expected in a thesis results chapter.

Outputs a summary CSV with chi-square statistic, df, p-value, and a plain
interpretation, plus individual cross-tab CSVs.

Usage:
    python scripts/chi_square_tests.py
"""

import pandas as pd
from scipy.stats import chi2_contingency
from pathlib import Path

DATA_PATH = Path("data/cleaned_survey_data.csv")
TABLES_DIR = Path("outputs/tables")

OUTCOME = "child_vaccinated_measles"

PREDICTORS = [
    "age_group", "sex", "marital_status", "education_level", "occupation",
    "monthly_income", "distance_to_facility", "waiting_time",
    "vaccine_stockout_experience", "schedule_info_provided", "facility_delivery",
    "knows_importance_of_vaccination", "received_health_education",
    "family_supports_immunization", "cultural_beliefs_influence",
    "number_of_children",
]

ALPHA = 0.05


def run_chi_square(df: pd.DataFrame, predictor: str, outcome: str = OUTCOME):
    ct = pd.crosstab(df[predictor], df[outcome])
    chi2, p, dof, expected = chi2_contingency(ct)

    # Flag if >20% of expected cell counts are below 5 (chi-square assumption)
    low_expected_pct = (expected < 5).mean() * 100
    note = "Caution: low expected cell counts" if low_expected_pct > 20 else ""

    return ct, chi2, dof, p, note


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run scripts/clean_data.py first.")

    df = pd.read_csv(DATA_PATH)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"Chi-square tests of association with '{OUTCOME}' — n = {len(df)}\n" + "=" * 60)

    for predictor in PREDICTORS:
        ct, chi2, dof, p, note = run_chi_square(df, predictor)

        sig = "Significant" if p < ALPHA else "Not significant"
        print(f"\n{predictor}")
        print(ct)
        print(f"  chi2 = {chi2:.3f}, df = {dof}, p = {p:.4f}  -> {sig} (alpha=0.05)"
              + (f"  [{note}]" if note else ""))

        results.append({
            "Predictor": predictor,
            "Chi-square": round(chi2, 3),
            "df": dof,
            "p-value": round(p, 4),
            "Significant (p<0.05)": "Yes" if p < ALPHA else "No",
            "Note": note,
        })

        ct.to_csv(TABLES_DIR / f"crosstab_{predictor}.csv")

    summary = pd.DataFrame(results).sort_values("p-value")
    summary_path = TABLES_DIR / "chi_square_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("SUMMARY (sorted by p-value)")
    print(summary.to_string(index=False))
    print(f"\nFull summary saved to {summary_path}")
    print(f"Individual cross-tabs saved to {TABLES_DIR}/crosstab_<variable>.csv")


if __name__ == "__main__":
    main()
