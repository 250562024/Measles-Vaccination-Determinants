"""
descriptive_stats.py

Produces frequency and percentage tables for each section of the
questionnaire:
  Part One   - Socio-demographic & health service factors
  Part Two   - Measles vaccination status
  Part Three - Family-related factors

Outputs CSV tables to outputs/tables/ and a console summary.

Usage:
    python scripts/descriptive_stats.py
"""

import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/cleaned_survey_data.csv")
TABLES_DIR = Path("outputs/tables")

SECTIONS = {
    "Part One: Socio-Demographic Factors": [
        "age_group", "sex", "marital_status", "education_level",
        "occupation", "monthly_income",
    ],
    "Part One: Health Service Factors": [
        "distance_to_facility", "waiting_time", "vaccine_stockout_experience",
        "schedule_info_provided", "facility_delivery",
    ],
    "Part Two: Measles Vaccination Status": [
        "child_vaccinated_measles", "vaccination_confirmation_source",
    ],
    "Part Three: Family-Related Factors": [
        "knows_importance_of_vaccination", "received_health_education",
        "family_supports_immunization", "cultural_beliefs_influence",
        "number_of_children",
    ],
}


def freq_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    counts = df[col].value_counts(dropna=False)
    pct = (counts / len(df) * 100).round(1)
    out = pd.DataFrame({"Category": counts.index, "n": counts.values, "%": pct.values})
    out.insert(0, "Variable", col)
    return out


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run scripts/clean_data.py first.")

    df = pd.read_csv(DATA_PATH)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Descriptive statistics — n = {len(df)}\n" + "=" * 50)

    for section_name, cols in SECTIONS.items():
        print(f"\n{section_name}\n" + "-" * len(section_name))
        section_tables = []
        for col in cols:
            if col == "vaccination_confirmation_source":
                # Only meaningful among vaccinated children
                sub = df[df["child_vaccinated_measles"] == "Yes"]
                tbl = freq_table(sub, col)
                tbl["Variable"] = f"{col} (among vaccinated, n={len(sub)})"
            else:
                tbl = freq_table(df, col)
            section_tables.append(tbl)
            print(tbl.to_string(index=False))
            print()

        combined = pd.concat(section_tables, ignore_index=True)
        fname = section_name.split(":")[0].strip().lower().replace(" ", "_")
        out_path = TABLES_DIR / f"descriptive_{fname}_{section_name.split(': ')[1].lower().replace(' ', '_')}.csv"
        combined.to_csv(out_path, index=False)

    print(f"\nAll frequency tables saved to {TABLES_DIR}/")


if __name__ == "__main__":
    main()
