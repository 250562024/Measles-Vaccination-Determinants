"""
visualize.py

Generates publication-style figures:
  1. Vaccination status overview (pie/bar)
  2. Socio-demographic distribution (age, education)
  3. Vaccination rate by key predictors (grouped bar charts)
  4. Forest plot of adjusted odds ratios from the logistic regression

Outputs PNGs to outputs/figures/.

Usage:
    python scripts/visualize.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("data/cleaned_survey_data.csv")
OR_PATH = Path("outputs/tables/logistic_regression_results.csv")
FIG_DIR = Path("outputs/figures")

OUTCOME = "child_vaccinated_measles"

sns.set_theme(style="whitegrid", font_scale=1.0)
PALETTE = {"Yes": "#2A9D8F", "No": "#E76F51"}


def vaccination_overview(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = df[OUTCOME].value_counts()
    colors = [PALETTE.get(k, "#999999") for k in counts.index]
    ax.pie(
        counts.values, labels=counts.index, autopct=lambda p: f"{p:.1f}%\n(n={int(round(p/100*len(df)))})",
        colors=colors, startangle=90, textprops={"fontsize": 11}
    )
    ax.set_title(f"Measles Vaccination Status of Children (12–23 months)\nn = {len(df)}", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_vaccination_status_overview.png", dpi=150)
    plt.close(fig)


def vaccination_rate_by(df: pd.DataFrame, col: str, title: str, fname: str, order=None):
    rate = (
        df.groupby(col)[OUTCOME]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .reset_index(name="vaccination_rate")
    )
    if order:
        rate[col] = pd.Categorical(rate[col], categories=order, ordered=True)
        rate = rate.sort_values(col)

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(rate[col].astype(str), rate["vaccination_rate"], color="#2A9D8F")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylabel("Vaccination rate (%)")
    ax.set_ylim(0, 105)
    ax.set_title(title, fontsize=13)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)


def demographic_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    age_order = ["15-24", "25-34", "35-44", "45+"]
    age_counts = df["age_group"].value_counts().reindex(age_order)
    axes[0].bar(age_counts.index, age_counts.values, color="#264653")
    axes[0].set_title("Caregiver Age Distribution")
    axes[0].set_ylabel("Number of respondents")

    edu_order = ["No formal education", "Primary", "Secondary", "College", "University"]
    edu_counts = df["education_level"].value_counts().reindex(edu_order)
    axes[1].barh(edu_counts.index, edu_counts.values, color="#E9C46A")
    axes[1].set_title("Caregiver Education Level")
    axes[1].set_xlabel("Number of respondents")

    fig.suptitle(f"Socio-Demographic Profile (n = {len(df)})", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_demographic_overview.png", dpi=150)
    plt.close(fig)


def odds_ratio_forest_plot():
    if not OR_PATH.exists():
        print(f"Skipping forest plot — {OR_PATH} not found. Run scripts/logistic_regression.py first.")
        return

    df = pd.read_csv(OR_PATH).sort_values("Adjusted OR")

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = range(len(df))

    colors = ["#2A9D8F" if sig == "Yes" else "#999999" for sig in df["Significant (p<0.05)"]]

    ax.errorbar(
        df["Adjusted OR"], y_pos,
        xerr=[df["Adjusted OR"] - df["OR 95% CI lower"], df["OR 95% CI upper"] - df["Adjusted OR"]],
        fmt="o", color="black", ecolor="gray", elinewidth=1.5, capsize=3, markersize=0,
    )
    ax.scatter(df["Adjusted OR"], y_pos, color=colors, s=80, zorder=3)
    ax.axvline(1.0, color="red", linestyle="--", linewidth=1, label="OR = 1 (no effect)")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["Predictor"].str.replace("_", " "))
    ax.set_xlabel("Adjusted Odds Ratio (95% CI)")
    ax.set_title("Predictors of Measles Vaccination Status\n(Multivariable Logistic Regression)", fontsize=12)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_odds_ratio_forest_plot.png", dpi=150)
    plt.close(fig)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run scripts/clean_data.py first.")

    df = pd.read_csv(DATA_PATH)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    vaccination_overview(df)
    demographic_overview(df)
    vaccination_rate_by(
        df, "received_health_education",
        "Vaccination Rate by Health Education Exposure",
        "04_rate_by_health_education.png",
    )
    vaccination_rate_by(
        df, "distance_to_facility",
        "Vaccination Rate by Distance to Health Facility",
        "05_rate_by_distance.png",
        order=["Less than 1 km", "1-5 km", "More than 5 km"],
    )
    vaccination_rate_by(
        df, "vaccine_stockout_experience",
        "Vaccination Rate by Vaccine Stock-Out Experience",
        "06_rate_by_stockout.png",
    )
    odds_ratio_forest_plot()

    print(f"Figures saved to {FIG_DIR}/")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
