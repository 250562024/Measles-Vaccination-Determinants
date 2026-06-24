# Determinants of Measles Vaccination Status Among Children Aged 12–23 Months

A Python-based statistical analysis pipeline examining socio-demographic, health-service, and family-related determinants of measles vaccination completion among children aged 12–23 months, based on caregiver survey data.

This repository was built as part of an MPH/public health thesis (sample size: 422 caregivers) and is shared publicly as a demonstration of the analysis methodology and a public health / M&E data analysis portfolio piece.

---

## ⚠️ Important note on the data

> **The dataset in this repository (`data/raw_survey_data.csv`) is synthetic — it is not the real thesis data.**
>
> Real participant data (422 caregivers, collected under institutional ethics approval) cannot be published publicly for participant confidentiality reasons, and is still being finalized for the thesis itself.
>
> The synthetic dataset is generated programmatically (`scripts/generate_synthetic_data.py`) to **exactly match the structure, variables, and response categories of the real questionnaire** (see `Appendix_V_Questionnaire.md`), with realistic associations built in so the statistical pipeline produces interpretable, demonstration-quality output.
>
> **The entire pipeline is designed so that dropping in the real cleaned dataset (same column names) and re-running the scripts in order reproduces the thesis results chapter directly** — no code changes required.

---

## What this project demonstrates

- Survey data cleaning & validation (`pandas`)
- Descriptive/frequency analysis by questionnaire section
- Bivariate inferential statistics (Pearson's chi-square tests of association)
- Multivariable logistic regression with adjusted odds ratios & bootstrapped 95% confidence intervals (`scikit-learn`, `numpy`)
- Publication-style data visualization, including a forest plot of odds ratios (`matplotlib`, `seaborn`)
- A documented, reproducible analysis notebook
- Clean, modular project structure suitable for handing real data to later

## Study variables

Drawn directly from the study questionnaire (Appendix V):

| Section | Variables |
|---|---|
| **Socio-demographic factors** | Caregiver age, sex, marital status, education, occupation, household income |
| **Health service factors** | Distance to nearest facility, waiting time, vaccine stock-out experience, schedule information provided, facility delivery |
| **Outcome** | Child (12–23 months) measles vaccination status, confirmation source |
| **Family-related factors** | Knowledge of vaccination importance, health education received, family support for immunization, cultural/traditional beliefs, number of children in household |

See [`Appendix_V_Questionnaire.md`](Appendix_V_Questionnaire.md) for the full original instrument.

## Project structure

```
measles-vaccination-determinants/
├── data/
│   ├── raw_survey_data.csv          # synthetic demo data (or real export, once available)
│   └── cleaned_survey_data.csv      # output of clean_data.py
├── scripts/
│   ├── generate_synthetic_data.py   # builds the synthetic demo dataset
│   ├── clean_data.py                # validation, deduplication, missing-data report
│   ├── descriptive_stats.py         # frequency tables by questionnaire section
│   ├── chi_square_tests.py          # bivariate association tests
│   ├── logistic_regression.py       # multivariable model + adjusted odds ratios
│   └── visualize.py                 # generates all figures
├── notebooks/
│   └── analysis_walkthrough.ipynb   # end-to-end narrative notebook
├── outputs/
│   ├── tables/                      # CSV outputs (cross-tabs, regression results, etc.)
│   └── figures/                     # PNG charts
├── Appendix_V_Questionnaire.md
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone <this-repo-url>
cd measles-vaccination-determinants
pip install -r requirements.txt

# 1. Generate the synthetic demo dataset (skip this step once real data is available —
#    just place the real export at data/raw_survey_data.csv with matching column names)
python scripts/generate_synthetic_data.py --n 422 --seed 42

# 2. Clean and validate
python scripts/clean_data.py

# 3. Run the full analysis
python scripts/descriptive_stats.py
python scripts/chi_square_tests.py
python scripts/logistic_regression.py
python scripts/visualize.py
```

Or open `notebooks/analysis_walkthrough.ipynb` for the full narrative walkthrough with embedded charts.

## Sample output

**Adjusted odds ratios for predictors of measles vaccination status:**

![Odds ratio forest plot](outputs/figures/03_odds_ratio_forest_plot.png)

## Methodology notes

- **Chi-square tests** use Pearson's test of independence; cells with expected counts < 5 are flagged per standard practice.
- **Logistic regression** uses `scikit-learn` with one-hot encoded categorical predictors (reference category dropped) and bootstrap resampling (1,000 iterations) for 95% confidence intervals and approximate p-values, avoiding a hard dependency on `statsmodels`. Researchers who prefer classic Wald-based standard errors/p-values can swap in `statsmodels.Logit` directly — the design matrix construction in `logistic_regression.py` is written to make this a drop-in change.
- Variables entered into the multivariable model were selected based on the bivariate (chi-square) screening step, following standard epidemiological model-building convention (variables with p < 0.25 in bivariate analysis are eligible for the multivariable model).

## License & use

This code and methodology are shared for portfolio/demonstration purposes. The real thesis data and findings are not included and remain under the author's institution's ethics approval and data-sharing policies. Feel free to reuse the analysis pipeline structure for your own (properly ethics-approved) survey data.

## Author

Built and maintained as part of an ongoing MPH thesis. Feedback and suggestions welcome via Issues.
