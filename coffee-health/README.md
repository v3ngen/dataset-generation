# Coffee & Health Dataset

**Version 3 is current.** v3 repurposes the dataset to carry a two-assignment unit — CW1 on
exploratory data analysis, CW2 on applied machine learning — by adding a binary ML target, two
stakeholder cost matrices, and a **held-out test set** drawn from a shifted recruitment wave.

- **Why it is built this way:** [V3_COURSEWORK_DESIGN.md](V3_COURSEWORK_DESIGN.md)
- **The stakeholder scenario and cost matrices (brief-ready):** [STAKEHOLDERS_AND_COSTS.md](STAKEHOLDERS_AND_COSTS.md)
- **What the numbers are:** [DATA_GENERATION_SPEC.md](DATA_GENERATION_SPEC.md) → Version 3 section
- **How they are produced:** [DATA_GENERATION_ALGORITHM.md](DATA_GENERATION_ALGORITHM.md)

## Files

| File | What it is |
|---|---|
| `coffee_health_v3_dev.csv` | **Development set** — 10,040 rows, messy, labelled. Students do everything here. |
| `coffee_health_v3_test.csv` | **Held-out test set** — 3,060 rows, no missing values, from a shifted cohort. The common benchmark. |
| `cw2_starter_notebook.ipynb` | The deliberately weak pipeline students are handed. |
| `cw2_reference_notebook.ipynb` | Instructor-facing worked solution, validation, and the "beat this" benchmark. |
| `costs.py` | Both cost matrices, cost helpers, and the analytic optimal threshold. Students get this. |
| `generate_dataset.py` | Produces both CSVs, seeded. |
| `validate_dataset.py` | 67 checks: ranges, distributions, quality issues, and every intended shift. |
| `validate_ml_ladder.py` | 23 checks that each rung of the CW2 improvement ladder actually pays off. |
| `dataset_validation.ipynb` | The v2 data-validation notebook (EDA and quality issues). |
| `data/` | Original source data and the archived v2 dataset. |

## Regenerating and checking

```bash
python3 generate_dataset.py --seed 42     # writes both CSVs
python3 validate_dataset.py               # 67 checks -> exit code
python3 validate_ml_ladder.py             # 23 checks -> exit code
```

`validate_ml_ladder.py` is the important one if the generator is ever retuned: it is what
guarantees the assignment still works. Two gotchas are documented in the algorithm doc — quadratic
terms must stay centred on the observed distribution, and the test cohort must reuse the
development set's quantile thresholds.

## The two targets

| Target | Type | Used for |
|---|---|---|
| `SelfRatedHealth` | 5-class ordinal | **CW1's** EDA target. In **CW2** it is the leakage trap (it must be excluded from the feature set) and then the error-analysis segmentation variable. |
| `HighHealthNeeds` | binary, ~20% positive | **CW2's** ML target: ≥14 days health-related absence or ≥6 primary-care contacts in the following 12 months. |

## The two stakeholders

Both want to identify the same people for a preventive programme, and pay very differently for
mistakes — so they can rank the same models differently. Full scenario in
[STAKEHOLDERS_AND_COSTS.md](STAKEHOLDERS_AND_COSTS.md).

| | FP | FN | Optimal threshold |
|---|---|---|---|
| National public health agency | €180 | €1,450 | **0.077** — wants recall |
| Employer occupational health | €520 | €780 | **0.377** — wants precision |

## Test-set rules

The test set is a clean extract: **no missing values**, so results cannot hinge on how a student
handled missing data. The only operations permitted on it are **encoding and scaling** — never
cleaning, imputation, or resampling.

It *does* carry anomalies (3.0% age, 2.5% BMI) and duplicates (2.0%, 83% of them UK). Those are
there to be **found and explained, not repaired**. Together with the covariate shift they account
for test performance sitting below the cross-validated estimate.

## Why test performance is lower — three separable layers

1. **Optimism / overfitting** — training-set accuracy 0.925 vs 5-fold CV 0.851.
2. **Corrupted rows** — a higher anomaly rate produces rows that are near-unpredictable by
   construction, plus duplicates that double-count some errors.
3. **Covariate shift** — UK 25%→47%, mean age 40.7→46.2, vaping 2.7%→9.7%, positive rate
   19.8%→25.6%. The rise in positive rate looks like label shift but is **compositional**: the
   model stays calibrated on the test set (predicted 0.259 vs observed 0.256), so the relationship
   between features and outcome did not change — only who was recruited.

---

## Dataset Overview

- **Size**: 10,040 rows × 16 columns (10,000 people + ~40 intentional duplicate rows)
- **Structure**: One row per person
- **Countries**: Italy, France, UK, Norway — chosen for meaningfully different coffee/lifestyle/health profiles while staying recognizable to UK students (~2,500 rows each)
- **Format**: CSV file with intentional data quality issues
- **Target**: A single classification task, `SelfRatedHealth`

## Features (16 columns)

### Demographics
1. **ID** (int): unique identifier
2. **Age** (int/float): 18-75, with intentional typo-style anomalies
3. **Gender** (str): {Male, Female, Other}
4. **Country** (str): {Italy, France, UK, Norway}

### Coffee & Caffeine
5. **Daily Coffees** (float): cups per day, country-varying mean
6. **Caffeine Intake** (float, mg): correlated with Daily Coffees but with country-varying mg/cup (Italian espresso vs. Nordic filter coffee) — and not a perfect correlation: ~10% of people are decaf/half-caf leaning, so they still register normal "coffees" counts with much lower actual caffeine

### Lifestyle
7. **Smoking Status** (str): {Never, Former, Light Smoker, Heavy Smoker} — captures dose-response, not just yes/no
8. **Alcohol Level** (str): {Non-Drinker, Light, Moderate, Heavy}
9. **Physical Activity Level** (str): {Sedentary, Lightly Active, Moderately Active, Very Active}
10. **Avg Sleep Hours Per Night** (float) — a typical/average value (self-reported or wearable-aggregated), not a single night's reading
    - *Note: has intentional missing values, increasing with Age (~3.5% at 18-29 up to ~15% at 60+) — older people are less likely to own/use a sleep-tracking wearable or app*
11. **Sleep Quality** (str): {Poor, Fair, Good, Excellent} — a genuine multi-factor composite (sleep hours, stress, smoking, alcohol, caffeine), not derived from sleep hours alone
12. **Stress Level** (str): {Low, Medium, High} — generated independently, not a relabeling of another column. Measurement source is deliberately ambiguous (self-evaluated or from an activity tracker's stress score)
    - *Note: has intentional missing values, increasing with Age (~3% at 18-29 up to ~10% at 60+), same rationale as above*

### Physiology
13. **BMI** (float)
14. **Avg Resting Heart Rate** (float) — a typical/average resting value (wearable or clinical measurement), not a single point-in-time reading
    - *Note: has intentional missing values, increasing with Age (~2.7% at 18-29 up to ~13% at 60+)*
15. **Health Issues** (str): {No Issues, Mild, Moderate, Severe}
    - *Note: has intentional missing values (~10%, flat — not device-related, unlike the three fields above)*

### Target Variable (for ML Classification)
16. **SelfRatedHealth** (str): {Poor, Fair, Good, Very Good, Excellent} — a real, widely-used public-health survey construct (Eurostat, WHO), built here as a weighted multi-factor composite of Health Issues, BMI, Sleep Quality, Physical Activity, Smoking, Stress, Age, Country, and Gender. No single feature determines it.

## Country Profiles

Chosen and calibrated against real statistics (see `DATA_GENERATION_SPEC.md`'s Country Profiles table for sources): Norway and Italy both have high coffee consumption *and* good self-rated health; the UK has the *lowest* coffee consumption and the *highest* obesity/*worst* self-rated health. This is a genuinely non-obvious pattern — coffee habits alone don't predict health outcomes, lifestyle does.

## Built-in Patterns for Discovery

1. **Country effects**: distinct coffee, smoking, alcohol, activity, and health profiles per country, matching real-world statistics
2. **Dose-response smoking**: Never > Former > Light Smoker > Heavy Smoker in self-rated health, not a flat smoker/non-smoker split
3. **Genuine multi-factor Sleep Quality**: responds to stress, smoking, and alcohol, not just sleep hours
4. **Independent Stress Level**: has its own signal, unlike some naively-generated datasets where it's a disguised copy of another column
5. **Gender-health paradox**: a small, real, documented effect where women self-report slightly lower health despite typically greater longevity
6. **No deterministic shortcuts**: the target isn't a threshold on any single included feature — there's no shortcut column that trivially predicts it

## Intentional Data Quality Issues

This dataset includes realistic data quality problems for teaching data cleaning:

### 1. Duplicate Rows (~0.4%)
- **Detection**: `df.duplicated()`
- **Cleaning**: `df.drop_duplicates()`

### 2. Age Anomalies (~0.7%)
- **Type**: typo-style doubled-digit errors (e.g. 34 recorded as 344)
- **Detection**: `df[(df['Age'] > 100) | (df['Age'] < 13)]`
- **Cleaning**: strip the last digit (`344` → `34`) — always exactly recovers the original value, since the doubled value never exceeds 755 for our 18-75 age range

### 3. BMI Anomalies (~0.6%)
- **Type**: typo-style missed/misplaced decimal point (e.g. 24.5 recorded as 245)
- **Detection**: `df[df['BMI'] > 60]`
- **Cleaning**: divide by 10 — exactly recovers the original value. Note: even at this low a rate, these outliers meaningfully distort a raw (uncleaned) BMI↔Activity correlation coefficient — a real demonstration of outlier sensitivity, not a generation flaw

### 4. Missing Values
**Feature-level, demographic-differential** (increases with Age — older people less likely to own/use a wearable or app that logs these):
- `Avg Sleep Hours Per Night`: ~3.5% (18-29) up to ~15% (60+)
- `Avg Resting Heart Rate`: ~2.7% (18-29) up to ~13% (60+)
- `Stress Level`: ~3% (18-29) up to ~10% (60+)

**Feature-level, flat:**
- `Health Issues`: ~10% missing (not device-related, so no demographic pattern)

**Row-level:** a handful of rows (5-10) with 4+ missing fields each, simulating corrupted records

**Immune fields** (never missing): ID, Country, SelfRatedHealth

## Machine Learning Task

**Target**: Predict `SelfRatedHealth` {Poor, Fair, Good, Very Good, Excellent}

**Notes for students**:
- This is an ordinal target — a prediction one class off (e.g. Good vs. Very Good) is a smaller error than one two classes off. Plain accuracy doesn't capture that; the demo notebook also reports an "off by N classes" breakdown.
- Ordinal categorical features (Smoking Status, Alcohol Level, Physical Activity Level, Stress Level, Sleep Quality, Health Issues) should be ordinal-encoded (0, 1, 2, ...) rather than one-hot encoded, since their order is meaningful.
- Nominal features (Country, Gender) should be one-hot encoded.
- No feature is a disguised copy of the target — unlike some datasets designed to teach data leakage, this one deliberately has no such shortcut, so there's nothing to exclude for leakage reasons.

**Baseline performance**: ~65% accuracy with Random Forest (5-class), see `dataset_validation.ipynb` for the full breakdown.

## Usage Example

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('generated_coffee_health_dataset.csv')

# Check data quality issues
print(f"Duplicates: {df.duplicated().sum()}")
print(f"Age anomalies: {len(df[df['Age'] > 100])}")
print(f"BMI anomalies: {len(df[df['BMI'] > 60])}")
print(f"Missing values:\n{df.isnull().sum()}")

# Clean
df_clean = df.drop_duplicates()
df_clean = df_clean.dropna(subset=['Avg Sleep Hours Per Night', 'Health Issues'])

# See dataset_validation.ipynb for the full cleaning, EDA, and ML workflow
```

## Dataset Generation

Generated using a synthetic data generation algorithm that ensures realistic, research-grounded correlations between features, country-level variation, and controlled data quality issues. For technical details, see:
- `DATA_GENERATION_SPEC.md` — feature spec, target definition, real-world grounding, decision log
- `DATA_GENERATION_ALGORITHM.md` — exact generation formulas and calibration
- `generate_dataset.py` — implementation

## Validation

`validate_dataset.py` checks value ranges, categorical/target distributions against spec targets, key correlations, and data quality issue counts.

Run: `python3 validate_dataset.py generated_coffee_health_dataset.csv`

## Validation Notebook

`dataset_validation.ipynb` is an **instructor-facing review notebook**, not student teaching material — it exists to manually confirm the generated dataset behaves as designed. This is the notebook deliverable for this dataset; no student-facing version is planned. Covers: data quality assessment and cleaning, EDA (distributions, correlations, country comparisons, demographic-sliced missingness, relationship analysis), and Random Forest classification with feature importance and an ordinal-aware error breakdown.

## Files Included

- `generated_coffee_health_dataset.csv` — the main dataset
- `generate_dataset.py` — dataset generation script
- `validate_dataset.py` — validation and quality checks
- `analyze_original_dataset.py` — quantitative limitations analysis of the original source dataset
- `dataset_validation.ipynb` — instructor validation/review notebook
- `DATA_GENERATION_SPEC.md` — feature specification and design rationale
- `DATA_GENERATION_ALGORITHM.md` — exact generation algorithm
- `requirements.txt` — Python dependencies
- `data/` — original source dataset and its README, plus `coffee_health_v2_README.md`: the **student-facing** assignment brief for this redesigned dataset, mirroring the original `coffee_health_README.docx`'s style/structure

## License

Synthetic dataset created for educational purposes. Free to use in teaching, tutorials, workshops, or any educational context.
