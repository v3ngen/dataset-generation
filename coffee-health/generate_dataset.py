"""
Coffee & Health Dataset Generator (v3)

Generates two files for the two-assignment coursework:

  * a *development* set  -- messy, labelled, what students work on
  * a *held-out test* set -- no missing values, drawn from a shifted recruitment
    mix, carrying anomalies and duplicates that are there to be analysed rather
    than repaired

See V3_COURSEWORK_DESIGN.md for intent, DATA_GENERATION_SPEC.md for the numbers,
and DATA_GENERATION_ALGORITHM.md for the step-by-step procedure.

Usage:
    python3 generate_dataset.py [--rows 10000] [--test-rows 3000] [--seed 42]
"""

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

COUNTRIES = ['Italy', 'France', 'UK', 'Norway']

# --------------------------------------------------------------------------------------
# Country-level distributions
# --------------------------------------------------------------------------------------

SMOKING_DIST = {
    'Norway': {'Never': 0.658, 'Former': 0.20, 'Light Smoker': 0.092, 'Heavy Smoker': 0.050},
    'Italy':  {'Never': 0.586, 'Former': 0.18, 'Light Smoker': 0.152, 'Heavy Smoker': 0.082},
    'France': {'Never': 0.504, 'Former': 0.15, 'Light Smoker': 0.225, 'Heavy Smoker': 0.121},
    'UK':     {'Never': 0.595, 'Former': 0.28, 'Light Smoker': 0.081, 'Heavy Smoker': 0.044},
}

ALCOHOL_DIST = {
    'Norway': {'Non-Drinker': 0.25, 'Light': 0.40, 'Moderate': 0.25, 'Heavy': 0.10},
    'Italy':  {'Non-Drinker': 0.15, 'Light': 0.40, 'Moderate': 0.35, 'Heavy': 0.10},
    'UK':     {'Non-Drinker': 0.12, 'Light': 0.28, 'Moderate': 0.40, 'Heavy': 0.20},
    'France': {'Non-Drinker': 0.10, 'Light': 0.30, 'Moderate': 0.40, 'Heavy': 0.20},
}

COFFEE_PARAMS = {
    'Norway': {'cups_mean': 3.2, 'cups_std': 1.0, 'mg_per_cup': 110},
    'Italy':  {'cups_mean': 3.0, 'cups_std': 1.1, 'mg_per_cup': 65},
    'France': {'cups_mean': 2.4, 'cups_std': 0.9, 'mg_per_cup': 80},
    'UK':     {'cups_mean': 1.8, 'cups_std': 0.9, 'mg_per_cup': 70},
}

STRESS_DIST = {
    'Norway': {'Low': 0.60, 'Medium': 0.30, 'High': 0.10},
    'Italy':  {'Low': 0.50, 'Medium': 0.35, 'High': 0.15},
    'UK':     {'Low': 0.45, 'Medium': 0.35, 'High': 0.20},
    'France': {'Low': 0.40, 'Medium': 0.35, 'High': 0.25},
}

ACTIVITY_DIST = {
    'Norway': {'Sedentary': 0.15, 'Lightly Active': 0.25, 'Moderately Active': 0.35, 'Very Active': 0.25},
    'Italy':  {'Sedentary': 0.25, 'Lightly Active': 0.30, 'Moderately Active': 0.30, 'Very Active': 0.15},
    'France': {'Sedentary': 0.25, 'Lightly Active': 0.32, 'Moderately Active': 0.28, 'Very Active': 0.15},
    'UK':     {'Sedentary': 0.30, 'Lightly Active': 0.30, 'Moderately Active': 0.25, 'Very Active': 0.15},
}

BMI_PARAMS = {
    'Norway': {'mean': 25.5, 'std': 4.2},
    'Italy':  {'mean': 24.0, 'std': 3.5},
    'France': {'mean': 25.7, 'std': 4.3},
    'UK':     {'mean': 26.8, 'std': 4.8},
}

# Median annual household income, EUR. Norway highest, Italy lowest, consistent
# with the SRH country ordering but not identical to it.
INCOME_MEDIAN = {'Norway': 62000, 'France': 44000, 'UK': 42000, 'Italy': 36000}
INCOME_SIGMA = 0.42

COUNTRY_OFFSET_SRH = {'Norway': -1.0, 'Italy': -1.75, 'France': 1.25, 'UK': -3.35}

# --------------------------------------------------------------------------------------
# Ordinal scales
# --------------------------------------------------------------------------------------

STRESS_ORDER = ['Low', 'Medium', 'High']
ACTIVITY_ORDER = ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active']
SLEEP_Q_ORDER = ['Poor', 'Fair', 'Good', 'Excellent']
HEALTH_ISSUES_ORDER = ['No Issues', 'Mild', 'Moderate', 'Severe']
SRH_ORDER = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
SMOKING_ORDER = ['Never', 'Former', 'Vaper', 'Light Smoker', 'Heavy Smoker']

# --------------------------------------------------------------------------------------
# Effect sizes. 'Vaper' sits between 'Former' and 'Light Smoker' throughout.
# --------------------------------------------------------------------------------------

SMOKING_BMI_ADJ = {'Never': 0.0, 'Former': 0.0, 'Vaper': 0.0, 'Light Smoker': 0.0, 'Heavy Smoker': -0.5}
SMOKING_HR_ADJ = {'Never': 0.0, 'Former': 1.0, 'Vaper': 2.0, 'Light Smoker': 3.0, 'Heavy Smoker': 6.0}
SMOKING_SLEEP_ADJ = {'Never': 0.0, 'Former': -1.0, 'Vaper': -2.5, 'Light Smoker': -4.0, 'Heavy Smoker': -8.0}
SMOKING_HEALTH_ADJ = {'Never': 0.0, 'Former': 2.0, 'Vaper': 3.5, 'Light Smoker': 5.0, 'Heavy Smoker': 12.0}
SMOKING_SRH_ADJ = {'Never': 0.0, 'Former': -3.0, 'Vaper': -5.0, 'Light Smoker': -8.0, 'Heavy Smoker': -18.0}

ALCOHOL_SLEEP_ADJ = {'Non-Drinker': 0.0, 'Light': -1.0, 'Moderate': -4.0, 'Heavy': -10.0}

ACTIVITY_BMI_ADJ = {'Sedentary': 1.0, 'Lightly Active': 0.0, 'Moderately Active': -0.7, 'Very Active': -1.5}
ACTIVITY_HR_ADJ = {'Sedentary': 3.0, 'Lightly Active': 0.0, 'Moderately Active': -3.0, 'Very Active': -6.0}
ACTIVITY_SRH_ADJ = {'Sedentary': -8.0, 'Lightly Active': -2.0, 'Moderately Active': 5.0, 'Very Active': 12.0}

STRESS_HR_ADJ = {'Low': 0.0, 'Medium': 2.0, 'High': 4.0}
STRESS_SLEEP_ADJ = {'Low': 0.0, 'Medium': -6.0, 'High': -15.0}
STRESS_SRH_ADJ = {'Low': 0.0, 'Medium': -6.0, 'High': -14.0}

SLEEP_Q_SRH_ADJ = {'Poor': -12.0, 'Fair': -4.0, 'Good': 5.0, 'Excellent': 12.0}
HEALTH_SRH_ADJ = {'No Issues': 0.0, 'Mild': -10.0, 'Moderate': -25.0, 'Severe': -45.0}
GENDER_SRH_ADJ = {'Male': 0.0, 'Female': -3.0, 'Other': 0.0}

# Latent frailty: an unobserved, standard-normal "how robust is this person really"
# factor that is never written to the CSV. It lowers SelfRatedHealth and raises the
# risk of a high-burden year, so SelfRatedHealth carries information about future
# outcomes OVER AND ABOVE every measured risk factor.
#
# This is real epidemiology -- self-rated health is famously predictive of mortality
# and utilisation after adjusting for measured risk factors, because people know
# things about themselves that a survey does not capture. It is also what makes
# SelfRatedHealth a genuine leakage trap rather than a redundant column: without a
# shared latent term it would add nothing a model cannot already get from the
# features it was built from.
SRH_FRAILTY = -18.0
HHB_FRAILTY = 1.35

# --------------------------------------------------------------------------------------
# HighHealthBurden: the binary ML target. Coefficients are log-odds contributions.
#
# The linear block is what a logistic regression can capture. The non-linear and
# interaction blocks below it are what make a tree ensemble or an MLP worth
# reaching for -- without them, LR would sit near Bayes-optimal and "introduce a
# more expressive model" would be a pointless exercise.
# --------------------------------------------------------------------------------------

HHB_INTERCEPT = -6.565           # calibrated so the positive rate lands near 20%

HHB_SMOKING = {'Never': 0.0, 'Former': 0.15, 'Vaper': 0.60, 'Light Smoker': 0.45, 'Heavy Smoker': 0.85}
HHB_ACTIVITY = {'Sedentary': 0.40, 'Lightly Active': 0.12, 'Moderately Active': -0.18, 'Very Active': -0.40}
HHB_STRESS = {'Low': 0.0, 'Medium': 0.30, 'High': 0.65}
HHB_SLEEP_Q = {'Poor': 0.55, 'Fair': 0.20, 'Good': -0.15, 'Excellent': -0.45}
HHB_HEALTH = {'No Issues': 0.0, 'Mild': 0.45, 'Moderate': 0.90, 'Severe': 1.45}
HHB_ALCOHOL = {'Non-Drinker': 0.05, 'Light': 0.0, 'Moderate': 0.12, 'Heavy': 0.55}
HHB_COUNTRY = {'Norway': -0.40, 'Italy': -0.18, 'France': 0.12, 'UK': 0.58}
HHB_GENDER = {'Male': 0.0, 'Female': 0.10, 'Other': 0.05}

HHB_AGE = 0.030                 # per year above 30
HHB_BMI = 0.055                 # per BMI unit outside the 19-25 band
HHB_HR = 0.006                  # per bpm above 70
HHB_INCOME = -4.5e-6            # per EUR above the 45k reference

# Non-monotonic terms. These apply to EVERY row rather than a rare subgroup, which
# is what makes them big enough to survive the noise and show up as a real gap
# between a linear model and a tree ensemble.
# Both are centred on the MIDDLE of the observed distribution, which is the whole
# point: a quadratic centred outside the data's range is very nearly linear across
# the support, and a linear model then captures it for free. Mean sleep is ~6.4h
# and mean intake ~2.8 cups, so both arms of each curve carry real mass.
HHB_SLEEP_U = 0.72              # per (hour - 6.4)^2   -- short AND long sleep are harmful
HHB_SLEEP_CENTRE = 6.4
HHB_COFFEE_U = 0.34             # per (cup - 2.8)^2    -- the J-curve: abstainers and heavy
HHB_COFFEE_CENTRE = 2.8         #   drinkers both fare worse than moderate drinkers

# Continuous interactions. Also always-on, and invisible to an additive model.
HHB_BMI_X_AGE = 0.0042          # per (BMI unit outside band) x (year above 30)
HHB_CAFFEINE_X_SLEEP = 0.00125  # per mg x (hour of sleep below 7)

HHB_NOISE_SD = 0.45             # unobserved heterogeneity

# Categorical interactions. Deliberately defined over broad groups rather than
# narrow ones: an interaction that only fires on 2% of rows contributes almost
# nothing to overall performance and cannot reward a more expressive model.
# Coffee's protective arm only holds for people who sleep well -- for poor sleepers
# it is roughly cancelled out. Thematically the centrepiece of a coffee dataset, and
# strongly non-additive.
HHB_COFFEE_X_SLEEPQ = 0.48      # per protective cup, for Poor/Fair sleepers

HHB_INTERACTIONS = {
    'smoker_x_overweight': 1.15,
    'stressed_x_poor_sleep': 1.00,
    'older_x_sedentary': 1.25,
    'older_x_very_active': -1.05,
}


# --------------------------------------------------------------------------------------
# Cohort configuration. Dev and test differ ONLY by these values -- the causal
# model (and therefore P(y | x_clean)) is identical between them.
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class CohortConfig:
    name: str
    country_mix: dict
    age_beta: tuple               # (a, b) for Beta(a, b) * 57 + 18
    vaper_share: float
    inject_missing: bool
    age_anomaly_rate: float
    bmi_anomaly_rate: float
    duplicate_rate: float
    duplicate_uk_weight: float    # relative sampling weight for UK rows when duplicating


DEV_CONFIG = CohortConfig(
    name='development',
    country_mix={'Italy': 0.25, 'France': 0.25, 'UK': 0.25, 'Norway': 0.25},
    age_beta=(2.0, 3.0),          # mean age ~40.8
    vaper_share=0.015,
    inject_missing=True,
    age_anomaly_rate=0.007,
    bmi_anomaly_rate=0.006,
    duplicate_rate=0.004,
    duplicate_uk_weight=1.0,      # uniform
)

TEST_CONFIG = CohortConfig(
    name='test',
    # A later recruitment wave that leaned heavily on UK sites.
    country_mix={'Italy': 0.18, 'France': 0.18, 'UK': 0.46, 'Norway': 0.18},
    age_beta=(2.55, 2.60),        # mean age ~46.2, about five years older
    vaper_share=0.090,            # vaping became far more common between waves
    inject_missing=False,         # a clean extract was supplied
    age_anomaly_rate=0.030,       # different data-entry process at the new site
    bmi_anomaly_rate=0.025,
    duplicate_rate=0.020,         # a double-submission incident
    duplicate_uk_weight=4.0,      # ...at the UK site specifically
)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def with_vaper_share(dist_map: dict, share: float) -> dict:
    """Return SMOKING_DIST with 'Vaper' set to `share`, taking the mass pro rata
    from 'Never' and 'Light Smoker' so each country still sums to 1."""
    out = {}
    for country, probs in dist_map.items():
        probs = dict(probs)
        delta = share - probs.get('Vaper', 0.0)
        donors = ['Never', 'Light Smoker']
        donor_total = sum(probs[d] for d in donors)
        for d in donors:
            probs[d] -= delta * (probs[d] / donor_total)
        probs['Vaper'] = share
        out[country] = {k: probs[k] for k in SMOKING_ORDER}
    return out


def sample_categorical_by_group(groups, dist_map, rng):
    result = np.empty(len(groups), dtype=object)
    groups = np.asarray(groups)
    for group_value, probs in dist_map.items():
        mask = groups == group_value
        n = int(mask.sum())
        if n == 0:
            continue
        result[mask] = rng.choice(list(probs.keys()), size=n, p=list(probs.values()))
    return result


def shift_ordinal(values, order, amount):
    """Shift categorical values along an ordered scale by `amount` steps (can be negative)."""
    idx = {v: i for i, v in enumerate(order)}
    codes = np.array([idx[v] for v in values])
    codes = np.clip(codes + amount, 0, len(order) - 1)
    return np.array(order)[codes]


def bucket(scores, thresholds, labels):
    """thresholds: ascending upper bounds for all but the last label."""
    idx = np.searchsorted(thresholds, scores, side='right')
    return np.array(labels)[idx]


def allocate_countries(n_rows: int, mix: dict, rng) -> np.ndarray:
    """Deterministic counts per country (so the mix is exact), then shuffled."""
    counts = {c: int(round(n_rows * p)) for c, p in mix.items()}
    drift = n_rows - sum(counts.values())
    if drift:
        largest = max(counts, key=counts.get)
        counts[largest] += drift
    country = np.repeat(list(counts.keys()), list(counts.values()))
    rng.shuffle(country)
    return country


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------

def generate(config: CohortConfig, n_rows: int, seed: int, thresholds: dict | None = None):
    """Generate one clean cohort.

    `thresholds` holds the quantile cut-points for the three ordinal composites.
    The test set MUST be passed the development set's thresholds: these composites
    are quantile-bucketed, so recomputing them on the test cohort would re-normalise
    every class to the same proportions and silently erase the distribution shift.

    Returns (DataFrame, thresholds_used).
    """
    rng = np.random.default_rng(seed)
    computing_thresholds = thresholds is None
    thresholds = dict(thresholds) if thresholds else {}

    country = allocate_countries(n_rows, config.country_mix, rng)
    n = len(country)
    df = pd.DataFrame({'ID': np.arange(1, n + 1), 'Country': country})

    # Level 0: demographics
    a, b = config.age_beta
    df['Age'] = np.round(rng.beta(a, b, size=n) * 57 + 18).astype(int)
    df['Gender'] = rng.choice(['Male', 'Female', 'Other'], size=n, p=[0.48, 0.48, 0.04])

    # Level 0.5: household income. Large in magnitude, modest in signal -- most of
    # its association with health runs *through* activity, smoking and BMI rather
    # than directly. That makes it honest for EDA (the SES gradient is real) while
    # ensuring an unscaled distance metric is dominated by a mostly-uninformative
    # axis, so feature scaling has something to fix.
    income_median = df['Country'].map(INCOME_MEDIAN).to_numpy()
    age_arr = df['Age'].to_numpy()
    age_income_factor = 1 + 0.010 * (np.minimum(age_arr, 52) - 30) - 0.012 * np.maximum(0, age_arr - 60)
    income = income_median * age_income_factor * rng.lognormal(0, INCOME_SIGMA, size=n)
    df['Household Income'] = np.clip(np.round(income, -2), 12000, 250000)

    # Standardised income, used below to nudge lifestyle variables.
    log_income = np.log(df['Household Income'].to_numpy())
    income_z = (log_income - log_income.mean()) / log_income.std()

    # Level 1: country-driven lifestyle, nudged by income
    smoking = sample_categorical_by_group(
        df['Country'], with_vaper_share(SMOKING_DIST, config.vaper_share), rng
    )
    # Higher income -> more likely to have quit / never started.
    quit_prob = np.clip(0.10 + 0.09 * income_z, 0, 0.40)
    smoking = np.where(rng.random(n) < quit_prob, shift_ordinal(smoking, SMOKING_ORDER, -1), smoking)
    df['Smoking Status'] = smoking

    df['Alcohol Level'] = sample_categorical_by_group(df['Country'], ALCOHOL_DIST, rng)

    cups_mean = df['Country'].map(lambda c: COFFEE_PARAMS[c]['cups_mean']).to_numpy()
    cups_std = df['Country'].map(lambda c: COFFEE_PARAMS[c]['cups_std']).to_numpy()
    mg_per_cup = df['Country'].map(lambda c: COFFEE_PARAMS[c]['mg_per_cup']).to_numpy()
    df['Daily Coffees'] = np.clip(rng.normal(cups_mean, cups_std), 0, 9).round(1)

    # ~10% of people lean decaf/half-caf (still order "coffees", but most cups are
    # low-caffeine) -- this is what keeps Daily Coffees <-> Caffeine Intake a strong
    # but imperfect correlation, with a real explanation rather than bare noise.
    decaf_leaning = rng.random(n) < 0.10
    decaf_multiplier = np.where(decaf_leaning, rng.uniform(0.05, 0.30, size=n), 1.0)
    effective_mg_per_cup = mg_per_cup * decaf_multiplier

    df['Caffeine Intake'] = np.clip(
        df['Daily Coffees'] * effective_mg_per_cup * rng.uniform(0.85, 1.15, size=n), 0, 750
    ).round(1)

    # Level 2: behavioural
    stress = sample_categorical_by_group(df['Country'], STRESS_DIST, rng)
    peak_age_mask = (df['Age'] >= 25) & (df['Age'] <= 45)
    escalate = peak_age_mask & (rng.random(n) < 0.15)
    stress = np.where(escalate, shift_ordinal(stress, STRESS_ORDER, 1), stress)
    df['Stress Level'] = stress

    activity = sample_categorical_by_group(df['Country'], ACTIVITY_DIST, rng)
    high_stress_mask = (df['Stress Level'] == 'High') & (rng.random(n) < 0.3)
    older_mask = (df['Age'] > 55) & (rng.random(n) < 0.3)
    activity = np.where(high_stress_mask, shift_ordinal(activity, ACTIVITY_ORDER, -1), activity)
    activity = np.where(older_mask, shift_ordinal(activity, ACTIVITY_ORDER, -1), activity)
    # Higher income -> more likely to be one step more active.
    active_prob = np.clip(0.12 + 0.10 * income_z, 0, 0.40)
    activity = np.where(rng.random(n) < active_prob, shift_ordinal(activity, ACTIVITY_ORDER, 1), activity)
    df['Physical Activity Level'] = activity

    # Level 3: physiology
    bmi_mean = df['Country'].map(lambda c: BMI_PARAMS[c]['mean']).to_numpy()
    bmi_std = df['Country'].map(lambda c: BMI_PARAMS[c]['std']).to_numpy()
    activity_bmi = df['Physical Activity Level'].map(ACTIVITY_BMI_ADJ).to_numpy()
    smoking_bmi = df['Smoking Status'].map(SMOKING_BMI_ADJ).to_numpy()
    age_bmi = 0.03 * np.minimum(age_arr, 60)
    income_bmi = -1.2e-5 * (df['Household Income'].to_numpy() - 45000)
    df['BMI'] = np.clip(
        bmi_mean + activity_bmi + smoking_bmi + age_bmi + income_bmi + rng.normal(0, bmi_std),
        16, 45
    ).round(1)

    activity_hr = df['Physical Activity Level'].map(ACTIVITY_HR_ADJ).to_numpy()
    smoking_hr = df['Smoking Status'].map(SMOKING_HR_ADJ).to_numpy()
    stress_hr = df['Stress Level'].map(STRESS_HR_ADJ).to_numpy()
    age_hr = 0.05 * age_arr
    df['Avg Resting Heart Rate'] = np.clip(
        72 + activity_hr + smoking_hr + stress_hr + age_hr + rng.normal(0, 7, size=n),
        45, 110
    ).round(0)

    # Level 4: sleep
    stress_sleep_hours = df['Stress Level'].map({'Low': 0.0, 'Medium': -0.4, 'High': -0.9}).to_numpy()
    caffeine_sleep_hours = -0.0015 * df['Caffeine Intake'].to_numpy()
    df['Avg Sleep Hours Per Night'] = np.clip(
        7.0 + stress_sleep_hours + caffeine_sleep_hours + rng.normal(0, 1.0, size=n),
        3, 10.5
    ).round(1)

    stress_sleep_q = df['Stress Level'].map(STRESS_SLEEP_ADJ).to_numpy()
    smoking_sleep_q = df['Smoking Status'].map(SMOKING_SLEEP_ADJ).to_numpy()
    alcohol_sleep_q = df['Alcohol Level'].map(ALCOHOL_SLEEP_ADJ).to_numpy()
    sleep_q_score = (
        40 + 6 * (df['Avg Sleep Hours Per Night'].to_numpy() - 7)
        + stress_sleep_q + smoking_sleep_q + alcohol_sleep_q
        - 0.01 * df['Caffeine Intake'].to_numpy()
        + rng.normal(0, 8, size=n)
    )
    # Quantile-based thresholds (rather than hand-picked constants) so the marginal
    # distribution reliably hits the target shape: Poor 10%, Fair 25%, Good 45%,
    # Excellent 20%.
    if computing_thresholds:
        thresholds['sleep_quality'] = np.quantile(sleep_q_score, [0.10, 0.35, 0.80])
    df['Sleep Quality'] = bucket(sleep_q_score, thresholds['sleep_quality'], SLEEP_Q_ORDER)

    # Level 5: chronic health
    smoking_health = df['Smoking Status'].map(SMOKING_HEALTH_ADJ).to_numpy()
    age_health = 0.5 * np.maximum(0, age_arr - 30)
    bmi_health = 1.2 * np.maximum(0, np.abs(df['BMI'].to_numpy() - 22) - 3)
    health_score = 15 + age_health + bmi_health + smoking_health + rng.normal(0, 10, size=n)
    # Target: No Issues 55%, Mild 30%, Moderate 11%, Severe 4%.
    if computing_thresholds:
        thresholds['health_issues'] = np.quantile(health_score, [0.55, 0.85, 0.96])
    df['Health Issues'] = bucket(health_score, thresholds['health_issues'], HEALTH_ISSUES_ORDER)

    # Latent frailty -- never written to the CSV. Shared by both targets, which is
    # what makes SelfRatedHealth genuinely informative about HighHealthBurden beyond
    # the measured features.
    frailty = rng.normal(0, 1, size=n)

    # Level 6: SelfRatedHealth (the 5-class EDA target, and CW2's leakage trap)
    # NOT clipped. An earlier version clipped this to [0, 100]; once the frailty term
    # was added, 8.6% of scores hit the floor, which exceeded the 8th-percentile
    # threshold -- so the first cut-point WAS zero, searchsorted(side='right') pushed
    # every clipped row up a bucket, and the 'Poor' class vanished entirely from the
    # data. The score is internal and the thresholds are quantile-based, so clipping
    # bought nothing and silently destroyed a class.
    srh_score = (
        60
        + df['Country'].map(COUNTRY_OFFSET_SRH).to_numpy()
        + df['Gender'].map(GENDER_SRH_ADJ).to_numpy()
        - 0.15 * np.maximum(0, age_arr - 30)
        + df['Physical Activity Level'].map(ACTIVITY_SRH_ADJ).to_numpy()
        - 1.2 * np.maximum(0, np.abs(df['BMI'].to_numpy() - 22) - 3)
        + df['Sleep Quality'].map(SLEEP_Q_SRH_ADJ).to_numpy()
        + df['Smoking Status'].map(SMOKING_SRH_ADJ).to_numpy()
        + df['Stress Level'].map(STRESS_SRH_ADJ).to_numpy()
        + df['Health Issues'].map(HEALTH_SRH_ADJ).to_numpy()
        + SRH_FRAILTY * frailty
        + rng.normal(0, 8, size=n)
    )
    # Poor 8%, Fair 17%, Good 40%, Very Good 25%, Excellent 10%.
    if computing_thresholds:
        thresholds['srh'] = np.quantile(srh_score, [0.08, 0.25, 0.65, 0.90])
    df['SelfRatedHealth'] = bucket(srh_score, thresholds['srh'], SRH_ORDER)

    # Level 7: HighHealthBurden -- the binary ML target
    df['HighHealthBurden'] = draw_high_health_burden(df, rng, frailty)

    return df, thresholds


def high_health_burden_logit(df: pd.DataFrame, frailty: np.ndarray | None = None) -> np.ndarray:
    """The noise-free log-odds of a high-burden health year.

    Split into three blocks so the spec, the validation script and the reference
    notebook can all talk about them separately.
    """
    age = df['Age'].to_numpy()
    bmi = df['BMI'].to_numpy()
    cups = df['Daily Coffees'].to_numpy()
    hours = df['Avg Sleep Hours Per Night'].to_numpy()
    caffeine = df['Caffeine Intake'].to_numpy()
    income = df['Household Income'].to_numpy()
    if frailty is None:
        frailty = np.zeros(len(df))

    linear = (
        HHB_INTERCEPT
        + df['Smoking Status'].map(HHB_SMOKING).to_numpy()
        + df['Physical Activity Level'].map(HHB_ACTIVITY).to_numpy()
        + df['Stress Level'].map(HHB_STRESS).to_numpy()
        + df['Sleep Quality'].map(HHB_SLEEP_Q).to_numpy()
        + df['Health Issues'].map(HHB_HEALTH).to_numpy()
        + df['Alcohol Level'].map(HHB_ALCOHOL).to_numpy()
        + df['Country'].map(HHB_COUNTRY).to_numpy()
        + df['Gender'].map(HHB_GENDER).to_numpy()
        + HHB_AGE * np.maximum(0, age - 30)
        + HHB_BMI * np.maximum(0, np.abs(bmi - 22) - 3)
        + HHB_HR * (df['Avg Resting Heart Rate'].to_numpy() - 70)
        + HHB_INCOME * (income - 45000)
        + HHB_FRAILTY * frailty
    )

    # Non-monotonic terms: a linear model on the raw features cannot represent these.
    bmi_dev = np.maximum(0, np.abs(bmi - 22) - 3)
    non_linear = (
        HHB_SLEEP_U * (hours - HHB_SLEEP_CENTRE) ** 2
        + HHB_COFFEE_U * (cups - HHB_COFFEE_CENTRE) ** 2
    )

    # Interactions: super-additive combinations of risk factors. The two continuous
    # ones apply to every row; the categorical ones are defined over broad groups so
    # they affect a meaningful share of the data.
    older = age > 55
    smoker = df['Smoking Status'].isin(['Vaper', 'Light Smoker', 'Heavy Smoker']).to_numpy()
    stressed = df['Stress Level'].isin(['Medium', 'High']).to_numpy()
    sleeps_badly = df['Sleep Quality'].isin(['Poor', 'Fair']).to_numpy()
    activity = df['Physical Activity Level'].to_numpy()

    interactions = (
        HHB_BMI_X_AGE * bmi_dev * np.maximum(0, age - 30)
        + HHB_CAFFEINE_X_SLEEP * caffeine * np.maximum(0, 7 - hours)
        + HHB_INTERACTIONS['smoker_x_overweight'] * (smoker & (bmi >= 27))
        + HHB_INTERACTIONS['stressed_x_poor_sleep'] * (stressed & sleeps_badly)
        + HHB_INTERACTIONS['older_x_sedentary'] * (older & (activity == 'Sedentary'))
        + HHB_INTERACTIONS['older_x_very_active'] * (older & (activity == 'Very Active'))
        + HHB_COFFEE_X_SLEEPQ * np.minimum(cups, 3) * sleeps_badly
    )

    return linear + non_linear + interactions


def draw_high_health_burden(df: pd.DataFrame, rng: np.random.Generator,
                            frailty: np.ndarray) -> np.ndarray:
    """Bernoulli draw, not a threshold -- so there is irreducible noise and a
    realistic performance ceiling rather than a boundary a flexible model could
    learn perfectly."""
    n = len(df)
    eta = high_health_burden_logit(df, frailty) + rng.normal(0, HHB_NOISE_SD, size=n)
    p = 1 / (1 + np.exp(-eta))
    return (rng.random(n) < p).astype(int)


# --------------------------------------------------------------------------------------
# Data quality issues, split so each can be applied independently
# --------------------------------------------------------------------------------------

IMMUNE_COLUMNS = {'ID', 'Country', 'SelfRatedHealth', 'HighHealthBurden'}


def age_missingness_multiplier(age: np.ndarray) -> np.ndarray:
    """Older people are less likely to own/use a wearable or app that auto-logs
    sleep, resting heart rate, or stress -- so those device/self-report fields
    should go missing more often as age increases."""
    return np.select([age < 30, age < 45, age < 60], [0.5, 0.9, 1.4], default=2.1)


def inject_missing_values(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Development set only. The test set is supplied as a clean extract, so that
    student performance does not hinge on how they handled missing values."""
    df = df.copy()

    # Device/self-report fields: missingness rate scales with age. Age itself is
    # still fully populated at this point (anomalies run later), so this is a clean
    # signal for students to find.
    age_mult = age_missingness_multiplier(df['Age'].to_numpy())
    for col, base_rate in [
        ('Avg Sleep Hours Per Night', 0.07),
        ('Avg Resting Heart Rate', 0.06),
        ('Stress Level', 0.05),
    ]:
        rate = np.clip(base_rate * age_mult, 0, 0.35)
        df.loc[rng.random(len(df)) < rate, col] = np.nan

    # Health Issues: a survey/clinical field rather than device-sourced, so a flat rate.
    df.loc[rng.random(len(df)) < 0.10, 'Health Issues'] = np.nan

    # Row-level incomplete records (5-10 rows, 4+ missing fields each)
    mutable_cols = [c for c in df.columns if c not in IMMUNE_COLUMNS]
    incomplete_row_idx = rng.choice(df.index, size=rng.integers(5, 11), replace=False)
    for row_idx in incomplete_row_idx:
        cols_to_null = rng.choice(mutable_cols, size=rng.integers(4, 7), replace=False)
        df.loc[row_idx, cols_to_null] = np.nan

    return df


def inject_anomalies(df: pd.DataFrame, rng: np.random.Generator,
                     age_rate: float, bmi_rate: float) -> pd.DataFrame:
    """Data-entry anomalies. Applied to BOTH cohorts, at a higher rate in the test
    set (a different data-entry process at the newer recruitment site).

    Crucially these run AFTER the label is drawn, so an anomalous row carries a
    label generated from its clean values but presents corrupted features -- which
    is exactly why those rows are near-guaranteed misclassifications in the test
    set, where students are not permitted to repair them.
    """
    df = df.copy()

    # Age typo (~0.7% dev / ~1.5% test): a digit gets doubled during entry
    # (e.g. 34 -> 344). For an 18-75 age range the doubled value never exceeds 755,
    # so stripping the trailing digit always recovers the exact original age --
    # a student can reason their way back rather than hitting a sentinel value.
    anomaly_mask = rng.random(len(df)) < age_rate
    ages = df.loc[anomaly_mask, 'Age']
    df.loc[anomaly_mask, 'Age'] = ages.apply(
        lambda a: min(999, int(f'{int(a)}{str(int(a))[-1]}')) if pd.notna(a) else a
    )

    # BMI typo (~0.6% dev / ~1.2% test): a missed decimal point (24.5 -> 245).
    # Dividing by 10 recovers the original value exactly.
    bmi_mask = rng.random(len(df)) < bmi_rate
    df.loc[bmi_mask, 'BMI'] = (df.loc[bmi_mask, 'BMI'] * 10).round(1)

    return df


def inject_duplicates(df: pd.DataFrame, rng: np.random.Generator,
                      rate: float, uk_weight: float) -> pd.DataFrame:
    """Exact duplicate rows appended to the end.

    In the test set these are drawn disproportionately from UK respondents (a
    double-submission at the UK recruitment site), so they skew the country mix and
    the class balance further -- and any duplicated row the model gets wrong is
    counted twice in the test metric.
    """
    n_dupes = int(round(len(df) * rate))
    if n_dupes <= 0:
        return df

    weights = np.where(df['Country'].to_numpy() == 'UK', uk_weight, 1.0)
    weights = weights / weights.sum()
    dupe_idx = rng.choice(df.index, size=n_dupes, replace=False, p=weights)
    return pd.concat([df, df.loc[dupe_idx]], ignore_index=True)


def apply_quality_issues(df: pd.DataFrame, config: CohortConfig,
                         rng: np.random.Generator) -> pd.DataFrame:
    if config.inject_missing:
        df = inject_missing_values(df, rng)
    df = inject_anomalies(df, rng, config.age_anomaly_rate, config.bmi_anomaly_rate)
    df = inject_duplicates(df, rng, config.duplicate_rate, config.duplicate_uk_weight)
    return df


# --------------------------------------------------------------------------------------

def build(rows: int, test_rows: int, seed: int):
    """Generate both cohorts. The test set reuses the development set's quantile
    thresholds -- see the docstring on generate()."""
    dev, thresholds = generate(DEV_CONFIG, rows, seed)
    dev = apply_quality_issues(dev, DEV_CONFIG, np.random.default_rng(seed + 1))

    test, _ = generate(TEST_CONFIG, test_rows, seed + 2, thresholds=thresholds)
    test = apply_quality_issues(test, TEST_CONFIG, np.random.default_rng(seed + 3))

    return dev, test


def main():
    parser = argparse.ArgumentParser(description='Generate the coffee & health v3 dataset')
    parser.add_argument('--rows', type=int, default=10000)
    parser.add_argument('--test-rows', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dev-output', type=str, default='coffee_health_v3_dev.csv')
    parser.add_argument('--test-output', type=str, default='coffee_health_v3_test.csv')
    args = parser.parse_args()

    dev, test = build(args.rows, args.test_rows, args.seed)

    dev.to_csv(args.dev_output, index=False)
    test.to_csv(args.test_output, index=False)

    print(f'Development set: {len(dev):>6} rows -> {args.dev_output}')
    print(f'  positive rate: {dev["HighHealthBurden"].mean():.1%}   '
          f'missing cells: {int(dev.isna().sum().sum())}')
    print(f'Test set:        {len(test):>6} rows -> {args.test_output}')
    print(f'  positive rate: {test["HighHealthBurden"].mean():.1%}   '
          f'missing cells: {int(test.isna().sum().sum())}')


if __name__ == '__main__':
    main()
