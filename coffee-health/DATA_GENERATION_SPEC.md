# Data Generation Specification — Coffee & Health

## Overview

This document captures the requirements for a redesigned coffee-consumption-and-health dataset, replacing the local `data/coffee_health_original.csv` (itself a modified-for-assignment version of Kaggle's ["Global Coffee Health Dataset"](https://www.kaggle.com/datasets/uom190346a/global-coffee-health-dataset)).

**Target Size**: ~10,000 rows, **one row per person** (no multi-entity structure like online-gaming's player-game rows — there's no natural repeating unit here).

**Status**: v3 (coursework redesign) is current. The sections below describe the feature-level
design, which v3 inherits and extends; the **Version 3** section immediately following records
what v3 adds and changes. Treat unchecked items and open questions as normal, not as things to
rush to close — see the repo root README.md's note on the iterative spec process.

> v3 repurposes the dataset to carry a two-assignment unit (EDA, then applied ML). The *why* is
> in [V3_COURSEWORK_DESIGN.md](V3_COURSEWORK_DESIGN.md); the numbers are here.

---

## Version 3 — coursework redesign

v3 adds a **binary ML target**, two new pieces of feature structure, and a **second cohort**
(a held-out test set drawn from a shifted recruitment mix). The v1/v2 feature design below is
unchanged except where noted.

### New features

| Feature | Design |
|---|---|
| **Household Income** | Annual, EUR. Log-normal, `median(country) x age_factor x LogNormal(0, 0.42)`, clipped to 12,000–250,000. Country medians: Norway 62k, France 44k, UK 42k, Italy 36k. Rises to a peak around age 52, then eases. Deliberately **large in magnitude, weak in direct signal** — most of its association with health runs *through* activity, smoking and BMI rather than directly, which is honest (the socioeconomic health gradient is real) while ensuring that an unscaled distance metric is dominated by a mostly-uninformative axis. Feature scaling then has something real to fix. |
| **Smoking Status: `Vaper`** | New level, ordered between `Former` and `Light Smoker`. ~1.5% configured share in development, 9% in test. Note the *observed* development share is ~2.7%, because the income-driven "quit" shift moves some light smokers down a step onto it — switching to vaping as a quit route, which is realistic. |

Income also nudges three existing features: a higher-income person is more likely to be one step
more active, one step further along the quit scale, and has a modestly lower BMI.

### New target — `HighHealthNeeds`

Binary. **Did this person have a year of high health needs in the twelve months after the survey** —
≥14 days of health-related absence from work or normal activity, **or** ≥6 primary-care contacts?

Drawn as `y ~ Bernoulli(sigmoid(eta))` — a **stochastic draw, not a threshold**, so there is
irreducible noise and a realistic ceiling rather than a boundary a flexible model could learn
perfectly. Calibrated to a **~20% positive rate**, which sets the majority-class baseline at 80%
accuracy: the supplied starter pipeline then scores ~80% while finding almost nobody, and in
fact lands slightly below that trivial baseline.

`eta` has three blocks. The split matters, and is the single most important design decision in v3.

**Linear block** (what logistic regression can capture): intercept −6.565; smoking
{Never 0, Former .15, Vaper .60, Light .45, Heavy .85}; activity {Sedentary .40, Lightly .12,
Moderately −.18, Very Active −.40}; stress {Low 0, Medium .30, High .65}; sleep quality
{Poor .55, Fair .20, Good −.15, Excellent −.45}; health issues {No Issues 0, Mild .45,
Moderate .90, Severe 1.45}; alcohol {Non-Drinker .05, Light 0, Moderate .12, Heavy .55};
country {Norway −.40, Italy −.18, France .12, UK .58}; gender {Male 0, Female .10, Other .05};
0.030 per year above 30; 0.055 per BMI unit outside the 19–25 band; 0.006 per bpm above 70;
−4.5e-6 per EUR of income above 45,000.

**Non-monotonic block**: `0.72 x (sleep hours − 6.4)^2` and `0.34 x (cups − 2.8)^2`.

> **The centring is the whole point.** Both quadratics are centred on the *middle of the observed
> distribution*. An earlier draft centred the sleep curve at 7.25 hours — but mean sleep is 6.36
> hours, so 79% of the data sat on one arm and the term correlated −0.76 with raw hours. A linear
> model captured it for free and the gap to a tree ensemble was **negative**. Likewise
> `min(cups, 3)` correlated +0.91 with cups and was, in effect, a linear term wearing a disguise.
> Re-centring both on the data was what made the non-linearity real.

**Interaction block**: `0.0042 x bmi_deviation x years_over_30`; `0.00125 x caffeine x
max(0, 7 − sleep hours)`; `0.48 x min(cups, 3) x sleeps_badly` (coffee's protective arm only
holds for people who sleep well); `+1.15` smoker x BMI≥27; `+1.00` stressed x poor/fair sleep;
`+1.25` over-55 x sedentary; `−1.05` over-55 x very active. Categorical interactions are defined
over **broad** groups deliberately: one firing on 2% of rows contributes almost nothing to overall
performance and cannot reward a more expressive model.

**Noise**: latent frailty `x 1.35`, plus idiosyncratic `N(0, 0.45)`.

### Latent frailty — why `SelfRatedHealth` is a genuine leakage trap

A standard-normal **frailty** term, never written to the CSV, lowers `SelfRatedHealth` (weight
−18.0) and raises `eta` (weight +1.35).

Without it, `SelfRatedHealth` is a *deterministic-ish function of columns already in the feature
set*, so a model gains nothing from it — the first v3 draft measured a leakage benefit of
essentially zero. The shared latent is what makes it informative **beyond** every measured risk
factor, which is both the mechanism of the trap and real epidemiology: self-rated health predicts
later mortality and utilisation after adjusting for measured risk factors, because respondents
incorporate information no questionnaire captures.

Measured effect of including it: **+0.025 F1, +6.6pp accuracy** — seductive, detectable, and
not so total that the exercise becomes trivial. Within each self-rated-health level the
high-needs rate runs 65.7% (Poor) down to 1.6% (Excellent): strongly informative, nowhere near
deterministic.

### Two cohorts

| | Development | Held-out test |
|---|---|---|
| Rows | 10,000 + 40 duplicates | 3,000 + 60 duplicates |
| Country mix | 25% each | Italy/France/Norway 18%, **UK 46%** |
| Age | `Beta(2.0, 3.0) x 57 + 18` (mean 40.7) | `Beta(2.55, 2.60) x 57 + 18` (mean **46.2**) |
| Vaper share | 1.5% configured (~2.7% observed) | **9%** |
| Missing values | yes — see below | **none** |
| Age / BMI anomalies | 0.7% / 0.6% | **3.0% / 2.5%** |
| Duplicates | 0.4%, uniform | 2.0%, **UK-weighted (x4)** |
| Positive rate | 19.8% | **25.6%** |

**The causal model is identical between them.** Every coefficient above is shared, so
`P(y | x_clean)` is unchanged and all shift is covariate shift. There is deliberately **no concept
drift** — it would leave students with nothing to do but complain, where covariate shift is
diagnosable and partly fixable.

The higher positive rate therefore emerges *automatically* from the shifted covariates rather than
being imposed. Reweighting the development set's own within-group rates to the test mix recovers
+0.034 of the +0.058 rise, and the model stays calibrated on the test set (predicted 0.259 vs
observed 0.256) — which is the decisive evidence that the relationship itself did not move.

### Test-set rules

The test set is supplied as a **clean extract**: no missing values, so student performance cannot
hinge on how they handled missing data — that is a training-data concern. The only operations
permitted on it are **encoding and scaling**.

It does carry anomalies and duplicates, at higher rates than development, and those are **for
analysis, not repair**. Because anomalies are injected *after* the label is drawn, an anomalous
row carries a label determined by its true values while presenting corrupted ones, so it is close
to unpredictable by construction — a measurable, separable component of the train/test gap. The
duplicates are UK-weighted (83% of them), so they are not merely redundant rows but a further
distortion of the mix.

### Quality-issue injection, split by cohort

Injection is three independent steps — missingness, anomalies, duplicates — each with its own
per-cohort rate, so the test set can take the last two and not the first. Missingness runs
**before** the anomalies so the age-differential missingness signal is computed on clean ages.

### Targets the dataset must hit

Enforced by `validate_ml_ladder.py` (23 checks) and `validate_dataset.py` (67 checks):

| Property | Target | Measured |
|---|---|---|
| Majority-class accuracy | ~0.80 | 0.804 |
| Starter pipeline recall | ≤0.45 | 0.156 |
| Scaling: KNN F1 gain | ≥+0.15 | 0.107 → 0.403 |
| Class weighting: recall | ≥0.60 | 0.326 → 0.755 |
| Tuning: RF F1 gain | ≥+0.05 | 0.451 → 0.601 |
| Hold-out instability | a pair of models must swap rank | swaps on 4/10 splits |
| Tree ensemble over LR (like for like) | +0.03 to +0.15 F1 | +0.070 |
| Leakage gain | +0.02 to +0.20 F1 | +0.025 |
| Stakeholder disagreement | ≥1 pair of models split them | 2 pairs |
| Overfitting: train − CV | ≥+0.03 | +0.073 |
| Drift: CV − test | +0.015 to +0.08 | +0.025 |


---

## Step (a): Limitations of the Current Dataset

Full quantitative analysis: `analyze_original_dataset.py` (run with `python3 analyze_original_dataset.py`). Summary of confirmed issues:

- **Sleep Quality** is a pure function of Sleep Hours (near-zero overlap between buckets), with ~0 relationship to caffeine, smoking, or alcohol.
- **Physical Activity** (numeric 0-15) correlates ~0 with Sleep Hours, BMI, Heart Rate, or Occupation — effectively random noise, contradicting its own documentation.
- **Stress Level** is **100% deterministic from Sleep Quality** (High stress → Poor sleep always, Medium → Fair always, Low → Good/Excellent). Zero independent signal — this is why it "felt like a feature" when used as a target last year.
- **Smoker / Drinks Alcohol** (binary) show no relationship to BMI, Heart Rate, or Stress — decorative, not linked to any outcome.
- **Daily Coffees / Caffeine Intake** are well-correlated (r=0.85, ~94.6mg/cup) but rigidly deterministic (cups × constant), with no cultural/brew-method variation.
- Missing-value rates are inconsistent (~0.7% baseline on most columns, but Sleep Hours 22.6%, Sleep Quality 15.7%, Health Issues 59.8%) and look like assignment-injected QA issues rather than something to preserve — this dataset will define its own missingness scheme from scratch (see Data Quality Issues below).
- Doc/data mismatches: Physical Activity is documented as categorical but stored as numeric; Age is documented as "whole years" but stored as float with a typo-style anomaly (max 199.7).

---

## Target Variable

**Variable name**: `SelfRatedHealth`
**Type**: Ordinal categorical
**Values**: `{Poor, Fair, Good, Very Good, Excellent}`

**Why this construct**: Self-rated/self-perceived health is a real, widely-used survey measure (Eurostat, WHO, national health surveys), not an invented label. It's subjective by design — which is realistic, since it doesn't reduce to one deterministic formula — and has a well-documented predictor set: sleep quality, physical activity, BMI, smoking, age, and psychological stress ([Sleep & self-rated health, elderly](https://doi.org/10.3390/su10113918), [Determinants of Self-Rated Health, Greece](https://pmc.ncbi.nlm.nih.gov/articles/PMC3367289/), [CDC longitudinal predictors](https://www.cdc.gov/pcd/issues/2014/13_0241.htm)).

### Contributing Factors (draft weights — open to tuning once generated & validated)
1. **Health Issues** (chronic condition severity) — strong negative weight
2. **BMI** — moderate negative weight outside a healthy range
3. **Sleep Quality** — moderate positive weight
4. **Physical Activity Level** — moderate positive weight
5. **Smoking Status** — negative weight, dose-response ordinal: Heavy Smoker > Light Smoker > Former > Never (see Smoking Status below)
6. **Stress Level** — moderate negative weight
7. **Age** — mild negative weight (older skews slightly lower, but not linear — captures general health decline without being deterministic)
8. **Country** — small direct effect on top of the above, reflecting real cross-national self-report differences (see Country Profiles below) — agreed in dialogue as a deliberate, modest effect
9. **Gender** — small direct effect: Female skews slightly lower than Male, reflecting the well-documented "gender-health paradox" (women self-report lower health despite typically greater longevity) — agreed in dialogue as a deliberate, modest effect

### Target Distribution
Should skew positive (matches real self-rated health surveys, where "Good" or better is the majority response) rather than being uniform across 5 classes. Country baselines below give a rough anchor; exact class thresholds to be tuned against validation once first-pass generation exists.

---

## Country Profiles

Four countries, chosen for meaningfully different coffee/lifestyle/health profiles while staying recognizable to UK students — Italy and Norway show that high coffee consumption doesn't predict poor health outcomes, while the UK has the lowest coffee consumption *and* the highest obesity, a genuinely non-obvious pattern for EDA.

| Country | Coffee (kg/yr) | Obesity (adult) | Smoking (current, adult) | Alcohol (L pure/capita) | Self-rated health "good+" | Happiness rank 2025 |
|---|---|---|---|---|---|---|
| Norway | 9.9 | ~23% | 14.2% | ~6 (est.) | ~80% (sourced) | Top 10 |
| Italy | 5.9 (espresso culture) | ~13% (lowest) | 23.4% | 7.7 | 75.5% (sourced) | mid-table |
| France | 5.4 | ~24% | 34.6% (highest) | 10.4 (highest) | 68.5% (sourced, declining — was 75.2% in 2017) | #33 |
| UK | 3.1, rising | ~30% (highest) | 12.5% (lowest) | 9.7 | ~65% (**extrapolated**, not sourced) | Top 25 |

**UK figure is an extrapolation, not a statistic**: no Eurostat/ONS/Health Survey for England figure was confidently sourced. Reasoning: UK has by far the highest obesity of the four (30% vs. France's 24%, more than double Italy's 13%), and obesity is one of the strongest predictors of self-rated health in the literature above — this outweighs UK's smoking advantage (lowest of the four, 12.5%) and lands UK below France (68.5%) despite France's much higher smoking rate. ~65% as the lowest of the four is a reasoned estimate; revisit if we later find a real sourced figure.

**Row distribution**: proposed roughly equal (~2,500/country) for balanced cross-country comparison — open to weighting by real population if you'd prefer that instead.

### Feature-to-country mapping (proposed)
- **Caffeine mg/cup**: vary by country (Italian espresso = smaller serving, higher concentration; Nordic filter coffee = larger volume, different mg profile) rather than a flat 95mg constant.
- **Smoking Status base rates**: Current-smoker % sourced above (France highest, UK lowest); the split of that % into Light/Heavy, and the remainder into Never/Former, is an **estimate**, not sourced — needs either research or an agreed simplifying assumption.
- **Alcohol Level base rates**: banded from L/capita figures above into `{Non-Drinker, Light, Moderate, Heavy}` — banding thresholds are an estimate, open to discussion. (Labeled "Non-Drinker" rather than "None" — see implementation note below.)
- **Physical Activity baseline**: Norway skews more active (outdoor culture); others closer to WHO's ~28%-of-adults-insufficiently-active baseline — not yet country-differentiated beyond Norway, open item.
- **BMI baseline**: shifted per country's obesity rate above.

---

## Dataset Structure (proposed feature list)

### Demographics
1. **ID** — unique identifier, int, no missing, no duplicates (immune field)
2. **Age** — int, adult range (proposed 18-75)
3. **Gender** — {Male, Female, Other}
4. **Country** — {Italy, France, UK, Norway}

Occupation has been dropped (see decision log) — Age, Gender, Country are the demographic anchors.

*v3 adds:* **Household Income** — annual EUR, see the Version 3 section.

### Coffee & Caffeine
5. **Daily Coffees** — cups, float
6. **Caffeine Intake** — mg, correlated with Daily Coffees but with country-varying mg/cup, and *not* a perfect correlation: ~10% of people lean decaf/half-caf (still order "coffees" but most cups are low-caffeine), which is what keeps the relationship strong-but-imperfect for a real reason rather than bare noise — see `DATA_GENERATION_ALGORITHM.md`.

### Lifestyle
7. **Smoking Status** — {Never, Former, *Vaper (v3)*, Light Smoker, Heavy Smoker} (replaces binary Smoker) — intensity matters: smoking dose-response (more cigarettes/day → worse outcomes) is one of the most robustly established findings in epidemiology, so collapsing all current smokers into one bucket would throw away a genuinely important, well-grounded relationship. Exact cigarettes/day threshold splitting Light vs. Heavy (proposed: <10/day vs. ≥10/day) still needs to be pinned down in the algorithm doc.
8. **Alcohol Level** — {Non-Drinker, Light, Moderate, Heavy} (replaces binary Drinks Alcohol)
9. **Physical Activity Level** — {Sedentary, Lightly Active, Moderately Active, Very Active} (standard 4-tier activity classification; replaces noisy 0-15 numeric) — driven by Age, Country, Stress
10. **Avg Sleep Hours Per Night** — float. Renamed from "Sleep Hours" for clarity: represents a typical/average nightly value (not a single night), consistent with how this would realistically be captured — self-reported or aggregated from a sleep-tracking wearable (ring, watch, band) over a recent period.
11. **Sleep Quality** — {Poor, Fair, Good, Excellent} — rebuilt from Sleep Hours + Caffeine (esp. late-day) + Stress + Smoking + Alcohol + Age
12. **Stress Level** — {Low, Medium, High} — generated independently from Age, Country (not derived from Sleep Quality). Ambiguous, realistic measurement source: could be self-evaluated or derived from an activity tracker's stress score — this ambiguity is *why* it's plausible for this field to go missing (see Data Quality Issues).

### Physiology
13. **BMI** — float, country-shifted baseline
14. **Avg Resting Heart Rate** — float, linked to Physical Activity/Smoking/Stress. Renamed from "Heart Rate" for clarity: a typical/average resting value as would be reported by a wearable device or clinical measurement, not a single point-in-time reading.
15. **Health Issues** — {No Issues, Mild, Moderate, Severe} — chronic condition indicator, strengthened links to Age, BMI, Smoking. (Labeled "No Issues" rather than "None" — pandas silently treats the literal string "None" as missing data on CSV read, which would have collided with the deliberate missing-value injection; see `DATA_GENERATION_ALGORITHM.md`'s Implementation Findings.)

### Target
16. **SelfRatedHealth** — {Poor, Fair, Good, Very Good, Excellent} — see Target Variable section above.
    *In v3 this is the CW1 EDA target and CW2's leakage trap; it must be excluded from the ML feature set.*
17. **HighHealthNeeds** *(v3)* — binary, the CW2 ML target — see the Version 3 section.

---

## Data Quality Issues (resolved after two rounds of review)

Following the online-gaming pattern (duplicates + typo-style anomalies + mixed feature/row-level missingness), but with two refinements requested after the first pass:

- **Duplicate rows**: ~0.4%, exact copies
- **Age anomalies**: typo-style doubled-digit errors (e.g. 34 → 344) — Age is the natural candidate since it's the one plausibly self-entered/manually-keyed field. **Fixed after v1**: the original online-gaming-style cap (at 199) caused almost every anomaly to collapse to the same sentinel value, losing the ability to reason about "how did this become wrong" — for our 18-75 age range the doubled value never exceeds 755, so the cap now sits at 999 and never actually binds. Stripping the trailing digit always recovers the exact original age.
- **BMI anomalies**: typo-style missed/misplaced decimal point (e.g. 24.5 → 245), ~0.6%. Added after v1 review, so anomalies aren't Age-only. Same "reasoned, not sentinel" property — dividing by 10 recovers the original value exactly. Note: even at only ~0.6% of rows, these outliers substantially distort the raw BMI↔Activity Pearson correlation (see `validate_dataset.py`'s output) — a real demonstration of outlier sensitivity, not a generation bug.
- **Missing values**:
  - `Health Issues`: flat ~10% (not device-related, so no demographic differential)
  - `Avg Sleep Hours Per Night`, `Avg Resting Heart Rate`, `Stress Level`: **demographic-differential** missingness, scaling up with Age (older people less likely to own/use a wearable or app that auto-logs these) — base rates ~7%/~6%/~5%, multiplied up to ~2.1x for the 60+ age band. This was a deliberate redesign after v1 (which had flat, uniform missingness) specifically so that slicing missingness by Age band reveals a real, explainable pattern rather than noise.
  - Plus a handful of row-level incomplete records (5-10 rows, 4+ missing fields each)
- **Immune fields**: ID, Country, SelfRatedHealth (target)

---

## Open Questions

All resolved for the first generated version — see `DATA_GENERATION_ALGORITHM.md` for exact parameters and `generate_dataset.py`/`validate_dataset.py` for implementation. Remaining items are refinements, not blockers:

- [ ] Never/Former/Light/Heavy smoker split and Alcohol Level banding are still estimates (documented as such in the algorithm doc) rather than sourced — worth deeper research if you want tighter grounding, otherwise fine to leave as reasoned assumptions
- [ ] Manual review of `generated_coffee_health_dataset.csv` — automated validation (`validate_dataset.py`) passes; this is the "your manual check" step per the process doc

## Notes & Decisions Log

- **2026-07-21**: Confirmed source dataset is Kaggle's "Global Coffee Health Dataset" (uom190346a); local copy has been further modified with different QA issues than the Kaggle original.
- **2026-07-21**: Target variable decided: `SelfRatedHealth` (5-level ordinal), chosen over "Overall Health Risk Tier" and "rebuilt Sleep Quality" alternatives.
- **2026-07-21**: Countries decided: Italy, France, UK, Norway.
- **2026-07-21**: Smoking/Alcohol to use categorical levels (not continuous units).
- **2026-07-21**: Country gets a small direct effect on the target, on top of indirect effects via lifestyle features.
- **2026-07-21**: Physical Activity to use the standard 4-tier classification {Sedentary, Lightly Active, Moderately Active, Very Active} rather than 3-tier Low/Medium/High, replacing the current noisy 0-15 numeric scale.
- **2026-07-21**: Gender gets a small direct effect on the target (gender-health paradox: Female skews slightly lower despite typically greater longevity).
- **2026-07-21**: UK self-rated health baseline (~65%) set by extrapolation from its obesity/smoking mix rather than a sourced statistic, since none was confidently found — flagged in the Country Profiles table rather than presented as fact.
- **2026-07-21**: Smoking Status expanded to {Never, Former, Light Smoker, Heavy Smoker} — collapsing intensity would have discarded a well-established dose-response relationship.
- **2026-07-21**: Occupation dropped entirely. Unlike Country/Age/BMI/Smoking, its inclusion wasn't backed by researched statistics in this session (only general domain plausibility) — and with Age, Gender, Country already anchoring demographics, dropping it simplifies the dataset as requested rather than adding an under-grounded feature. Physical Activity Level and Stress Level, which had used Occupation as an input, now derive from Age/Country/Stress and Age/Country respectively instead.
- **2026-07-21**: First dataset generated and validated (`generate_dataset.py` → `generated_coffee_health_dataset.csv`, `validate_dataset.py` all checks passing). Two implementation issues found and fixed during validation: (1) Sleep Quality/Health Issues/SelfRatedHealth switched from hand-picked to quantile-based bucketing thresholds after the first attempt came out badly skewed; (2) the category label `"None"` (Health Issues, Alcohol Level) collided with pandas' default NA strings and was silently converted to missing on CSV read — renamed to `"No Issues"` / `"Non-Drinker"`. Full detail in `DATA_GENERATION_ALGORITHM.md`'s "Implementation Findings" section.
- **2026-07-21 (v1 review)**: After reviewing the first generated dataset, four changes made: (1) Age anomaly cap fixed so it no longer collapses almost all anomalies to one sentinel value (199), preserving the "reason about how this happened" property; (2) BMI anomalies added (decimal-point-shift), so Age isn't the only anomalous field; (3) `Heart Rate` → `Avg Resting Heart Rate` and `Sleep Hours` → `Avg Sleep Hours Per Night`, both documented as wearable/self-reported measurements — the original names were ambiguous about whether they were a single reading or a typical/average value, and about their real-world data source; (4) missingness on the wearable/self-report fields (`Avg Sleep Hours Per Night`, `Avg Resting Heart Rate`, `Stress Level`) redesigned to scale with Age rather than being flat, so the dataset actually has a demographic pattern to discover, not just uniform randomness. Also added a decaf/half-caf subpopulation (~10% of people) to `Caffeine Intake` generation, giving the imperfect Daily Coffees↔Caffeine Intake correlation a real explanation rather than being bare multiplicative noise. The demo notebook (`analysis_demo.ipynb`) was renamed `dataset_validation.ipynb` and reframed as an instructor validation tool rather than a student-facing teaching notebook.
- **2026-07-21 (final sign-off)**: Correction to the entry above — a student-facing notebook was never requested and is **not** a planned deliverable for this dataset (that assumption had mistakenly made its way into the process docs and was removed from `README.md`/`CLAUDE.md`). `dataset_validation.ipynb` is the complete, final notebook deliverable for coffee-health. User reviewed and approved the dataset; status is complete.
