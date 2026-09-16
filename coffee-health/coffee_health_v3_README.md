# Coffee & Health Dataset — v3 Reference

**Instructor-facing.** This document contains the performance ladder and the full distribution-shift
design, so it must not be handed to students. The student brief is a separate artifact — see
`data/coffee_health_v2_README.md` for the v2 example of that format.

v3 carries a two-assignment unit: **CW1** on exploratory data analysis, **CW2** on applied machine
learning. Design rationale is in [V3_COURSEWORK_DESIGN.md](V3_COURSEWORK_DESIGN.md); generation
detail is in [DATA_GENERATION_SPEC.md](DATA_GENERATION_SPEC.md) and
[DATA_GENERATION_ALGORITHM.md](DATA_GENERATION_ALGORITHM.md).

---

## 1. The two files

| | `coffee_health_v3_dev.csv` | `coffee_health_v3_test.csv` |
|---|---|---|
| Rows | 10,040 (10,000 + 40 duplicates) | 3,060 (3,000 + 60 duplicates) |
| Missing values | 2,977 cells | **none** |
| Purpose | everything: cleaning, EDA, training, validation | scored once, at the end |
| Permitted operations | all | **encoding and scaling only** |

Release the test set whenever suits the unit — for example after CW1 has been submitted.

## 2. Columns (18)

| # | Column | Type | Notes |
|---|---|---|---|
| 1 | `ID` | int | Identifier, not a feature. Repeats on duplicated rows. |
| 2 | `Country` | cat | Italy / France / UK / Norway |
| 3 | `Age` | int | 18–75. Values >100 are data-entry anomalies. |
| 4 | `Gender` | cat | Male / Female / Other |
| 5 | `Household Income` | float | EUR/year, 12,000–250,000. **New in v3.** |
| 6 | `Smoking Status` | ord | Never / Former / **Vaper** / Light Smoker / Heavy Smoker |
| 7 | `Alcohol Level` | ord | Non-Drinker / Light / Moderate / Heavy |
| 8 | `Daily Coffees` | float | Cups per day, 0–9 |
| 9 | `Caffeine Intake` | float | mg/day, 0–750 |
| 10 | `Stress Level` | ord | Low / Medium / High |
| 11 | `Physical Activity Level` | ord | Sedentary / Lightly / Moderately / Very Active |
| 12 | `BMI` | float | 16–45. Values >60 are misplaced decimal points. |
| 13 | `Avg Resting Heart Rate` | float | 45–110 bpm |
| 14 | `Avg Sleep Hours Per Night` | float | 3–10.5 |
| 15 | `Sleep Quality` | ord | Poor / Fair / Good / Excellent |
| 16 | `Health Issues` | ord | No Issues / Mild / Moderate / Severe |
| 17 | `SelfRatedHealth` | ord | **CW1 target.** Poor → Excellent |
| 18 | `HighHealthNeeds` | binary | **CW2 target.** ~20% positive |

### The two targets

**`SelfRatedHealth`** — the 5-class EDA target for CW1. In CW2 it is a **leakage trap**: it must be
excluded from the feature set, because (a) it comes from a survey that is not run on the deployment
population, and (b) it shares a latent frailty term with the outcome, so it is a partial observation
of the thing being predicted. Including it gains +0.025 F1 and over six points of accuracy. It is
then useful again for CW2 error analysis, as a segmentation variable.

**`HighHealthNeeds`** — binary: did this person have ≥14 days of health-related absence, **or** ≥6
primary-care contacts, in the twelve months after the survey? Development 19.8% positive, test 25.6%.

---

## 3. What EDA should find

All figures below are measured on the development set with anomalies and missing rows excluded.

### Correlations with `SelfRatedHealth`

| Feature | r | | Feature | r |
|---|---|---|---|---|
| Health Issues | **−0.47** | | Physical Activity | **+0.33** |
| Sleep Quality | **+0.38** | | Stress Level | **−0.31** |
| BMI | −0.27 | | Smoking Status | −0.26 |
| Resting Heart Rate | −0.26 | | Age | −0.18 |
| Sleep Hours | +0.18 | | Household Income | +0.05 |

`SelfRatedHealth` itself is distributed Poor 8% / Fair 17% / Good 40% / Very Good 25% /
Excellent 10%, and the high-needs rate within those levels runs 65.7% / 40.0% / 16.0% / 5.1% /
1.6% — strongly informative about the CW2 target without being deterministic, which is what makes
it a usable leakage trap.

### Strong feature-to-feature relationships

`Daily Coffees ↔ Caffeine Intake` **+0.75** (imperfect because ~10% of people lean decaf) ·
`Sleep Hours ↔ Sleep Quality` **+0.59** · `Stress ↔ Sleep Quality` **−0.54** ·
`Resting HR ↔ Activity` **−0.42** · `Age ↔ Health Issues` **+0.39** · `BMI ↔ Health Issues` **+0.27**

### Two relationships that are invisible to correlation

This is the most valuable EDA lesson in the dataset, and it is what rewards binning over a
correlation matrix.

**The coffee J-curve.** Linear correlation with the outcome is **−0.02** — apparently nothing. Binned:

| Cups/day | 0–1 | 1–2 | 2–3 | 3–4 | 4+ |
|---|---|---|---|---|---|
| High-needs rate | **30.6%** | 19.4% | **16.8%** | 18.3% | **23.7%** |

**The sleep U-curve.** Both short *and* long sleep are harmful:

| Hours | <4.5 | 4.5–5.5 | 5.5–6.5 | 6.5–7.5 | 7.5–8.5 | >8.5 |
|---|---|---|---|---|---|---|
| High-needs rate | **75.0%** | 34.7% | 13.5% | **9.3%** | 14.6% | **42.0%** |

### Interactions worth finding

**Smoking × overweight is strongly super-additive.** Additivity would predict ~29.8%; the observed
rate is 45.0%.

| | BMI < 27 | BMI ≥ 27 |
|---|---|---|
| Non-smoker | 13.0% | 20.5% |
| Smoker/vaper | 22.3% | **45.0%** |

**Activity matters far more with age.** A sedentary over-55 is at more than double a sedentary
under-55's risk, while a very active over-55 is at *lower* risk than a very active under-55.

| | Sedentary | Lightly | Moderately | Very Active |
|---|---|---|---|---|
| Age ≤ 55 | 24.0% | 18.8% | 17.1% | 12.0% |
| Age > 55 | **53.2%** | 28.2% | 24.0% | **9.6%** |

### A confounded correlation that disappears under stratification

`Household Income ↔ Caffeine Intake` is **+0.245** overall — but within every country it is
approximately **zero** (France −0.01, Italy −0.01, Norway +0.06, UK −0.02). Norway has both the
highest incomes and much stronger coffee (319 mg/day against the UK's 116). The correlation is
entirely a country effect. A genuinely good CW1 submission stratifies and reports this.

### The socioeconomic gradient is real but mediated

Income correlates only +0.05 with self-rated health directly, because most of its effect runs
through behaviour:

| Income quintile | Q1 lowest | Q2 | Q3 | Q4 | Q5 highest |
|---|---|---|---|---|---|
| Share sedentary | 26.9% | 27.9% | 25.2% | 20.9% | **17.7%** |
| Share smoking | 21.2% | 22.1% | 18.1% | 17.9% | **15.3%** |
| Mean BMI | 26.6 | 26.4 | 26.5 | 26.2 | **26.0** |
| High-needs rate | 21.8% | 19.0% | 21.2% | 19.0% | **16.9%** |

### Country profiles

| | Norway | Italy | France | UK |
|---|---|---|---|---|
| "Good or better" health | **83.2%** | 78.0% | 71.7% | **69.6%** |
| High-needs rate | 14.7% | 15.3% | 23.7% | **24.7%** |
| Coffees / caffeine | 3.17 / **319mg** | 3.01 / 178mg | 2.37 / 174mg | **1.80** / **116mg** |
| Mean BMI | 25.8 | 25.0 | 26.6 | **27.9** |

### Two further discoveries

**Women rate their health worse than men** (mean 2.09 vs 2.18 on the 0–4 scale) despite near-identical
rates of diagnosed health issues (43.3% vs 43.0%) and a *higher* high-needs rate (20.5% vs 18.9%) —
a well-documented reporting asymmetry, and a good prompt for discussing self-report as a measure.

**Vapers report better health than smokers but fare worse.** Mean self-rated health places vaping
between former (2.11) and light smoking (1.83) at 1.98, while the high-needs rate places it
*above* light smoking (31.8% vs 29.4%). This is deliberate — see §7.

---

## 4. Data quality issues

| Issue | Development | Test | Recoverable? |
|---|---|---|---|
| Duplicate rows | 40 (0.40%) | 60 (2.0%), **83% UK** | drop them in dev; analyse only in test |
| Age anomalies | 71 (0.71%) | 101 (3.30%) | yes — drop the trailing digit (`344 → 34`) |
| BMI anomalies | 56 (0.56%) | 51 (1.67%) | yes — divide by 10 (`245 → 24.5`) |
| Missing: Sleep Hours | 7.5% | none | age-differential |
| Missing: Resting HR | 6.1% | none | age-differential |
| Missing: Stress Level | 5.3% | none | age-differential |
| Missing: Health Issues | 10.4% | none | flat rate |
| Row-level incomplete records | 5–10 rows, 4+ fields each | none | — |

**The missingness is not random.** Device-sourced fields go missing more often with age — older
people are less likely to use a wearable that auto-logs sleep, heart rate and stress. `dropna()`
therefore removes 26% of rows and skews the sample younger (mean age 42.9 dropped vs 39.9 kept).
Both anomaly types are *exactly* recoverable, so a student can reason their way back to the true
value rather than hitting a sentinel.

---

## 5. Distribution shift in the test set

**The causal model is identical between the two files.** Every coefficient is shared, so
`P(y | x_clean)` is unchanged and all shift is **covariate shift**. There is deliberately no concept
drift — that would leave students nothing to do but complain.

| | Development | Test |
|---|---|---|
| UK share | 25.0% | **46.7%** |
| Mean age | 40.7 | **46.2** |
| Vaper share | 2.7% | **9.7%** |
| Positive rate | 19.8% | **25.6%** |

In-world story: a later recruitment wave that leaned heavily on UK sites, recruited through GP
practices (hence older), ran after vaping had become far more common, and used a different
data-entry process (hence more anomalies), with a double-submission incident at the UK site (hence
UK-weighted duplicates).

### The finding that separates the best submissions

The positive rate rises by 5.8 points. That *looks* like label shift, which would mean the model is
stale. It is not.

- Reweighting the development set's own within-group rates (country × age band × smoking) to the
  test set's composition recovers **+0.034 of the +0.058** rise.
- Within each country, the model's **precision barely moves** (mean absolute gap ~5pp) and recall
  mostly rises. A model whose learned relationship had gone stale would get *worse* at identifying
  cases inside a country, not better.

So the relationship between features and outcome is intact; what changed is *who was recruited*.
The model does not need retraining — but the **costs each stakeholder should expect do need
recalculating**, because they were estimated against a population with a different mix.

Coarse stratification tells you *which variables* drive the shift; per-subgroup precision and
recall tell you *whether the relationship itself* moved. The second is the decisive test, and is
worth teaching as such.

### Why test performance is lower — three separable layers

1. **Optimism / overfitting** — training-set accuracy 0.925 vs 5-fold CV 0.851.
2. **Corrupted rows** — anomalies at ~4× the development rate. Because anomalies are injected
   *after* the label is drawn, these rows carry labels from their true values while presenting
   corrupted ones, so they are near-unpredictable by construction. Students may not repair them.
   Duplicates compound this: a duplicated row the model gets wrong is counted twice.
3. **Covariate shift** — as above.

**Accuracy falls while precision and recall hold or improve.** Within each country the model's
precision moves by about five percentage points and its recall mostly rises; what falls is
accuracy, the metric most sensitive to how common the positive class is. Reporting accuracy alone
makes this look like model failure. It is not.

---

## 6. The performance ladder — what to expect students to find

Measured on the development set, 20% hold-out, fixed seeds. **These are the reference points for
what should change and by how much.** `validate_ml_ladder.py` asserts all 23 of them; if the
generator is ever retuned, run it or the assignment quietly stops working.

| Rung | Change | Headline effect |
|---|---|---|
| 0 | Predict the majority class | accuracy **0.804**, recall 0 |
| 1 | **Starter pipeline** — defaults, unscaled, single split, accuracy | accuracy 0.830 against a 0.804 baseline, but **recall 0.156** — it finds almost nobody |
| 2 | Repair anomalies, impute rather than drop | +35% training rows; small and model-dependent gain. **Justified on bias, not accuracy** |
| 3 | Feature scaling | KNN F1 **0.107 → 0.403**; unscaled recall is 0.068. Barely affects logistic regression |
| 4 | Class weighting or SMOTE | recall **0.326 → 0.755**; balanced accuracy +0.118; agency cost **+€154 → −€3** per person |
| 5 | Hyperparameter tuning | random forest F1 **0.451 → 0.601**, recall 0.333 → 0.664, agency cost €149 → €21. Tuning LR's `C` changes almost nothing |
| 6 | Cross-validation / repeated hold-out | two models with means **0.5800 and 0.5802** swap rank on **4/10** splits |
| 7 | Gradient boosting vs logistic regression, like for like | **+0.070 F1** (0.555 → 0.625), both class-weighted and tuned |
| 8 | Pricing models with the cost matrices | the most accurate model is **worst for both** customers; pairs of models split the two stakeholders |
| 9 | Spotting the leak | including `SelfRatedHealth` gives +0.025 F1 and +6.6pp accuracy |
| 10 | Train vs CV vs test | **0.925 → 0.851 → 0.826** accuracy |

### Traps worth marking for

- **Optimising the wrong thing.** Accuracy is the default instinct and it is the wrong objective
  here: the model with the best accuracy in the whole exercise is the one that costs both
  stakeholders the most, because accuracy is dominated by the 80% of people who were never going
  to have a high-needs year.
- **Improvements that pay nothing on their own.** Better cleaning does not move the headline metric
  until scaling and class weighting are also in place. Reporting a change that did not help, and
  arguing for keeping it anyway, is stronger work than hiding it.
- **Resampling the test set.** SMOTE belongs to training data only.
- **Stopping at "the data drifted."** See §5.

---

## 7. Two stakeholders, two cost matrices

Full scenario prose in [STAKEHOLDERS_AND_COSTS.md](STAKEHOLDERS_AND_COSTS.md); both matrices are in
`costs.py` and are already used in the starter notebook.

| | TP | FN | FP | TN | What it wants |
|---|---|---|---|---|---|
| **National public health agency** | −€720 | **€1,450** | €180 | €0 | recall — a miss costs ~8x a false alarm |
| **Employer occupational health** | −€80 | €780 | **€520** | €0 | precision — the two mistakes cost about the same |

Everything is computed from `model.predict()`: there are no predicted probabilities and no decision
thresholds anywhere in the assignment. The lever students pull is **class weighting**, and it is
enough to split the two customers — logistic regression with and without
`class_weight='balanced'` is preferred by opposite stakeholders. "Which model is best?" cannot be
answered without "best for whom?".

### A deliberate design choice worth knowing about

**Vaping is modelled as carrying more near-term health burden than light smoking, while being
self-rated as less serious.** The smoking dose-response is therefore clean on `SelfRatedHealth`
(2.32 → 2.11 → 1.98 → 1.83 → 1.23) but *not* on `HighHealthNeeds`, where Vaper (31.8%) sits above
Light Smoker (29.4%).

This is intentional and does two jobs: it gives CW1 a genuine insight about self-report diverging
from outcomes, and it gives CW2 a subgroup that is rare in training (2.7%, n≈201) but common in test
(9.7%) and higher-risk than the model will have learned. If you would rather the dose-response be
monotonic on both targets, set `HHN_SMOKING['Vaper']` to below the `Light Smoker` value in
`generate_dataset.py` (currently 0.60 vs 0.45), then regenerate and re-run both validators.

---

## 8. Regenerating and checking

```bash
python3 generate_dataset.py --seed 42     # writes both CSVs, byte-reproducible
python3 validate_dataset.py               # 67 checks: ranges, distributions, quality, shift
python3 validate_ml_ladder.py             # 23 checks: every rung of §6
```

Two failure modes to know about before touching the generator, both documented in
`DATA_GENERATION_ALGORITHM.md`:

1. **Quadratic terms must be centred on the observed distribution.** An off-centre quadratic is
   near-linear across the support, so a linear model captures it for free and the tree-ensemble gain
   disappears. Check `corr(term, raw_feature)` stays near zero.
2. **The test cohort must reuse the development set's quantile thresholds.** The three ordinal
   composites are bucketed at quantiles computed within the call; recomputing them re-normalises
   every class to identical proportions and silently erases the shift.
