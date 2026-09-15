# Data Generation Algorithm — Coffee & Health

Implements `DATA_GENERATION_SPEC.md`. This doc resolves the spec's remaining open items with concrete, reasoned defaults (marked **[assumption]** where not directly sourced) and pins down the exact formulas `generate_dataset.py` implements. Treat the assumptions as tunable — first-pass output gets checked against the targets below via `validate_dataset.py`, and thresholds get adjusted if they're off, same as online-gaming's iterative approach.

**Target**: 10,000 rows, one row per person. **Row distribution**: equal across countries, ~2,500 each (resolves spec open item).

---

## Generation Order & Dependencies

```
Level 0 (Independent):        ID, Country, Age, Gender
Level 0.5 (v3):                Household Income          <- nudges Levels 1-3
Level 1 (Country-driven lifestyle): Smoking Status, Alcohol Level, Daily Coffees, Caffeine Intake
Level 2 (Behavioral):          Stress Level, Physical Activity Level
Level 3 (Physiology):          BMI, Avg Resting Heart Rate
Level 4 (Sleep):               Avg Sleep Hours Per Night, Sleep Quality
Level 5 (Chronic health):      Health Issues
Level 6 (Target):              SelfRatedHealth
Level 7 (v3 target):           HighHealthNeeds
Level 8 (Data quality issues): missing values, then Age/BMI anomalies, then duplicates
```

A latent **frailty** variable is drawn just before Level 6 and feeds both targets. It is never
written to the CSV — see "Latent frailty" below.

Levels 6 and 7 are run for **two cohorts** (development and held-out test) that differ only in
configuration. See "Two cohorts and distribution shift" at the end of this document, and note the
threshold-reuse requirement there: it is the single easiest thing to get wrong.

---

## Level 0: Demographics

- **Country**: uniform ~2,500 rows each of `{Italy, France, UK, Norway}`.
- **Age**: `Beta(2, 3)` scaled to 18–75 (working-age skew, long tail to 75).
- **Gender**: `{Male: 0.48, Female: 0.48, Other: 0.04}`, same across countries — no research grounding found for country-specific gender skew here, so kept flat rather than invented.

## Level 1: Country-Driven Lifestyle

### Smoking Status `{Never, Former, Light Smoker, Heavy Smoker}`
Current-smoker % is sourced (spec table). Never/Former split and Light/Heavy split within current smokers are **[assumption]**: countries with steeper historical smoking decline are assumed to have more ex-smokers.

| Country | Never | Former | Light Smoker | Heavy Smoker | (Current total) |
|---|---|---|---|---|---|
| Norway | 65.8% | 20% | 9.2% | 5.0% | 14.2% |
| Italy | 58.6% | 18% | 15.2% | 8.2% | 23.4% |
| France | 50.4% | 15% | 22.5% | 12.1% | 34.6% |
| UK | 59.5% | 28% | 8.1% | 4.4% | 12.5% |

Light/Heavy split within current smokers: 65%/35% **[assumption]** — right-skewed, reflecting that heavy (pack-a-day+) smoking has declined faster than light/intermittent smoking in most tobacco-control literature.

### Alcohol Level `{Non-Drinker, Light, Moderate, Heavy}`
Banded from L/capita (spec table: France 10.4 highest → UK 9.7 → Italy 7.7 → Norway ~6 lowest). Banding thresholds are **[assumption]**, ordered consistently with the sourced L/capita ranking. Category is named "Non-Drinker", not "None" — see the "Implementation Finding" note below.

| Country | Non-Drinker | Light | Moderate | Heavy |
|---|---|---|---|---|
| Norway | 25% | 40% | 25% | 10% |
| Italy | 15% | 40% | 35% | 10% |
| UK | 12% | 28% | 40% | 20% |
| France | 10% | 30% | 40% | 20% |

### Daily Coffees & Caffeine Intake
Fixes the original's rigid `cups × 95mg` formula by varying mg/cup per country (brew-culture based) and adding row-level noise.

| Country | Daily Coffees (mean, std) | mg per cup |
|---|---|---|
| Norway | 3.2, 1.0 (large mugs, filter coffee) | 110 |
| Italy | 3.0, 1.1 (small, frequent espresso shots) | 65 |
| France | 2.4, 0.9 | 80 |
| UK | 1.8, 0.9 (still tea-influenced, rising) | 70 |

`Daily Coffees = clip(Normal(mean, std), 0, 9)`

~10% of people are randomly flagged **decaf/half-caf leaning**: their effective mg/cup is scaled by `Uniform(0.05, 0.30)` instead of 1.0. They still show up as ordinary "Daily Coffees" counts, but with much lower caffeine per cup. This is what gives the Daily Coffees ↔ Caffeine Intake relationship a real, discoverable reason to be strong-but-imperfect (r≈0.75, not r≈1.0) rather than being unexplained multiplicative noise — added after v1 review specifically to answer "why isn't this a perfect correlation?" with a real-world cause instead of a shrug.

`Caffeine Intake = Daily Coffees × effective_mg_per_cup × Uniform(0.85, 1.15)`, clipped to `[0, 750]`.

## Level 2: Behavioral

### Stress Level `{Low, Medium, High}`
Country baseline from Happiness Rank 2025 (Norway lowest stress → France highest), then a mild Age adjustment (25–45 "peak career/family load" shifts mass toward Medium/High) **[assumption — Age effect]**. Measurement source is deliberately ambiguous, matching real life: could be self-evaluated or pulled from an activity tracker/wearable's stress score — that ambiguity is why it's plausible for this field to go missing (see Data Quality Issues).

| Country | Low | Medium | High |
|---|---|---|---|
| Norway | 60% | 30% | 10% |
| Italy | 50% | 35% | 15% |
| UK | 45% | 35% | 20% |
| France | 40% | 35% | 25% |

### Physical Activity Level `{Sedentary, Lightly Active, Moderately Active, Very Active}`
Country baseline (Norway's outdoor culture skews active; UK skews sedentary, consistent with its obesity rate), then reduced by High Stress and by Age > 55 **[assumption — magnitudes]**.

| Country | Sedentary | Lightly Active | Moderately Active | Very Active |
|---|---|---|---|---|
| Norway | 15% | 25% | 35% | 25% |
| Italy | 25% | 30% | 30% | 15% |
| France | 25% | 32% | 28% | 15% |
| UK | 30% | 30% | 25% | 15% |

## Level 3: Physiology

### BMI
`BMI = country_baseline_mean + activity_adjustment + smoking_adjustment + age_adjustment + Normal(0, country_std)`, clipped `[16, 45]`.

- Country baseline (mean, std), set from obesity rate ranking: Norway (25.5, 4.2), Italy (24.0, 3.5), France (25.7, 4.3), UK (26.8, 4.8)
- Activity: Very Active −1.5, Moderately −0.7, Lightly 0, Sedentary +1.0
- Smoking: Heavy Smoker −0.5 (nicotine appetite suppression — real, documented effect), others 0
- Age: `+0.03 × min(age, 60)` (BMI creeps up with age, plateaus after 60)

### Avg Resting Heart Rate
`HeartRate = 72 + activity_adj + smoking_adj + stress_adj + age_adj + Normal(0, 7)`, clipped `[45, 110]`.

- Activity: Very Active −6, Moderately −3, Lightly 0, Sedentary +3 (fitness lowers resting HR — well established)
- Smoking: Heavy +6, Light +3, Former +1, Never 0
- Stress: High +4, Medium +2, Low 0
- Age: `+0.05 × age`

Named "Avg Resting Heart Rate" (not just "Heart Rate") after v1 review: the plain name didn't make clear this represents a typical/average value from a wearable or clinical measurement rather than a single point-in-time reading — real resting HR varies meaningfully within a day and across measurement devices, which is part of why this field is plausible to go missing (see Data Quality Issues).

## Level 4: Sleep

### Avg Sleep Hours Per Night
`SleepHours = 7.0 − stress_adj − caffeine_adj + Normal(0, 1.0)`, clipped `[3, 10.5]`.
- Stress: High −0.9, Medium −0.4, Low 0
- Caffeine: `−0.0015 × CaffeineIntake` (mild — real effect is timing-dependent, which we don't model directly, so kept small)

Named "Avg Sleep Hours Per Night" (not just "Sleep Hours") after v1 review, for the same reason as Avg Resting Heart Rate: makes explicit this is a typical/average value (self-reported or wearable-aggregated), not a single night's reading.

### Sleep Quality `{Poor, Fair, Good, Excellent}`
Composite score, not sleep-hours-only (fixes the core original-dataset complaint):
`score = 40 + 6×(SleepHours − 7) − stress_pen − smoking_pen − alcohol_pen − 0.01×CaffeineIntake + Normal(0, 8)`
- Stress: High −15, Medium −6, Low 0
- Smoking: Heavy −8, Light −4, Former −1, Never 0
- Alcohol: Heavy −10, Moderate −4, Light −1, None 0 (alcohol disrupts sleep architecture — well documented)

Bucketed via **quantile thresholds** (see note below) targeting: Poor 10%, Fair 25%, Good 45%, Excellent 20%.

## Level 5: Health Issues `{No Issues, Mild, Moderate, Severe}`

Replaces the original's broken behavior where "None" was never actually assigned (everyone non-missing had some issue — missingness was silently standing in for "none"). Category is named **"No Issues"**, not "None" — see the "Implementation Finding" note below on why the literal string "None" is unsafe to use as a category value. Composite score:
`score = 15 + 0.5×max(0, age−30) + 1.2×max(0, |BMI−22|−3) + smoking_pen + Normal(0, 10)`
- Smoking: Heavy +12, Light +5, Former +2, Never 0

Bucketed via **quantile thresholds** (see note below) targeting: No Issues 55%, Mild 30%, Moderate 11%, Severe 4%.

## Level 6: Target — SelfRatedHealth

`{Poor, Fair, Good, Very Good, Excellent}`, composite score 0–100:

```
score = 60
  + country_offset          # calibrated below to hit sourced/extrapolated "good+" rates
  + gender_offset           # Female -3, Male 0, Other 0  [gender-health paradox]
  - 0.15 × max(0, age - 30)
  + activity_bonus          # Sedentary -8, Lightly -2, Moderately +5, Very Active +12
  - bmi_penalty             # 1.2 × max(0, |BMI-22| - 3)
  + sleep_quality_bonus     # Poor -12, Fair -4, Good +5, Excellent +12
  - smoking_penalty         # Former -3, Light -8, Heavy -18
  - stress_penalty          # Medium -6, High -14
  - health_issues_penalty   # Mild -10, Moderate -25, Severe -45
  + Normal(0, 8)
```

**Country offsets, tuned by validation**: Norway −1.0, Italy −1.75, France +1.25, UK −3.35. These look counter-intuitive at first (Norway's offset is lower than France's, despite Norway's much higher target) — that's because most of the country-level separation is already produced *indirectly*, through Norway/Italy's better smoking/activity/stress profiles pulling their scores up before any direct offset is applied. The direct offset is the *residual* adjustment on top of that, not the whole country effect — see the Implementation Findings note below for how these were derived.

Bucketed via **quantile thresholds** (see note below) targeting: Poor 8%, Fair 17%, Good 40%, Very Good 25%, Excellent 10% (population-level "good or better" ≈ 75%, close to the ~72% average of the four country targets — per-country rates then vary around that due to the offsets above).

**Achieved after calibration** (validate_dataset.py, seed 42): Norway 82.5% (target 80.0%, +2.5pp), Italy 78.7% (75.5%, +3.2pp), France 70.5% (68.5%, +2.0pp), UK 68.2% (65.0%, +3.2pp) — all within the ±5pp tolerance.

### Implementation Findings (from the generate→validate loop)

Two issues surfaced during the first validation pass and were fixed rather than left as spec deviations:

1. **Quantile thresholds instead of hand-picked cut points.** The first attempt used fixed score thresholds (e.g. `<35` Poor, `35-55` Fair, ...) chosen by hand-estimating the weighted formula's average value. That estimate was off (Sleep Quality came out 78% "Poor"), because it's hard to correctly predict where a multi-term weighted sum centers by inspection. Fix: compute thresholds as `np.quantile(score, [cumulative target proportions])` at generation time — this guarantees the marginal distribution hits the target shape regardless of the formula's actual scale, and is the more reliable approach in general.
2. **"None" collides with pandas' default NA strings.** `Health Issues` and `Alcohol Level` originally used the category label `"None"`. On `pd.read_csv()`, pandas' default `na_values` list treats the literal string `"None"` as missing data — so every genuinely-generated "no issues"/"no alcohol" row silently became NaN on load, both erasing the category and wildly inflating the apparent missing-value rate (Health Issues showed ~39% "missing" against an intended ~10%). Fixed by renaming to `"No Issues"` and `"Non-Drinker"`, which don't collide with any pandas NA sentinel. Worth remembering for any future dataset in this repo — avoid `None`/`NA`/`NULL`/`n/a` etc. as literal category values.

The country-offset calibration itself was done by generating the dataset once, checking `validate_dataset.py`'s per-country "good or better" rates against spec targets, then solving for the offset adjustment needed (using the quantile of each country's pre-offset score distribution against the global threshold) rather than guessing-and-checking blindly — same generate→validate→refine loop the process doc describes, just done analytically instead of by trial and error.

---

## Data Quality Issues (resolves spec open item; revised after v1 review)

- **Duplicate rows**: 0.4% (~40 rows), exact copies — consistent with online-gaming's rate for cross-project familiarity.
- **Age anomalies**: ~0.7% typo-style doubled-digit errors (e.g. 34 → 344). **Cap changed from 199 to 999 after v1 review**: with the old cap, almost every anomaly landed on exactly 199 (since doubling the last digit of any age ≥20 already exceeds 199), destroying the "reason about what happened" property the anomaly is supposed to teach. For our 18-75 range the doubled value never exceeds 755, so the 999 cap never actually binds — every anomaly is now exactly invertible by stripping the trailing digit.
- **BMI anomalies**: ~0.6% typo-style missed/misplaced decimal point (e.g. 24.5 → 245). **Added after v1 review** so Age isn't the only field with anomalies. Detect via `BMI > 60`; recover via `/10`. Note: even at this low a rate, these outliers meaningfully distort the raw BMI↔Activity Pearson correlation (see Validation Checks below) — real outlier sensitivity, worth keeping in the dataset as a lesson rather than "fixing" away.
- **Feature-level missing, demographic-differential** (redesigned after v1 review — previously flat/uniform, which gave nothing for students/reviewer to find when slicing by demographics):
  - `Avg Sleep Hours Per Night` (base 7%), `Avg Resting Heart Rate` (base 6%), `Stress Level` (base 5%) — all scale with an age-band multiplier (`age_missingness_multiplier` in `generate_dataset.py`): 0.5× under 30, 0.9× 30-44, 1.4× 45-59, 2.1× 60+. Rationale: older people are less likely to own/use a wearable or app that auto-logs these, so device/self-report fields go missing more as age increases.
  - `Health Issues`: flat ~10% (not device-sourced, so no demographic differential).
- **Row-level missing**: 5-10 rows with 4+ missing fields (incomplete records).
- **Immune fields**: ID, Country, SelfRatedHealth (target).

**Achieved** (seed 42, 10,000 base rows): duplicates 0.40% (40 rows, on target); Age anomalies 0.72%, BMI anomalies 0.56%; missingness by age band (Sleep / Heart Rate / Stress): 18-29 → 3.5% / 2.7% / 3.1%, 30-44 → 6.4% / 5.2% / 4.4%, 45-59 → 9.6% / 7.9% / 6.9%, 60+ → 15.2% / 12.7% / 10.2% — clearly monotonic with age, as intended; Health Issues missing 10.4%.

---

---

## Level 0.5 (v3): Household Income

```
age_factor = 1 + 0.010 x (min(age, 52) - 30) - 0.012 x max(0, age - 60)
income     = INCOME_MEDIAN[country] x age_factor x LogNormal(0, 0.42)
income     = clip(round(income, -2), 12000, 250000)
```

`INCOME_MEDIAN` = Norway 62,000 / France 44,000 / UK 42,000 / Italy 36,000 EUR.

A standardised `income_z = (log(income) - mean) / sd` then nudges three downstream features:

- **Smoking Status** — with probability `clip(0.10 + 0.09 x income_z, 0, 0.40)`, shift one step
  *down* the smoking scale (towards Never). Applied after the country draw.
- **Physical Activity Level** — with probability `clip(0.12 + 0.10 x income_z, 0, 0.40)`, shift
  one step *up*. Applied after the stress and age adjustments.
- **BMI** — additive `-1.2e-5 x (income - 45000)`.

Income's *direct* contribution to the binary target is deliberately small; most of its association
with health is mediated by those three. That keeps the socioeconomic gradient visible in EDA while
leaving income a poor basis for a nearest-neighbour distance — which is what makes feature scaling
demonstrably necessary.

---

## Latent frailty (v3)

```
frailty ~ Normal(0, 1)          # per person, never written to the CSV
srh_score += -18.0 x frailty
eta       +=  +1.35 x frailty
```

Drawn immediately before `SelfRatedHealth`. This is the only thing that makes `SelfRatedHealth`
informative about `HighHealthNeeds` beyond the measured features, and therefore the only thing
that makes it a real leakage trap rather than a redundant column. Without it the measured leakage
benefit is approximately zero.

---

## Level 7 (v3): HighHealthNeeds

```
eta = linear + non_linear + interactions + 1.35 x frailty + Normal(0, 0.45)
p   = 1 / (1 + exp(-eta))
y   = 1 if Uniform(0,1) < p else 0        # a DRAW, not a threshold
```

A Bernoulli draw rather than a cut on the score is what gives the problem irreducible noise and a
realistic ceiling (Bayes AUC ~0.93 on these coefficients). The full coefficient tables are in
`DATA_GENERATION_SPEC.md`; the three blocks are:

**`linear`** — intercept plus additive terms in smoking, activity, stress, sleep quality, health
issues, alcohol, country, gender, age, BMI deviation, resting heart rate and income.

**`non_linear`** — `0.72 x (sleep_hours - 6.4)^2 + 0.34 x (cups - 2.8)^2`.

> Both centres are the **middle of the observed distribution**, not a textbook ideal. Mean sleep is
> 6.36 hours, so an earlier draft centred at 7.25 put 79% of the data on one arm; the term
> correlated −0.76 with raw hours, a linear model captured it for free, and the measured gap
> between logistic regression and a gradient-boosted tree came out **negative**. Re-centring on the
> data moved that gap to +0.048 in one change. If these coefficients are ever retuned, check
> `corr(term, raw_feature)` stays near zero.

**`interactions`** — two continuous products (`bmi_deviation x years_over_30`,
`caffeine x sleep_debt`), one that modulates the coffee curve
(`min(cups,3) x sleeps_badly`), and four categorical terms defined over deliberately **broad**
groups: any-smoker x BMI≥27, stressed x poor-or-fair sleep, over-55 x sedentary, over-55 x very
active. Narrow interactions firing on 2% of rows are invisible in aggregate performance and cannot
reward a more expressive model.

**Calibration.** `HHN_INTERCEPT` is set so the positive rate lands at ~20%. Any change to the
coefficients or to `HHN_FRAILTY`/`HHN_NOISE_SD` changes the variance of `eta` and therefore the
positive rate, so the intercept must be re-solved — bisect on the mean of `sigmoid(eta)` using the
*combined* unobserved standard deviation `hypot(HHN_FRAILTY, HHN_NOISE_SD)`, not the idiosyncratic
noise alone.

---

## Two cohorts and distribution shift (v3)

`CohortConfig` holds everything that differs between the development and test cohorts: country
mix, age Beta parameters, vaper share, whether missingness is injected, the two anomaly rates, and
the duplicate rate and UK weighting. **Nothing in the causal model varies**, so `P(y | x_clean)`
is identical and all shift is covariate shift.

### The threshold-reuse requirement

`Sleep Quality`, `Health Issues` and `SelfRatedHealth` are all bucketed at **quantiles computed
within the call**. If the test cohort recomputed its own quantiles, each of those three would
re-normalise to exactly the development set's class proportions and **every shift in them would be
silently erased** — the data would look correct and the drift exercise would have no content.

`generate()` therefore returns the thresholds it computed, and the test call is passed them:

```python
dev,  thresholds = generate(DEV_CONFIG,  rows,      seed)
test, _          = generate(TEST_CONFIG, test_rows, seed + 2, thresholds=thresholds)
```

`HighHealthNeeds` needs no such treatment — it is a Bernoulli draw from a sigmoid, so the shifted
covariates move the positive rate on their own. That is why the label shift is *automatically*
compositional rather than imposed, which is precisely the property the drift exercise turns on.

### Quality issues, per cohort

Injection is three independent steps so cohorts can take different subsets:

1. **Missing values** — development only. Device fields (`Avg Sleep Hours Per Night` 7%,
   `Avg Resting Heart Rate` 6%, `Stress Level` 5%) scaled by an age multiplier
   (0.5 / 0.9 / 1.4 / 2.1 by band); `Health Issues` a flat 10%; plus 5–10 row-level incomplete
   records. Runs **before** the anomalies, so the age-differential signal is computed on clean ages.
2. **Anomalies** — both cohorts, development 0.7% age / 0.6% BMI, test 3.0% / 2.5%. Runs **after**
   the label is drawn, so an anomalous row's label reflects its true values while its features are
   corrupted.
3. **Duplicates** — both cohorts, development 0.4% uniform, test 2.0% sampled with UK rows weighted
   x4.

`HighHealthNeeds` joins `ID`, `Country` and `SelfRatedHealth` in the immune set, so no target is
ever nulled.

---

## Validation Checks

1. Feature ranges/types match spec (Age/BMI range checks allow for anomaly-inclusive upper bounds)
2. Country/Gender/Smoking/Alcohol/Activity/Stress distributions match target probabilities (±3-5%)
3. SelfRatedHealth "Good or better" rate per country within ~±5% of spec targets (Norway 80%, Italy 75.5%, France 68.5%, UK 65%)
4. Key correlations present and correctly signed, checked both raw (anomalies included, to see their distorting effect) and with Age/BMI anomalies excluded (to validate the actual underlying relationship): BMI↔Activity (negative), Avg Resting Heart Rate↔Activity (negative), SleepQuality↔Stress (negative), SelfRatedHealth↔HealthIssues (strong negative)
5. Missingness on Avg Sleep Hours Per Night / Avg Resting Heart Rate / Stress Level increases monotonically across age bands (18-29 < 30-44 < 45-59 < 60+)
6. Logical consistency: Never/Former smokers have no Light/Heavy status; missing-value rates match spec; duplicate count matches spec
7. No feature is 100% deterministic from another (the original dataset's core flaw)
