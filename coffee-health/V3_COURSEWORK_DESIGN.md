# Coffee Health v3 — Coursework Design

This document records **why v3 exists and how the coursework built on it works**. It is
deliberately separate from [DATA_GENERATION_SPEC.md](DATA_GENERATION_SPEC.md), which records
*how the data is made*. When the two disagree, the spec wins on mechanics and this document wins
on intent.

---

## 1. Why v3

v2 is a 10,040-row dataset with a single 5-class target, `SelfRatedHealth`, built for teaching
EDA with a light applied-ML demonstration attached. v3 repurposes it to carry a **two-assignment
unit**:

- **CW1 (40%) — Exploratory Data Analysis.** Basic dataset description; data-quality analysis
  (missing values, duplicates, noise/anomalies); insight generation (demographic, clustering and
  correlation analysis).
- **CW2 (60%) — Applied Machine Learning.** Students are handed a deliberately weak baseline
  pipeline and must improve and critically evaluate it.

Three problems in previous runs of the unit drive the redesign.

**CW2 was too coding-heavy, and AI has made that worse.** The last cohort produced large volumes
of near-identical AI-generated pipeline code. Marking that code distinguishes almost nobody. The
response is to *supply* the pipeline and move the marks onto evaluation, justification and
analysis — work that is much harder to outsource and much closer to what the unit is actually
trying to teach.

**The link between CW1 and CW2 was weak.** EDA was treated as a box ticked in the first
assignment and then abandoned. In v3 the EDA skills from CW1 are the tools needed to do well in
CW2: error analysis by subgroup, interrogating whether the test set is representative, and
diagnosing distribution shift are all CW1 techniques applied to a new question.

**Students struggled to connect results to the real world.** A model was "good" because accuracy
was high, with no sense of what a mistake costs anyone. The response is a binary target with two
stakeholder cost matrices that pull in opposite directions, so "which model is better?" has no
single answer and must be argued.

### The hard design constraint

**Every rung of the improvement ladder must measurably pay off.** It is pointless to ask students
to scale features, handle class imbalance, tune hyperparameters, improve validation and
introduce a more expressive model if those things do not change the numbers. It is equally
pointless if scaling alone reaches a ceiling and everything after it is noise.

v2's data cannot deliver this. `SelfRatedHealth` is a purely additive, linear composite of its
inputs, which puts logistic regression close to Bayes-optimal — "introduce a more complex model"
would gain essentially nothing. Building genuine non-linearity and interaction structure into the
new target is therefore the core data-design work of v3, and the ladder is enforced by an
automated check (§7) rather than assumed.

---

## 2. The binary target: `HighHealthBurden`

**Definition.** Did the person have a high-burden health year in the twelve months following the
survey — **≥14 days of health-related absence from work or normal activity, OR ≥6 primary-care
contacts**?

It is a forward-looking *event*, not a relabelling of `SelfRatedHealth`. This matters: v2 exists
precisely because the original Kaggle dataset was riddled with deterministic relabelling (Stress
Level was a 100% deterministic function of Sleep Quality), and reintroducing that flaw in the
target would undo the point of the dataset.

The composite definition is chosen so that **both stakeholders genuinely care about the same
event**: the employer pays for the absence, the health service pays for the contacts.

### How it is generated

`y ~ Bernoulli(sigmoid(eta))` — a **stochastic draw, not a threshold**. This gives irreducible
noise and therefore a realistic performance ceiling (target Bayes AUC ~0.88-0.90), rather than a
deterministic boundary a sufficiently flexible model could learn perfectly.

`eta` shares its drivers with `SelfRatedHealth` but adds structure v2 does not have at all:

| Component | Why it is there |
|---|---|
| Linear terms: age, BMI deviation, smoking, activity, stress, sleep quality, health issues, alcohol, income, country | Baseline signal that logistic regression can capture |
| **U-shaped sleep hours**, `(hours - 7.25)^2` | Short *and* long sleep are both harmful — real epidemiology, and invisible to a linear model on raw hours |
| **J-shaped coffee curve**: moderate protective, heavy harmful | Genuinely evidenced in the literature, and thematically ideal for a coffee dataset |
| **Interactions**: heavy smoking x obesity; high stress x poor sleep; high caffeine x short sleep; age>55 x sedentary; age>55 x very active (protective) | The main reason a tree ensemble or MLP will beat logistic regression |
| Logistic noise | Sets the ceiling |

**Positive rate is calibrated to ~20%**, and that number is doing deliberate work. A
majority-class baseline scores **80% accuracy**. The starter pipeline will therefore look
respectable at ~81% while achieving roughly 25% recall on the positive class. That gap is the
most useful teaching moment available in the whole assignment, and it connects directly to the
false-negative-heavy cost matrix. Roughly 2,000 positives is enough for stable learning, and few
enough that imbalance handling moves the needle hard.

`SelfRatedHealth` is **retained** as the 5-class target, regenerated to pick up the new features,
so CW1's EDA stays rich.

---

## 3. The leakage design

`SelfRatedHealth` is CW2's leakage trap, and it is defensible on two levels:

- **Statistical.** It is a strong proxy for the same latent health construct as the target, so
  including it as a feature gives a seductive accuracy jump of roughly 8-12 percentage points —
  big enough to look obviously too good, small enough not to be degenerate.
- **Deployment.** The stated scenario is that the model scores people from *routinely available*
  administrative and wearable data. `SelfRatedHealth` comes from a survey that is not collected
  for the deployment population. This is availability-at-inference-time leakage, and it is the
  more valuable of the two lessons.

One column therefore does three jobs across the unit: **CW1's EDA target**, **CW2's leakage
trap**, and **CW2's error-analysis segmentation variable** ("which self-rated-health groups does
my model misclassify?").

---

## 4. The two stakeholders

Full brief-ready prose and the cost matrices live in
[STAKEHOLDERS_AND_COSTS.md](STAKEHOLDERS_AND_COSTS.md); both matrices are mirrored in code in
[costs.py](costs.py) and are part of the starter notebook alongside accuracy.

In summary: a **national public health agency** whose false negatives are catastrophic
(cost-optimal threshold ~0.077, strongly recall-oriented) and an **employer occupational health
team** with a capped budget whose false positives are expensive (cost-optimal threshold ~0.377,
precision-oriented). The default 0.5 threshold is wrong for both. Models can — and should —
legitimately swap rank between the two.

---

## 5. Making every improvement pay off

| Lever students should discover | How the data guarantees a payoff |
|---|---|
| **Better data-quality handling** | Age typos and BMI-decimal anomalies in the *training* data; repairing rather than dropping them, and imputing rather than dropping whole rows, is worth ~1-2pp |
| **Feature scaling** | `Household Income` (~EUR 15k-150k) is large in magnitude but weak in signal. Unscaled KNN/MLP distance is dominated by near-noise while the informative features sit on small scales, so scaling produces a large and unmistakable jump. It also earns its keep in CW1, since the socioeconomic health gradient is real |
| **Class imbalance handling** | 20% positive, overlapping class-conditional distributions and dense mass near the boundary, so class weighting or SMOTE moves positive-class recall from ~0.25 to ~0.60 and slashes cost under the public-health matrix |
| **Hyperparameter tuning** | A noisy latent model with a smooth boundary makes KNN's default `k=5` badly suboptimal (optimum ~30-60), and an unscaled MLP at defaults fails to converge |
| **A more expressive model** | The interactions and the U- and J-shaped terms are invisible to logistic regression on raw features; a tuned gradient-boosted tree should gain ~0.03-0.05 AUC over tuned LR. **This is the change v2 most needs** |
| **Better validation** | With ~400 positives in a 20% hold-out, minority-class recall and F1 vary by roughly +/-3-4pp across split seeds — enough to flip model rankings, so students can *demonstrate* that single hold-out is unreliable rather than being told |

---

## 6. The two data files, and why test performance is lower

### Development set

~10,000 rows plus ~40 duplicates, labelled, and messy. Students do everything here: cleaning,
imputation, resampling, tuning, validation.

### Held-out test set

~3,000 rows, labelled, and **free of missing values**. Released at the unit leader's discretion —
for example, after CW1 has been submitted.

The only operations students should apply to it are **encoding of non-numeric features and
scaling**. Performance must not vary according to how a student handled missing values; that is a
training-data concern, and letting it leak into test scoring would make the leaderboard measure
the wrong thing. The in-world framing is simply that a clean test extract was provided.

The test set does, however, carry two kinds of problem that are **for analysis, not remediation**:

- **Anomalies** (age typos, BMI recorded ten times too large) at a **higher rate than the
  development set** — roughly 1.5% and 1.2% against 0.7% and 0.6% — framed as a different
  data-entry process at the newer recruitment site. Because anomalies are injected *after* the
  label is drawn, these rows carry labels generated from clean values but corrupted features, so
  they are near-guaranteed misclassifications.
- **Duplicates** at ~2% (around 60 rows), drawn disproportionately from UK respondents and framed
  as a double-submission at the UK recruitment site. They skew the country mix and class balance
  further, and any duplicated row the model gets wrong is counted twice in the test metric.

Every student scores against the same test set, so the comparison stays fair.

### Three distinct reasons test performance is lower

This is the analytical spine of CW2, and the layers are deliberately separable so that students
can be credited for how deep they get.

**Layer 1 — optimism and overfitting (shallowest).** Resubstitution accuracy on the training data
is much higher than the cross-validated estimate, which is in turn higher than test. Default
KNN(k=5) and unregularised trees make this vivid. The intended realisation is "my training score
was never an estimate of anything".

**Layer 2 — measurement corruption in the test set (middle).** The elevated anomaly rate produces
a small set of essentially unpredictable rows, and the duplicates double-count some errors. Both
are quantifiable: identify them, exclude them, recompute, and report the difference.

**Layer 3 — distribution shift (deepest, top marks).** The test set comes from a different
recruitment mix:

- **Country mix** — UK over-represented, 40% against 25%, and skewed further by the UK-weighted
  duplicates. Detectable with a single bar chart; pure CW1 skill transfer.
- **Age drift** — the test cohort is around three years older, framed as GP-practice recruitment.
- **Vaper prevalence** — a `Vaper / E-cigarette` smoking category at ~1.5% in development and ~9%
  in test. Present in both, so naive one-hot encoding does not crash, but subgroup performance
  and calibration suffer.
- **Apparent label shift** — the positive rate rises from ~20% to ~25%. Because `y` is drawn from
  the *same* sigmoid over shifted covariates, this emerges automatically and is **entirely
  compositional**. Students who decompose by country and age discover it is not a genuine change
  in underlying risk. "It looks like label shift but it is covariate shift" is the best available
  high-marks finding, and it falls out of the construction rather than being bolted on.

**`P(y | x_clean)` is identical between the two files by construction** — the coefficients of
`eta` do not change. There is deliberately **no concept drift**: it would leave students with
nothing to do but complain, whereas covariate shift is diagnosable and partly fixable through
reweighting and stratified reporting.

Shift magnitudes are tuned so that test performance lands roughly 2-4pp below the cross-validated
estimate: enough to notice and investigate, not enough to dominate the assignment.

---

## 7. The improvement ladder, enforced automatically

The central risk in this design is that the improvements students are asked to make turn out not
to matter. `validate_ml_ladder.py` converts that risk into an automated assertion: it runs the
whole ladder with fixed seeds and **fails** if a rung does not pay off.

| Rung | Setup | Assertion |
|---|---|---|
| 0 | Majority-class baseline | accuracy ~0.80, positive recall 0 |
| 1 | Starter notebook pipeline | accuracy 0.79-0.82, positive recall 0.15-0.30 |
| 2 | + better data-quality handling | +1-2pp |
| 3 | + feature scaling | KNN +3-5pp; MLP large jump |
| 4 | + class weights / SMOTE | positive recall >= 0.60; balanced accuracy +>= 8pp; large cost drop under matrix A |
| 5 | + hyperparameter tuning | +2-4pp |
| 6 | + cross-validation | hold-out standard deviation across seeds >= 1.5pp on F1 |
| 7 | + gradient boosting / random forest | AUC +0.03-0.05 over tuned LR |
| 8 | + threshold and cost optimisation | cost down >= 25% under A, >= 10% under B, **and model ranking differs between A and B** |
| 9 | Leakage check | adding `SelfRatedHealth` gives +8-12pp |
| 10 | Train/test gap | resubstitution > CV > test; test 2-4pp below CV |

This script is also the **generation loop tool**: the latent-model coefficients are tuned until
every rung passes. Expect several generate-validate-adjust iterations.

---

## 8. Notebooks

**`cw2_starter_notebook.ipynb`** — what students are handed. Deliberately weak, and honest about
being a starting point rather than a solution:

- Loads the development set; crude quality handling (drop duplicates, drop rows with any missing
  value)
- Encoding to numeric only — **no scaling**
- `LogisticRegression()` and `KNeighborsClassifier()` at defaults, no tuning
- A single hold-out split, one seed
- **Accuracy only**, plus both cost matrices imported from `costs.py` and reported
- Scores the held-out test set at the end, so the gap is visible from day one
- Expected output: ~81% accuracy, ~25% positive-class recall, poor cost under matrix A

**`cw2_reference_notebook.ipynb`** — the reference solution, the marking reference point, and the
"here is what I got, try to beat it" benchmark. It walks the full ladder as a narrative with each
stage showing its gain, and doubles as the fit-for-purpose validation of the dataset: baseline
reproduction, data-quality repair, scaling, imbalance (class weighting against SMOTE, compared),
tuning, validation upgrade showing hold-out variance across seeds, a more expressive model,
per-stakeholder threshold optimisation with cost tables and the rank reversal, the leakage
demonstration, the train/CV/test gap and overfitting discussion, the full three-layer drift
analysis of §6, and a closing summary ladder table.

**`dataset_validation.ipynb`** — the existing instructor-facing data-validation notebook, reworked
for v3. It stays focused on *the data* (distributions, correlations, quality issues, both
targets); the ML story now lives in the reference notebook.

---

## 9. Status

Design agreed. Implementation proceeds in this order, with a commit at each stage on
`feature/coffee-health-v3`:

1. [x] This document
2. [ ] `STAKEHOLDERS_AND_COSTS.md` and `costs.py` — pins the cost numbers down early
3. [ ] `DATA_GENERATION_SPEC.md` with exact coefficients (spec precedes code, per repo process)
4. [ ] `generate_dataset.py` rework and first generation
5. [ ] `validate_ml_ladder.py`, then iterate coefficients until all rungs pass — **this is where
       the real work is**, and it is expected to send us back to the spec more than once
6. [ ] `validate_dataset.py` extension and dev-vs-test shift checks
7. [ ] Notebooks, then documentation
