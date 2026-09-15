# CW1 (EDA) — Discovery Ladder and Marking Rubric

**Instructor-facing.** Contains the answers. The student-facing brief should carry the strand
descriptions and weightings but not §2.

**Scope.** CW1 uses `coffee_health_v3_dev.csv` only. The held-out test set is not released until
CW2, so nothing about distribution shift belongs in this assignment.

Everything below is verified against the generated data — see `dataset_validation.ipynb` for the
worked version and `coffee_health_v3_README.md` for the figures.

---

## 1. Strands and weighting

| Strand | Weight | What it assesses |
|---|---|---|
| **A. Dataset overview** | 20% | Can they describe what they have, accurately and in terms of the problem? |
| **B. Data quality analysis** | 35% | Can they find the problems, fix them defensibly, and reason about consequences? |
| **C. Insights** | 45% | Can they get past the correlation matrix to something the data is actually saying? |

Visualisation and written clarity are **not** a separate strand. A finding that is not communicated
is not a finding, so presentation quality is marked inside whichever strand it belongs to. Adding a
fourth strand for it tends to reward decoration.

The weighting puts nearly half the marks on strand C deliberately: A and B have a ceiling (there
are only so many columns and so many defects), whereas C has genuine range.

---

## 2. The discovery ladder

This is the spine of the rubric — what is actually findable, ordered by how hard it is to find.
Each tier assumes the ones below it.

### Tier 1 — Basics (threshold, 40–49)

| Strand | Finding |
|---|---|
| A | 10,040 rows, 18 columns. Data types. `describe()` for numerics, value counts for categoricals. Identifies `SelfRatedHealth` and `HighHealthBurden` as the two targets |
| B | Counts the 40 duplicate rows. Tabulates missing values per column (Health Issues 10.4%, Sleep Hours 7.5%, Resting HR 6.0%, Stress 5.3%). Notices that some `Age` and `BMI` values are impossible |
| C | Univariate distributions. A correlation matrix. Names the strongest correlates of self-rated health (Health Issues −0.47, Sleep Quality +0.38, Activity +0.33, Stress −0.31). One demographic breakdown |

### Tier 2 — Sound / Good (50–69)

| Strand | Finding |
|---|---|
| A | Distinguishes nominal, ordinal and continuous, and **respects the natural order** of the ordered categoricals in every table and chart. Notes `ID` is an identifier, not a feature. Notes the class imbalance in `HighHealthBurden` (19.8% positive) |
| B | Removes the duplicates and justifies it. Recognises the anomalies are **exactly recoverable** — `344 → 34` by dropping the trailing digit, `245 → 24.5` by dividing by ten — and repairs rather than discards. Quantifies what `dropna()` would cost: **26% of all rows** |
| C | Bivariate analysis against both targets. Country comparison with real profile differences (Norway 83.2% "good or better" down to UK 69.6%; UK has the highest BMI at 27.9 and the lowest coffee intake at 1.8 cups). Smoking dose-response. Activity and sleep-quality gradients |

### Tier 3 — Advanced (60–79)

| Strand | Finding |
|---|---|
| A | Describes **measurement provenance** — which fields come from a wearable, which from a survey — and draws the implication for how far each can be trusted. Articulates what one row represents and what the targets actually measure |
| B | Discovers that **missingness is not random**: it rises with age across the device-sourced fields (roughly 3.6% to 16.2% from the youngest band to the oldest). Names the MCAR/MAR distinction and uses it correctly |
| C | **Bins or stratifies continuous features against the target**, and thereby finds the two curves that correlation cannot see: the **coffee J-curve** (~30% high-burden among near-abstainers, ~17% at 2–3 cups, ~24% above 4) and the **sleep U-curve** (75% below 4.5 hours, 9% at 6.5–7.5, 43% above 8.5). States explicitly that coffee's near-zero linear correlation is *not* evidence of no relationship. Clustering with interpreted segments |

### Tier 4 — Excellent (70+)

| Strand | Finding |
|---|---|
| A | Frames the dataset in terms of the decision it supports. Handles **both** targets without conflating them — notes they are related (high-burden rate falls 65.7% → 40.0% → 16.0% → 5.1% → 1.6% across the self-rated health levels) but that neither is a relabelling of the other |
| B | Runs the **negative control**: missingness is flat by gender and country, and flat by age for `Health Issues`, so the age effect is specific to device-sourced fields and has a mechanism rather than being a coincidence. Quantifies the resulting bias — rows dropped by `dropna()` have mean age **42.9** against **39.9** for rows kept — and reasons about who a downstream model would then underserve |
| C | At least two of: <br>• the **confound** — `Household Income` ↔ `Caffeine Intake` is +0.24 overall but ≈0 *within every country*, because Norway has both the highest incomes and the strongest coffee <br>• **mediation** — income correlates only +0.05 with health directly, but the sedentary share falls 28.1% → 17.8% across income quintiles, so the gradient runs through behaviour rather than being absent <br>• **interaction** — smoking and overweight compound: 45.3% observed against 30.0% if the two effects were additive; and activity × age crosses over, with a very active over-55 at lower risk than a very active under-55 <br>• **ecological inference** — the UK has high BMI and low coffee consumption, and saying so does not license "coffee protects against obesity" |

### What 85+ looks like

Not more findings — better handling of the ones they have.

- **Causal discipline throughout.** Treats every association as an association, names plausible
  confounders unprompted, and distinguishes what the data can and cannot establish.
- **Honest about clustering.** k-means produces interpretable segments (burden rates from 7% to
  38%), but the silhouette score is ~0.10 at every sensible *k*. A top answer says the clusters are
  a **useful description of a continuum, not evidence of natural kinds**, and does not present four
  "population types" as though they were discovered facts.
- **Negative results reported.** Says which expected relationships did *not* appear, and does not
  quietly drop the analyses that found nothing.
- **Findings connected to the problem.** The gender result is a good test: women report worse health
  than men (mean 2.09 vs 2.18) with near-identical rates of diagnosed issues (43.3% vs 43.0%) *and*
  a genuinely higher high-burden rate (20.5% vs 18.9%). A strong answer notices that the reporting
  gap and the outcome gap point the same way but are not the same size, and is careful about which
  one self-rated health is measuring.

---

## 3. Rubric grids

### Strand A — Dataset overview (20%)

| Band | Descriptor |
|---|---|
| **70+ Excellent** | Describes the dataset in terms of the problem it serves, not just its shape. Measurement provenance is discussed and its implications drawn. Both targets are handled distinctly and their relationship characterised. Every table and chart is correctly typed and ordered. Nothing is present merely because it was easy to produce |
| **60–69 Good** | Accurate and complete description. Variable types correctly distinguished, ordered categoricals ordered correctly throughout. Targets identified, imbalance noted. Some connection to the problem domain |
| **50–59 Sound** | Accurate description of size, types and distributions. Targets identified. Ordering of categoricals mostly respected. Largely mechanical — reports what the data *is* with little sense of what it is *for* |
| **40–49 Threshold** | Shape, types and basic summary statistics present and broadly correct. Targets named. Some misclassification of variable types or arbitrary ordering of ordered categories |
| **<40** | Description absent, substantially inaccurate, or a raw tool dump with no interpretation |

### Strand B — Data quality analysis (35%)

| Band | Descriptor |
|---|---|
| **70+ Excellent** | All three defect types found, quantified and handled with justified choices. Missingness shown to be age-dependent **and** verified against a negative control. The bias introduced by naive row-dropping is quantified and its consequences for a downstream model argued. Anomalies repaired by reasoning, not by deletion or by a magic constant |
| **60–69 Good** | All three defect types found and quantified. Anomalies recognised as recoverable and repaired. Missingness investigated beyond a column count and the age pattern identified. Handling choices are stated and defended |
| **50–59 Sound** | All three defect types found. Duplicates removed. Anomalies identified, though possibly dropped rather than repaired. Missing values handled by a stated method. Little investigation of *why* data is missing |
| **40–49 Threshold** | Duplicates and missing values counted. Some awareness that impossible `Age`/`BMI` values exist. Handling is present but unjustified — typically `dropna()` with no comment on its cost |
| **<40** | Defects missed, or "handled" silently. Impossible values left in and treated as real in later analysis |

### Strand C — Insights (45%)

| Band | Descriptor |
|---|---|
| **70+ Excellent** | Goes beyond the correlation matrix to structure it cannot show: at least one confound, mediation, interaction or non-linearity properly demonstrated and explained. Findings are connected to the real-world problem. Causal language is controlled. Analysis is driven by questions, and negative results are reported |
| **60–69 Good** | Systematic bivariate and segmented analysis. At least one non-linear relationship found through binning or stratification. Country and demographic comparisons interpreted, not just displayed. Clustering, if attempted, is interpreted. Charts support the argument |
| **50–59 Sound** | Correlation analysis plus demographic and country breakdowns, correctly executed and described. Largely one-variable-at-a-time; conclusions restate the numbers rather than explaining them. Some findings stated more strongly than the evidence supports |
| **40–49 Threshold** | A correlation matrix and some distribution plots, with the strongest relationships named. Descriptive rather than analytical. Little segmentation |
| **<40** | Charts without interpretation, conclusions unsupported by the analysis shown, or claims contradicted by the data |

---

## 4. Marker's quick checklist

Tick what is *demonstrated*, not what is mentioned.

**Strand A** — [ ] shape/types correct · [ ] ordered categoricals ordered · [ ] both targets identified ·
[ ] imbalance noted · [ ] `ID` excluded as a feature · [ ] provenance discussed · [ ] framed against the problem

**Strand B** — [ ] 40 duplicates found · [ ] missing values quantified per column · [ ] anomalies found ·
[ ] anomalies **repaired** not dropped · [ ] cost of `dropna()` quantified · [ ] missingness–age pattern found ·
[ ] negative control run · [ ] bias consequences argued

**Strand C** — [ ] correlation analysis · [ ] country comparison · [ ] smoking dose-response ·
[ ] **binning reveals the coffee J-curve** · [ ] **binning reveals the sleep U-curve** ·
[ ] clustering interpreted · [ ] income–caffeine confound · [ ] income mediation ·
[ ] an interaction demonstrated · [ ] causal caution · [ ] negative results reported

---

## 5. Notes on applying this

**Mark the interpretation, not the output.** Producing a correlation heatmap, a pairplot and twenty
histograms is a Tier 1 activity and is now nearly free to generate. Explaining *why the heatmap
misleads about coffee* is the thing being assessed. Where two submissions contain similar analysis,
the marks should separate on what is said about it.

**Volume is not evidence of depth.** A submission with six well-chosen charts and a clear argument
should outscore one with thirty charts and a caption each. Say so in the brief, or the incentive
runs the other way.

**Two conclusions that look reasonable and are wrong.** Both are worth watching for, because both
are what a student gets by stopping one step early:

- *"Coffee intake is unrelated to health outcomes"* — true of the linear correlation (−0.01), false
  of the data. The J-curve is one `pd.cut` away.
- *"Household income doesn't affect health"* — true of the direct correlation (+0.05), false of the
  data. The gradient runs through activity, smoking and BMI.

Neither is a careless error, which is what makes them good discriminators: a student who reports
either has done competent work and drawn the wrong conclusion from it. Tier 2 or low Tier 3, not a
fail — but they cannot reach Tier 4 on strand C.

**Reward the negative control.** Showing that missingness is flat by gender and country is what
turns "missingness correlates with age" from an observation into an argument. Very few will do it,
and it is the cleanest single discriminator in strand B.

**The test set is not part of CW1.** Any analysis of distribution shift belongs to CW2 and should
not attract credit here.
