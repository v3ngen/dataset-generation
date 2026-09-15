# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

This repository generates multiple coherent, synthetic datasets for educational purposes — each one built for teaching **Exploratory Data Analysis (EDA)** and then applied **Machine Learning (ML)** (scikit-learn) in an ML unit.

Each dataset lives in its own top-level folder (e.g. [online-gaming/](online-gaming/), [coffee-health/](coffee-health/)). See [README.md](README.md) for the full repository structure and the step-by-step process used to build a new dataset — read it before starting work on a new or existing dataset folder.

**Why synthetic datasets**: real Kaggle-style datasets used as a starting point are often generated with no logical correlations or patterns between features (fine for applied ML, unsuitable for teaching EDA). The fix is a synthetic dataset with similar features but realistic, logical relationships and patterns students can discover through analysis — while remaining suitable for applied ML.

## Established Process (per dataset)

Follow this sequence when building or extending a dataset folder — see [README.md](README.md) for full detail:

1. **Limitations analysis** (optional — skip if the dataset is being created from scratch rather than improving an existing one): analyze the source/original dataset and document why it's unsuitable for EDA teaching as-is.
2. **Spec**: built iteratively through dialogue with the user, not written solo — expect requests to pull stats on the existing dataset or fetch domain research mid-conversation. Document target features, their real-world/research grounding, feature relationships/correlations to build in, and the intentional data quality issues (missing values, noise/anomalies, duplicates) with precise rates, affected columns, and detection/cleaning approach. Open checklist items / open questions in the spec doc are a normal byproduct of this process, not a defect — don't tidy them away just to close them out.
3. **Generation scripts**: Python (`generate_dataset.py`, plus a `validate_dataset.py` that checks ranges, distributions, correlations and quality-issue counts against the spec), seeded for reproducibility. Loops with step 2: after generating, proactively validate against the spec yourself before handing off — the user does a manual check once it looks aligned, and mismatches typically send you back to adjusting the spec/algorithm and regenerating.
4. **Notebook**: a Jupyter notebook covering EDA (data quality assessment/cleaning, distributions, correlations, segmentation) and applied ML with scikit-learn (classification tasks, feature engineering, handling the injected data quality issues and any class imbalance). **This defaults to a review/validation notebook for the user, not student-facing teaching material** — no exercises or "for students" framing unless explicitly asked for as its own separate task. Don't assume a student-facing notebook is a pending deliverable just because a validation notebook exists.

[online-gaming/](online-gaming/) is the reference implementation of this process — its README.md, `DATA_GENERATION_SPEC.md`, `DATA_GENERATION_ALGORITHM.md`, `generate_dataset.py`, `validate_dataset.py`, and `analysis_demo.ipynb` show the expected shape and depth for each artifact in a new dataset folder.

## Repository Conventions

- One root `CLAUDE.md` (this file) and one root `README.md` — dataset folders do **not** get their own `CLAUDE.md`, only their own more specific `README.md`.
- Each dataset folder keeps its own `data/` subfolder for any original/source data it starts from, its own `requirements.txt`, and its own generation/validation scripts and notebook — folders are self-contained and don't share code.

## Datasets

### online-gaming/ (complete — reference implementation)

Synthetic online gaming behavior dataset, ~10,000 player-game combination rows across 21 features, improving on the original Kaggle ["Predict Online Gaming Behavior Dataset"](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset/data) which lacked logical correlations.

- Original dataset: `online-gaming/data/online_gaming_original.csv` (~40,000 records: PlayerID, Age, Gender, Location, GameGenre, PlayTimeHours, InGamePurchases, GameDifficulty, SessionsPerWeek, AvgSessionDurationMinutes, PlayerLevel, AchievementsUnlocked, EngagementLevel)
- Generated dataset: `online-gaming/generated_gaming_dataset.csv` — player-level attributes (PlayerID, Age, Gender, Location), game-specific attributes (GameID, GameName, GameGenre, GameDifficulty), behavioral & spending metrics, and two ML target variables (SpendingPropensity: NonSpender/Occasional/Whale; PlayerExpertise: Beginner/Intermediate/Expert)
- Design goals achieved: logical correlations (e.g. PlayTimeHours ↔ PlayerLevel), demographic influences on genre/spending preferences, multi-factorial (non-deterministic) target variables, and intentional data quality issues (duplicates, age-typo anomalies, missing values) for teaching data cleaning
- Full detail: `online-gaming/README.md`, `online-gaming/DATA_GENERATION_SPEC.md`, `online-gaming/DATA_GENERATION_ALGORITHM.md`

### coffee-health/ (v3 complete — coursework redesign)

Synthetic coffee consumption / lifestyle / health dataset across Italy, France, UK and Norway,
improving on a modified Kaggle "Global Coffee Health Dataset" whose features had no meaningful
correlations (e.g. Sleep Quality was purely a function of Sleep Hours, Stress Level a 100%
deterministic relabeling of Sleep Quality).

- Original dataset: `coffee-health/data/coffee_health_original.csv`; limitations quantified in `coffee-health/analyze_original_dataset.py`. v2 archived at `coffee-health/data/coffee_health_v2.csv`.
- Full detail: `coffee-health/README.md`, `V3_COURSEWORK_DESIGN.md`, `STAKEHOLDERS_AND_COSTS.md`, `DATA_GENERATION_SPEC.md`, `DATA_GENERATION_ALGORITHM.md`

**v1/v2 (complete, approved).** Single 5-class target `SelfRatedHealth`, 10,040 rows, built as a
multi-factor composite calibrated against sourced country-level statistics, plus
`dataset_validation.ipynb` (the step-d instructor review notebook).

**v3 (complete) — repurposed to carry a two-assignment unit**: CW1 on EDA, CW2 (60%) on applied
ML. The driving problems were that CW2 had become coding-heavy and AI-homogenised, that CW1 and
CW2 were weakly linked, and that students did not connect results to the real world. The response
moves marks onto evaluation and critical analysis rather than pipeline code.

What v3 adds:

- **`HighHealthBurden`** — binary ML target (~20% positive), drawn as a Bernoulli from a sigmoid. Its latent model carries genuine non-additive structure (U-shaped sleep, J-shaped coffee, continuous products, broad categorical interactions) *because v2's purely additive composite put logistic regression near Bayes-optimal, so "use a better model" would have gained nothing*.
- **A latent frailty term**, shared by both targets and never written to the CSV. This is what makes `SelfRatedHealth` a genuine leakage trap rather than a redundant function of columns already present.
- **`Household Income`** — large in magnitude, weak in direct signal, so feature scaling has something real to fix.
- **Two stakeholder cost matrices** with opposing trade-offs (`costs.py`, `STAKEHOLDERS_AND_COSTS.md`), so model ranking legitimately reverses and "which model is best?" requires "best for whom?".
- **Two files**: `coffee_health_v3_dev.csv` (messy) and `coffee_health_v3_test.csv` (**no missing values**; carries anomalies and duplicates for analysis only, never repair). Covariate shift only — `P(y|x_clean)` is identical, so the apparent label shift is compositional.
- **`cw2_starter_notebook.ipynb`** (deliberately weak; the student starting point) and **`cw2_reference_notebook.ipynb`** (instructor worked solution, validation, and the "beat this" benchmark).
- **`validate_ml_ladder.py`** — 23 assertions that each rung of the improvement ladder pays off. This is the guard: if the generator is ever retuned, run it, or the assignment quietly stops working.

Two failure modes are documented in `DATA_GENERATION_ALGORITHM.md` and worth knowing before
touching the generator: quadratic terms must be centred on the **observed** distribution (an
off-centre quadratic is near-linear across the support and a linear model gets it for free), and
the test cohort must **reuse the development set's quantile thresholds** (recomputing them
re-normalises the ordinal composites and silently erases the shift).

Note that the earlier "no student-facing notebook is planned" position applied to v1/v2 and has
been superseded: the CW2 starter notebook was explicitly requested for v3. The general default
still stands — review notebooks are instructor-facing unless a student-facing artifact is asked
for as its own task.
