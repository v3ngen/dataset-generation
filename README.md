# Dataset Generation for ML Teaching

This repository generates synthetic datasets for an ML unit that teaches **Exploratory Data Analysis (EDA)** and then **applied Machine Learning** (scikit-learn). Each dataset is self-contained in its own top-level folder and follows the same four-step process, so a student (or Claude Code, in a future session) can pick up any dataset folder and immediately understand how it was built and why.

The motivating problem: publicly available "messy" datasets (e.g. from Kaggle) are often *randomly* generated — no logical correlations between features, no discoverable patterns. That makes them fine for practicing model-fitting mechanics, but useless for teaching EDA, since there's nothing real to find. Datasets here are synthetic instead, built with realistic, research-grounded relationships between features **and** deliberately injected data quality issues, so students can discover genuine patterns during EDA and then have to handle realistic messiness during the applied ML phase.

## Repository Structure

```
dataset-generation/
├── CLAUDE.md                          # Claude Code guidance (this repo's process, kept high-level)
├── README.md                          # This file
├── online-gaming/                     # Dataset 1 — complete, reference implementation
│   ├── README.md                      # Dataset overview for students
│   ├── DATA_GENERATION_SPEC.md        # Feature spec, relationships, data quality issues
│   ├── DATA_GENERATION_ALGORITHM.md   # Exact generation algorithm/formulas
│   ├── generate_dataset.py            # Generation script
│   ├── validate_dataset.py            # Validation script
│   ├── analysis_demo.ipynb            # EDA + applied ML notebook
│   ├── requirements.txt
│   ├── generated_gaming_dataset.csv   # Output of generate_dataset.py
│   └── data/
│       └── online_gaming_original.csv # Original source dataset (pre-improvement)
└── coffee-health/                     # Dataset 2 — complete (v3)
    ├── README.md                          # Dataset overview
    ├── DATA_GENERATION_SPEC.md            # Feature spec, relationships, data quality issues
    ├── DATA_GENERATION_ALGORITHM.md       # Exact generation algorithm/formulas
    ├── generate_dataset.py                # Generation script
    ├── validate_dataset.py                # Validation script
    ├── analyze_original_dataset.py        # Limitations analysis of the original source dataset
    ├── dataset_validation.ipynb           # Review notebook (instructor-facing, not student material)
    ├── requirements.txt
    ├── generated_coffee_health_dataset.csv # Output of generate_dataset.py
    └── data/
        ├── coffee_health_original.csv
        └── coffee_health_README.docx
```

Every dataset folder is self-contained: its own README, spec, scripts, `requirements.txt`, and `data/` subfolder for any original/source data it starts from. Nothing is shared between dataset folders. There is a single `CLAUDE.md` and `README.md` at the repository root — individual dataset folders get their own `README.md` only.

## Datasets

| Folder | Status | Topic |
|---|---|---|
| [online-gaming/](online-gaming/) | Complete | Online gaming behavior (playtime, engagement, spending) |
| [coffee-health/](coffee-health/) | Complete (v3) | Coffee consumption, lifestyle, and health across Italy/France/UK/Norway. v3 adds a binary ML target, two stakeholder cost matrices, and a held-out test set with distribution shift, to carry a two-assignment EDA + applied-ML unit |

## The Process

Each dataset is built in four steps. `online-gaming/` is the reference implementation — look at its files for the expected shape and depth of each artifact.

### a) Analyze limitations of an existing dataset (optional)

If the dataset is improving on an existing one (rather than being built from scratch), start by identifying *why* it's unsuitable for EDA teaching — e.g. features with no logical correlation to each other, unrealistic distributions, no demographic patterns. This becomes the justification for the redesign and shapes the spec in the next step.

If there's no existing dataset to improve on, skip this step and go straight to spec'ing the new dataset from research/domain knowledge.

### b) Spec the dataset

The spec is written **iteratively, as a dialogue** — not handed over as a one-shot brief. Expect a back-and-forth: you give general direction and domain knowledge, Claude may be asked to pull key stats/analysis on the existing dataset or fetch relevant research/domain details, and features get defined collaboratively from there. The spec document accumulates through that conversation rather than being drafted complete up front.

Because of that, it's normal for a spec (see `online-gaming/DATA_GENERATION_SPEC.md`) to carry unchecked `[ ]` checklist items, an "Open Questions" section, and a running "Notes & Decisions" log even once the dataset is finished and in use — the checklist was a working tool during the dialogue, not a deliverable that needs to end at 100% checked off. Don't treat a spec with open items as incomplete or in need of tidying; only close items when a decision has actually superseded the question.

A spec (once matured through the dialogue) covers:

- **Features**: full list, types, ranges, and where they sit in the dataset's grain (e.g. per-player vs per-player-per-game).
- **Real-world grounding**: for each meaningful relationship, note the research or domain reasoning behind it (e.g. gender/genre distributions, age/spending patterns) so the correlations are defensible, not arbitrary.
- **Feature relationships**: which features should correlate with which, how strongly, and via what generation order/dependency chain — this becomes the basis for a companion algorithm doc (see `online-gaming/DATA_GENERATION_ALGORITHM.md`) if the logic is complex enough to need step-by-step formulas.
- **Target variable(s)**: for ML, define one or more classification/regression targets. Favor multi-factorial targets (derived from several features, non-deterministically) over trivial ones, so prediction requires real feature engineering rather than reading off a single column.
- **Intentional data quality issues**: this is what turns the notebook's EDA/cleaning section into a teaching tool. For each issue, document explicitly:
  - What it is (duplicate rows, missing values, noisy/anomalous values like typo-style errors, etc.)
  - Which columns/rows it affects and at what rate
  - How a student should detect it (e.g. `df.duplicated()`, range checks, `df.isnull().sum()`)
  - How it should be cleaned/handled
  - Any fields that must remain immune (e.g. primary keys, target variables)

  See the "Data Quality Issues" section of `online-gaming/DATA_GENERATION_SPEC.md` for the level of precision expected here — rates and detection/cleaning approach should be specified before writing the generation script, not decided ad hoc while coding.

### c) Generation scripts (Python)

- `generate_dataset.py` — generates the dataset from the spec. Should accept a `--seed` for reproducibility and expose key parameters (row count, output path) as CLI args. Keep distribution parameters in a config object near the top of the file rather than scattered through the logic.
- `validate_dataset.py` — checks the generated output against the spec: value ranges, data types, categorical distributions, target variable distributions, correlation strengths, logical consistency (e.g. derived columns actually match their formula), and data quality issue counts.
- `requirements.txt` — dataset-specific dependencies (numpy, pandas, scipy at minimum).

**This step loops with (b), it doesn't just follow it once.** After the first generation, Claude runs `validate_dataset.py` (and any other checks needed) and should proactively check the output against the spec without being asked — flagging mismatches rather than assuming the spec was hit. Once it looks aligned, that's the signal for you to do a manual check. Mismatches typically send you back to adjusting the spec (or the algorithm/generation logic) and regenerating, so treat b/c as one iterative cycle rather than sequential one-off steps.

### d) Dataset review notebook

A Jupyter notebook — e.g. `dataset_validation.ipynb` (see coffee-health's for the pattern; online-gaming's older `analysis_demo.ipynb` predates this convention) — that:

1. Loads the dataset and does an initial data quality assessment — explicitly surfacing the issues injected in step (b): duplicates, anomalies, missing values.
2. Cleans the data using the detection/cleaning approach documented in the spec.
3. Runs EDA: descriptive statistics, distributions, correlation analysis, categorical breakdowns, and relationship/segmentation analysis that surfaces the patterns built into the spec.
4. Trains ML model(s) with scikit-learn for the target variable(s) defined in the spec, including appropriate handling of categorical encoding, class imbalance, and any data-leakage traps (e.g. features that were used to derive the target must be excluded).
5. Closes with a findings summary confirming the intentional patterns and data quality issues actually show up as designed.

**Default assumption: this notebook is for the user's own review of the generated dataset, not a student-facing teaching artifact.** Don't add exercises, "for students" framing, or other teaching scaffolding unless the user explicitly asks for a student-facing notebook as its own task — that's a distinct deliverable from dataset validation, not an automatic next step, and shouldn't be assumed or built speculatively.

## Adding a new dataset

1. Create a new top-level folder (short, kebab/lower-case name matching the topic).
2. If starting from an existing dataset, drop the source file(s) into `<folder>/data/`.
3. Work through the process above: limitations analysis (optional) → `DATA_GENERATION_SPEC.md` (and `DATA_GENERATION_ALGORITHM.md` if needed) → `generate_dataset.py` + `validate_dataset.py` → review notebook.
4. Write `<folder>/README.md` summarizing the dataset for students, following `online-gaming/README.md`'s structure (overview, feature list, built-in patterns, data quality issues, ML tasks, usage example).
5. Add the dataset to the table in this file.

## Getting Started

Each dataset folder is independent:

```bash
cd <dataset-folder>
pip install -r requirements.txt
python3 generate_dataset.py --rows 10000 --seed 42
python3 validate_dataset.py <output>.csv --eda
jupyter notebook analysis_demo.ipynb
```

See the dataset folder's own `README.md` for exact filenames and options.
