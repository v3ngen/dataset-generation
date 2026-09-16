"""
Validate that the Coffee Health v3 dataset is fit for the CW2 assignment.

The assignment asks students to improve a deliberately weak baseline pipeline.
That only works if each improvement actually pays off -- it is pointless to ask
someone to scale features, handle class imbalance, tune hyperparameters, improve
validation and reach for a better model if none of it changes the numbers.

This script runs the whole ladder with fixed seeds and FAILS if a rung does not
deliver. It is both the fit-for-purpose check and the tuning loop for the
generator's coefficients.

Usage:
    python3 validate_ml_ladder.py [--dev coffee_health_v3_dev.csv]
                                  [--test coffee_health_v3_test.csv]
"""

import argparse
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from costs import COST_EMPLOYER, COST_PUBLIC_HEALTH, cost_per_person

SEED = 42
TARGET = 'HighHealthNeeds'
LEAK_COLUMN = 'SelfRatedHealth'

ORDINAL_SCALES = {
    'Stress Level': ['Low', 'Medium', 'High'],
    'Physical Activity Level': ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active'],
    'Sleep Quality': ['Poor', 'Fair', 'Good', 'Excellent'],
    'Health Issues': ['No Issues', 'Mild', 'Moderate', 'Severe'],
    'Smoking Status': ['Never', 'Former', 'Vaper', 'Light Smoker', 'Heavy Smoker'],
    'Alcohol Level': ['Non-Drinker', 'Light', 'Moderate', 'Heavy'],
    LEAK_COLUMN: ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
}

# --------------------------------------------------------------------------------------
# Results bookkeeping
# --------------------------------------------------------------------------------------

RESULTS = []


def check(rung, description, condition, detail):
    RESULTS.append((rung, description, bool(condition), detail))
    flag = 'PASS' if condition else 'FAIL'
    print(f'  [{flag}] {description}: {detail}')


def section(title):
    print(f'\n{title}\n' + '-' * len(title))


# --------------------------------------------------------------------------------------
# Feature preparation
# --------------------------------------------------------------------------------------

def encode(df, keep_leak=False):
    """Ordinal-encode the ordered categoricals, one-hot the nominal ones."""
    d = df.copy()
    for col, order in ORDINAL_SCALES.items():
        if col == LEAK_COLUMN and not keep_leak:
            continue
        if col in d.columns:
            d[col] = d[col].map({v: i for i, v in enumerate(order)})
    if not keep_leak and LEAK_COLUMN in d.columns:
        d = d.drop(columns=[LEAK_COLUMN])
    d = pd.get_dummies(d, columns=['Country', 'Gender'], drop_first=True)
    y = d.pop(TARGET)
    return d.drop(columns=['ID']).astype(float), y


def align(X, reference):
    return X.reindex(columns=reference.columns, fill_value=0)


def naive_clean(df):
    """What the starter notebook does: drop duplicates, drop any row with a missing
    value, and leave the anomalies alone."""
    return df.drop_duplicates().dropna()


def careful_clean(df, impute_from=None):
    """What a student should work out: repair the anomalies rather than discarding
    the rows, and impute rather than dropping."""
    d = df.drop_duplicates().copy()

    age_typo = d['Age'] > 100
    d.loc[age_typo, 'Age'] = d.loc[age_typo, 'Age'].map(lambda v: int(str(int(v))[:-1]))
    bmi_typo = d['BMI'] > 60
    d.loc[bmi_typo, 'BMI'] = (d.loc[bmi_typo, 'BMI'] / 10).round(1)

    source = d if impute_from is None else impute_from
    for col in d.columns:
        if d[col].isna().any():
            fill = source[col].median() if d[col].dtype != object else source[col].mode()[0]
            d[col] = d[col].fillna(fill)
    return d


def split(X, y, seed=SEED):
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def scores(model, X, y):
    """Every metric here is computed from model.predict() alone.

    No predicted probabilities, no decision thresholds: the coursework is scored on
    the metrics a second-year course already teaches, plus the two stakeholders' cost
    per person.
    """
    pred = model.predict(X)
    return {
        'acc': accuracy_score(y, pred),
        'bal_acc': balanced_accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred),
        'f1': f1_score(y, pred),
        'cost_a': cost_per_person(y, pred, COST_PUBLIC_HEALTH),
        'cost_b': cost_per_person(y, pred, COST_EMPLOYER),
    }


def show(label, s):
    print(f'    {label:<34} acc={s["acc"]:.3f}  prec={s["precision"]:.3f}  '
          f'rec={s["recall"]:.3f}  F1={s["f1"]:.4f}  '
          f'costA={s["cost_a"]:+7.1f}  costB={s["cost_b"]:+7.1f}')


# --------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Validate the CW2 improvement ladder')
    parser.add_argument('--dev', default='coffee_health_v3_dev.csv')
    parser.add_argument('--test', default='coffee_health_v3_test.csv')
    args = parser.parse_args()

    dev_raw = pd.read_csv(args.dev)
    test_raw = pd.read_csv(args.test)

    # The test set is used exactly as students are permitted to use it: encoded and
    # scaled, never cleaned, never resampled.
    X_test, y_test = encode(test_raw)

    # ---------------------------------------------------------------- rung 0
    section('Rung 0 -- majority-class baseline')
    dev_naive = naive_clean(dev_raw)
    Xn, yn = encode(dev_naive)
    Xtr_n, Xva_n, ytr_n, yva_n = split(Xn, yn)
    majority_acc = 1 - yva_n.mean()
    print(f'    positive rate dev={yn.mean():.3f}  test={y_test.mean():.3f}')
    check(0, 'majority-class accuracy near 0.80',
          0.77 <= majority_acc <= 0.83, f'{majority_acc:.3f}')
    check(0, 'positive class is a meaningful minority',
          0.15 <= yn.mean() <= 0.25, f'{yn.mean():.3f}')

    # ---------------------------------------------------------------- rung 1
    section('Rung 1 -- the starter pipeline (defaults, unscaled, accuracy only)')
    lr0 = LogisticRegression().fit(Xtr_n, ytr_n)          # default max_iter=100
    knn0 = KNeighborsClassifier().fit(Xtr_n, ytr_n)       # default k=5
    s_lr0, s_knn0 = scores(lr0, Xva_n, yva_n), scores(knn0, Xva_n, yva_n)
    show('LogisticRegression() unscaled', s_lr0)
    show('KNeighborsClassifier() unscaled', s_knn0)
    starter_best_acc = max(s_lr0['acc'], s_knn0['acc'])
    starter_best_recall = max(s_lr0['recall'], s_knn0['recall'])
    check(1, 'starter accuracy looks respectable but beats baseline by little',
          starter_best_acc <= majority_acc + 0.05,
          f'{starter_best_acc:.3f} vs baseline {majority_acc:.3f}')
    check(1, 'starter misses most positives (the teaching moment)',
          starter_best_recall <= 0.45, f'best recall {starter_best_recall:.3f}')

    # ---------------------------------------------------------------- rung 2
    section('Rung 2 -- better data-quality handling')
    dev_clean = careful_clean(dev_raw)
    Xc, yc = encode(dev_clean)
    Xtr, Xva, ytr, yva = split(Xc, yc)
    print(f'    rows kept: naive {len(dev_naive)}  careful {len(dev_clean)}')
    lr_clean = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    s_clean = scores(lr_clean, Xva, yva)
    show('LR on repaired + imputed data', s_clean)
    check(2, 'careful cleaning retains many more rows',
          len(dev_clean) > len(dev_naive) * 1.15,
          f'{len(dev_clean)} vs {len(dev_naive)} (+{len(dev_clean)/len(dev_naive)-1:.0%})')

    # ---------------------------------------------------------------- rung 3
    section('Rung 3 -- feature scaling')
    scaler = StandardScaler().fit(Xtr)
    Str, Sva = scaler.transform(Xtr), scaler.transform(Xva)
    knn_unscaled = KNeighborsClassifier().fit(Xtr, ytr)
    knn_scaled = KNeighborsClassifier().fit(Str, ytr)
    s_ku = scores(knn_unscaled, Xva, yva)
    s_ks = scores(knn_scaled, Sva, yva)
    show('KNN k=5 unscaled', s_ku)
    show('KNN k=5 scaled', s_ks)
    check(3, 'scaling lifts KNN F1 substantially',
          s_ks['f1'] - s_ku['f1'] >= 0.15,
          f'{s_ku["f1"]:.4f} -> {s_ks["f1"]:.4f}')
    check(3, 'unscaled KNN finds almost nobody (income dominates the distance)',
          s_ku['recall'] <= 0.15, f'recall {s_ku["recall"]:.3f}')

    # ---------------------------------------------------------------- rung 4
    section('Rung 4 -- class imbalance')
    lr_plain = LogisticRegression(max_iter=2000).fit(Str, ytr)
    lr_bal = LogisticRegression(max_iter=2000, class_weight='balanced').fit(Str, ytr)
    s_plain, s_bal = scores(lr_plain, Sva, yva), scores(lr_bal, Sva, yva)
    show('LR scaled', s_plain)
    show('LR scaled + class_weight', s_bal)
    check(4, 'class weighting transforms positive-class recall',
          s_bal['recall'] >= 0.60 and s_bal['recall'] - s_plain['recall'] >= 0.25,
          f'{s_plain["recall"]:.3f} -> {s_bal["recall"]:.3f}')
    check(4, 'class weighting improves balanced accuracy',
          s_bal['bal_acc'] - s_plain['bal_acc'] >= 0.06,
          f'+{s_bal["bal_acc"] - s_plain["bal_acc"]:.3f}')
    check(4, 'class weighting cuts cost for the public health agency',
          s_bal['cost_a'] < s_plain['cost_a'] - 15,
          f'{s_plain["cost_a"]:+.1f} -> {s_bal["cost_a"]:+.1f} per person')

    # ---------------------------------------------------------------- rung 5
    section('Rung 5 -- hyperparameter tuning')
    # KNN and logistic regression barely respond to tuning here, so the demonstration
    # uses a random forest, where the defaults are genuinely bad: an unconstrained
    # forest overfits this data badly.
    rf_default = RandomForestClassifier(class_weight='balanced', random_state=SEED,
                                        n_jobs=-1).fit(Xtr, ytr)
    s_rf_default = scores(rf_default, Xva, yva)
    show('RF balanced, all defaults', s_rf_default)

    rf_search = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=SEED, n_jobs=-1),
        {'max_depth': [6, 10, 14, None], 'min_samples_leaf': [1, 5, 20, 50]},
        scoring='f1', cv=5, n_jobs=-1).fit(Xtr, ytr)
    rf_tuned = rf_search.best_estimator_
    s_rf_tuned = scores(rf_tuned, Xva, yva)
    show(f'RF balanced, tuned {rf_search.best_params_}', s_rf_tuned)

    check(5, 'tuning lifts F1 substantially over the defaults',
          s_rf_tuned['f1'] - s_rf_default['f1'] >= 0.05,
          f'{s_rf_default["f1"]:.4f} -> {s_rf_tuned["f1"]:.4f}')
    check(5, 'tuning also cuts cost for the public health agency',
          s_rf_tuned['cost_a'] < s_rf_default['cost_a'] - 25,
          f'{s_rf_default["cost_a"]:+.1f} -> {s_rf_tuned["cost_a"]:+.1f} per person')

    # Tuning is NOT uniformly valuable, and the contrast is worth having: logistic
    # regression's regularisation strength barely moves anything here. Knowing which
    # knobs matter for which model family is the actual skill.
    lr_search = GridSearchCV(
        LogisticRegression(max_iter=4000, class_weight='balanced'),
        {'C': [0.01, 0.1, 1.0, 10.0]}, scoring='f1', cv=5, n_jobs=-1).fit(Str, ytr)
    s_lr_tuned = scores(lr_search.best_estimator_, Sva, yva)
    show(f'LR balanced, tuned C={lr_search.best_params_["C"]}', s_lr_tuned)
    print(f'    (tuning LR moved F1 by {s_lr_tuned["f1"] - s_bal["f1"]:+.4f} -- '
          f'not every model responds)')

    # ---------------------------------------------------------------- rung 6
    section('Rung 6 -- validation method')
    # Raw variance is only half the story. What actually misleads a student is when
    # hold-out noise is large enough to REVERSE a comparison between two models, so
    # that is what we assert.
    contenders = {
        'LR balanced': lambda: LogisticRegression(max_iter=2000, class_weight='balanced'),
        'KNN k=25': lambda: KNeighborsClassifier(n_neighbors=25),
        'RF small': lambda: RandomForestClassifier(n_estimators=120, max_depth=8,
                                                   class_weight='balanced',
                                                   random_state=SEED, n_jobs=-1),
        'RF deep': lambda: RandomForestClassifier(n_estimators=200, max_depth=14,
                                                  min_samples_leaf=3, class_weight='balanced',
                                                  random_state=SEED, n_jobs=-1),
    }
    per_seed = {name: [] for name in contenders}
    for seed in range(10):
        a, b, c, d = split(Xc, yc, seed=seed)
        sc = StandardScaler().fit(a)
        for name, make in contenders.items():
            m = make().fit(sc.transform(a), c)
            per_seed[name].append(f1_score(d, m.predict(sc.transform(b))))

    for name, vals in per_seed.items():
        print(f'    {name:<14} F1 over 10 seeds: mean {np.mean(vals):.4f}  sd {np.std(vals):.4f}  '
              f'range {min(vals):.3f}-{max(vals):.3f}')
    # A student comparing several models on one split can easily be misled about any
    # pair of them. Find the pair that actually reverses.
    names = list(contenders)
    worst_pair, worst_flips = None, 0
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            wins_a = sum(1 for x, y in zip(per_seed[a_name], per_seed[b_name]) if x > y)
            flips = min(wins_a, 10 - wins_a)
            if flips > worst_flips:
                worst_pair, worst_flips = (a_name, b_name, wins_a), flips
    sd = max(float(np.std(v)) for v in per_seed.values())
    if worst_pair:
        a_name, b_name, wins_a = worst_pair
        print(f'    most fragile comparison: {a_name} beats {b_name} on {wins_a}/10 splits '
              f'(means {np.mean(per_seed[a_name]):.4f} vs {np.mean(per_seed[b_name]):.4f})')
    check(6, 'hold-out noise is enough to reverse a model comparison',
          worst_flips >= 1, f'a pair of models swaps rank on {worst_flips}/10 splits')
    check(6, 'single-split scores vary enough to mislead',
          sd >= 0.007, f'sd {sd:.4f} on positive-class F1')

    # ---------------------------------------------------------------- rung 7
    section('Rung 7 -- a more expressive model')
    # Compared LIKE FOR LIKE: both class-weighted, both tuned. Comparing an
    # unweighted tree against a weighted linear model would confound model family
    # with imbalance handling and prove nothing.
    weights = np.where(ytr == 1, (ytr == 0).sum() / (ytr == 1).sum(), 1.0)
    gbm_search = GridSearchCV(
        HistGradientBoostingClassifier(random_state=SEED),
        {'max_leaf_nodes': [15, 31, 63], 'learning_rate': [0.05, 0.1],
         'min_samples_leaf': [20, 50]},
        scoring='f1', cv=5, n_jobs=-1).fit(Xtr, ytr, sample_weight=weights)
    gbm = gbm_search.best_estimator_
    s_gbm = scores(gbm, Xva, yva)
    show('LR balanced, tuned (linear)', s_lr_tuned)
    show('GBM balanced, tuned (trees)', s_gbm)

    gap = s_gbm['f1'] - s_lr_tuned['f1']
    check(7, 'a tree ensemble beats logistic regression (non-additive structure)',
          gap >= 0.03, f'+{gap:.4f} F1 over tuned class-weighted LR')
    check(7, 'the gain is not so large that logistic regression looks pointless',
          gap <= 0.15, f'+{gap:.4f} F1')

    # ---------------------------------------------------------------- rung 8
    section('Rung 8 -- cost, and the two stakeholders')
    # The headline finding, and it needs no thresholds: run several plausible models
    # with plain .predict(), price the confusion matrix under each stakeholder's cost
    # matrix, and the two stakeholders pick different models.
    candidates = {
        'LR balanced': (lr_bal, Sva),
        'GBM balanced tuned': (gbm, Xva),
        'RF balanced tuned': (rf_tuned, Xva),
        'LR unweighted': (lr_plain, Sva),
    }
    table = {}
    for name, (model, features) in candidates.items():
        s = scores(model, features, yva)
        table[name] = s
        print(f'    {name:<22} acc={s["acc"]:.3f}  prec={s["precision"]:.3f}  '
              f'rec={s["recall"]:.3f}  F1={s["f1"]:.4f}  '
              f'agency={s["cost_a"]:+8.2f}  employer={s["cost_b"]:+8.2f}')

    best_accuracy = max(table, key=lambda n: table[n]['acc'])
    best_agency = min(table, key=lambda n: table[n]['cost_a'])
    best_employer = min(table, key=lambda n: table[n]['cost_b'])
    print(f'    best by accuracy: {best_accuracy}')
    print(f'    cheapest for the agency: {best_agency}')
    print(f'    cheapest for the employer: {best_employer}')

    # A model that is better on BOTH precision and recall wins for both stakeholders,
    # and that is a legitimate outcome -- the interesting case is a pair on the
    # precision/recall frontier, where neither dominates and the two stakeholders
    # genuinely disagree. Find such a pair; students comparing their own models will
    # hit these constantly.
    reversals = []
    names = list(table)
    for i, one in enumerate(names):
        for other in names[i + 1:]:
            agency_prefers = one if table[one]['cost_a'] < table[other]['cost_a'] else other
            employer_prefers = one if table[one]['cost_b'] < table[other]['cost_b'] else other
            if agency_prefers != employer_prefers:
                reversals.append((one, other, agency_prefers, employer_prefers))
    for one, other, ap, ep in reversals:
        print(f'    DISAGREEMENT  {one} vs {other}:  agency -> {ap},  employer -> {ep}')

    check(8, 'at least one pair of plausible models splits the two stakeholders',
          len(reversals) >= 1, f'{len(reversals)} reversing pair(s) found')
    check(8, 'the most accurate model is not what either stakeholder wants',
          best_accuracy != best_agency and best_accuracy != best_employer,
          f'most accurate is {best_accuracy}, agency wants {best_agency}, '
          f'employer wants {best_employer}')
    check(8, "the agency's preferred model turns a cost into a net saving",
          table[best_agency]['cost_a'] < 0,
          f'{table[best_agency]["cost_a"]:+.2f} per person')

    # ---------------------------------------------------------------- rung 9
    section('Rung 9 -- the leakage trap')
    Xl, yl = encode(dev_clean, keep_leak=True)
    Xltr, Xlva, yltr, ylva = split(Xl, yl)
    gbm_leak = HistGradientBoostingClassifier(random_state=SEED).fit(Xltr, yltr)
    s_leak = scores(gbm_leak, Xlva, ylva)
    show('GBM including SelfRatedHealth', s_leak)
    leak_gain = s_leak['f1'] - s_gbm['f1']
    check(9, 'SelfRatedHealth gives a seductive, detectable gain',
          leak_gain >= 0.02,
          f'+{leak_gain:.4f} F1, {s_gbm["acc"]:.3f} -> {s_leak["acc"]:.3f} accuracy')
    check(9, 'the leak is not so total that the exercise is trivial',
          leak_gain <= 0.20, f'+{leak_gain:.4f} F1')

    # ---------------------------------------------------------------- rung 10
    section('Rung 10 -- train vs cross-validation vs held-out test')
    gbm_full = HistGradientBoostingClassifier(random_state=SEED).fit(Xc, yc)
    resub = accuracy_score(yc, gbm_full.predict(Xc))
    cv = cross_val_score(HistGradientBoostingClassifier(random_state=SEED), Xc, yc,
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                         scoring='accuracy').mean()
    Xt = align(X_test, Xc)
    test_acc = accuracy_score(y_test, gbm_full.predict(Xt))
    test_f1 = f1_score(y_test, gbm_full.predict(Xt))
    print(f'    resubstitution (train)  acc={resub:.4f}')
    print(f'    5-fold cross-validation acc={cv:.4f}')
    print(f'    held-out test           acc={test_acc:.4f}  F1={test_f1:.4f}')
    check(10, 'training-set performance overstates reality (overfitting is visible)',
          resub - cv >= 0.03, f'resubstitution exceeds CV by {resub - cv:.3f}')
    check(10, 'the test set is harder than cross-validation suggests (drift is visible)',
          cv - test_acc >= 0.015, f'CV exceeds test by {cv - test_acc:.3f}')
    check(10, 'the drop is not so large that the test set looks broken',
          cv - test_acc <= 0.08, f'{cv - test_acc:.3f}')

    # ---------------------------------------------------------------- summary
    section('Summary')
    failed = [r for r in RESULTS if not r[2]]
    for rung in sorted({r[0] for r in RESULTS}):
        rung_results = [r for r in RESULTS if r[0] == rung]
        bad = sum(1 for r in rung_results if not r[2])
        print(f'  Rung {rung:>2}: {len(rung_results) - bad}/{len(rung_results)} checks passed'
              + ('' if not bad else '   <-- FAILED'))
    print(f'\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed.')
    if failed:
        print('\nFailed checks:')
        for rung, desc, _, detail in failed:
            print(f'  rung {rung}: {desc} ({detail})')
        return 1
    print('The dataset supports every rung of the improvement ladder.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
