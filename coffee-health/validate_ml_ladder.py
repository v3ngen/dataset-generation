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
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from costs import COST_EMPLOYER, COST_PUBLIC_HEALTH, cost_per_person, optimal_threshold

SEED = 42
TARGET = 'HighHealthBurden'
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


def scores(model, X, y, proba=True):
    pred = model.predict(X)
    out = {
        'acc': accuracy_score(y, pred),
        'bal_acc': balanced_accuracy_score(y, pred),
        'recall': recall_score(y, pred),
        'f1': f1_score(y, pred),
        'cost_a': cost_per_person(y, pred, COST_PUBLIC_HEALTH),
        'cost_b': cost_per_person(y, pred, COST_EMPLOYER),
    }
    out['auc'] = roc_auc_score(y, model.predict_proba(X)[:, 1]) if proba else np.nan
    return out


def show(label, s):
    print(f'    {label:<34} acc={s["acc"]:.3f}  rec+={s["recall"]:.3f}  '
          f'auc={s["auc"]:.4f}  costA={s["cost_a"]:+7.1f}  costB={s["cost_b"]:+7.1f}')


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
    check(3, 'scaling lifts KNN AUC substantially',
          s_ks['auc'] - s_ku['auc'] >= 0.10,
          f'+{s_ks["auc"] - s_ku["auc"]:.4f} AUC')
    check(3, 'unscaled KNN is close to useless (income dominates the distance)',
          s_ku['auc'] <= 0.62, f'{s_ku["auc"]:.4f} AUC')

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
    knn_grid = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [5, 15, 25, 45, 75]},
                            scoring='roc_auc', cv=3, n_jobs=-1).fit(Str, ytr)
    s_knn_tuned = scores(knn_grid.best_estimator_, Sva, yva)
    show(f'KNN tuned (k={knn_grid.best_params_["n_neighbors"]})', s_knn_tuned)
    check(5, 'tuning k lifts KNN AUC clearly',
          s_knn_tuned['auc'] - s_ks['auc'] >= 0.03,
          f'+{s_knn_tuned["auc"] - s_ks["auc"]:.4f} AUC over default k=5')

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
    gbm = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
    s_gbm, s_rf = scores(gbm, Xva, yva), scores(rf, Xva, yva)
    show('HistGradientBoosting', s_gbm)
    show('RandomForest', s_rf)
    gap = s_gbm['auc'] - s_plain['auc']
    check(7, 'a tree ensemble beats logistic regression (non-additive structure)',
          gap >= 0.03, f'+{gap:.4f} AUC over tuned-scale LR')
    check(7, 'the gain is not so large that LR looks pointless',
          gap <= 0.10, f'+{gap:.4f} AUC')

    # ---------------------------------------------------------------- rung 8
    section('Rung 8 -- decision threshold and the two stakeholders')
    p_gbm = gbm.predict_proba(Xva)[:, 1]
    p_lr_bal = lr_bal.predict_proba(Sva)[:, 1]
    candidates = {'GBM': p_gbm, 'LR balanced': p_lr_bal}
    stakeholders = {'A public health': COST_PUBLIC_HEALTH, 'B employer': COST_EMPLOYER}

    at_default, at_optimal = {}, {}
    for name, prob in candidates.items():
        for sh, matrix in stakeholders.items():
            t = optimal_threshold(matrix)
            at_default[(name, sh)] = cost_per_person(yva, (prob >= 0.5).astype(int), matrix)
            at_optimal[(name, sh)] = cost_per_person(yva, (prob >= t).astype(int), matrix)
            print(f'    {name:<12} {sh:<18} p*={t:.3f}  cost@0.5={at_default[(name, sh)]:+7.1f}  '
                  f'cost@p*={at_optimal[(name, sh)]:+7.1f}  (EUR per person)')

    # The headline finding: at the default threshold the two stakeholders disagree
    # about which model to deploy. This is what makes "which model is best?" a
    # question that cannot be answered without asking "best for whom?".
    pick_default = {sh: min(candidates, key=lambda n: at_default[(n, sh)]) for sh in stakeholders}
    print(f'    at threshold 0.5:  {pick_default}')
    check(8, 'at the default threshold the two stakeholders prefer DIFFERENT models',
          len(set(pick_default.values())) == 2,
          ', '.join(f'{sh} -> {m}' for sh, m in pick_default.items()))

    # Costs cross zero, so report absolute improvement per person rather than a
    # percentage of a sign-changing quantity.
    gains = {k: at_default[k] - at_optimal[k] for k in at_default}
    best_gain = max(gains.values())
    check(8, 'moving off the default threshold saves real money',
          best_gain >= 25, f'best saving EUR {best_gain:.1f} per person')
    check(8, 'the two stakeholders want thresholds far apart',
          abs(optimal_threshold(COST_PUBLIC_HEALTH) - optimal_threshold(COST_EMPLOYER)) >= 0.20,
          f'{optimal_threshold(COST_PUBLIC_HEALTH):.3f} vs {optimal_threshold(COST_EMPLOYER):.3f}')

    # ---------------------------------------------------------------- rung 9
    section('Rung 9 -- the leakage trap')
    Xl, yl = encode(dev_clean, keep_leak=True)
    Xltr, Xlva, yltr, ylva = split(Xl, yl)
    gbm_leak = HistGradientBoostingClassifier(random_state=SEED).fit(Xltr, yltr)
    s_leak = scores(gbm_leak, Xlva, ylva)
    show('GBM including SelfRatedHealth', s_leak)
    leak_gain = s_leak['auc'] - s_gbm['auc']
    check(9, 'SelfRatedHealth gives a seductive, detectable gain',
          leak_gain >= 0.015, f'+{leak_gain:.4f} AUC, +{s_leak["acc"]-s_gbm["acc"]:.3f} accuracy')
    check(9, 'the leak is not so total that the exercise is trivial',
          leak_gain <= 0.08, f'+{leak_gain:.4f} AUC')

    # ---------------------------------------------------------------- rung 10
    section('Rung 10 -- train vs cross-validation vs held-out test')
    gbm_full = HistGradientBoostingClassifier(random_state=SEED).fit(Xc, yc)
    resub = accuracy_score(yc, gbm_full.predict(Xc))
    cv = cross_val_score(HistGradientBoostingClassifier(random_state=SEED), Xc, yc,
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                         scoring='accuracy').mean()
    Xt = align(X_test, Xc)
    test_acc = accuracy_score(y_test, gbm_full.predict(Xt))
    test_auc = roc_auc_score(y_test, gbm_full.predict_proba(Xt)[:, 1])
    print(f'    resubstitution (train)  acc={resub:.4f}')
    print(f'    5-fold cross-validation acc={cv:.4f}')
    print(f'    held-out test           acc={test_acc:.4f}  auc={test_auc:.4f}')
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
