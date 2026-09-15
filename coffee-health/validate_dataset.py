"""
Validation script for the coffee & health v3 dataset.

Checks both generated files against DATA_GENERATION_SPEC.md: ranges, distributions,
per-country SelfRatedHealth rates, key correlations, and data-quality issue counts --
then checks the two cohorts against each other, confirming that every intended
distribution shift is present at the intended magnitude and that the relationship
between features and outcome is NOT what changed.

For the ML properties the coursework depends on, see validate_ml_ladder.py.

Usage:
    python3 validate_dataset.py [--dev coffee_health_v3_dev.csv]
                                [--test coffee_health_v3_test.csv]
"""

import argparse
import sys

import numpy as np
import pandas as pd

COUNTRY_TARGETS_GOOD_PLUS = {'Norway': 0.80, 'Italy': 0.755, 'France': 0.685, 'UK': 0.65}

ORDERS = {
    'SelfRatedHealth': ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'],
    'Physical Activity Level': ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active'],
    'Stress Level': ['Low', 'Medium', 'High'],
    'Sleep Quality': ['Poor', 'Fair', 'Good', 'Excellent'],
    'Health Issues': ['No Issues', 'Mild', 'Moderate', 'Severe'],
    'Smoking Status': ['Never', 'Former', 'Vaper', 'Light Smoker', 'Heavy Smoker'],
    'Alcohol Level': ['Non-Drinker', 'Light', 'Moderate', 'Heavy'],
}

EXPECTED_COLUMNS = [
    'ID', 'Country', 'Age', 'Gender', 'Household Income', 'Smoking Status',
    'Alcohol Level', 'Daily Coffees', 'Caffeine Intake', 'Stress Level',
    'Physical Activity Level', 'BMI', 'Avg Resting Heart Rate',
    'Avg Sleep Hours Per Night', 'Sleep Quality', 'Health Issues',
    'SelfRatedHealth', 'HighHealthNeeds',
]

RESULTS = []


def check(description, condition, detail=''):
    RESULTS.append((bool(condition), description, detail))
    print(f'  [{"PASS" if condition else "FAIL"}] {description}' + (f': {detail}' if detail else ''))


def section(title):
    print(f'\n{title}\n' + '-' * len(title))


def ordinal(series, order):
    return series.map({v: i for i, v in enumerate(order)})


# --------------------------------------------------------------------------------------

def validate_cohort(df, name, *, expect_missing):
    print('\n' + '=' * 78)
    print(f'{name.upper()}  ({len(df)} rows)')
    print('=' * 78)

    section('Schema')
    check('all expected columns present, in order',
          list(df.columns) == EXPECTED_COLUMNS,
          f'{len(df.columns)} columns')
    check('target is never missing',
          df['HighHealthNeeds'].notna().all() and df['SelfRatedHealth'].notna().all())
    check('HighHealthNeeds is binary 0/1',
          set(df['HighHealthNeeds'].unique()) <= {0, 1})

    section('Ranges (anomaly-tolerant bounds)')
    for col, lo, hi in [('Age', 13, 999),            # >100 are doubled-digit typos
                        ('BMI', 16, 450),            # >60 are missed decimal points
                        ('Household Income', 12000, 250000),
                        ('Avg Resting Heart Rate', 45, 110),
                        ('Avg Sleep Hours Per Night', 3, 10.5),
                        ('Daily Coffees', 0, 9),
                        ('Caffeine Intake', 0, 750)]:
        s = df[col].dropna()
        check(f'{col} within [{lo}, {hi}]', s.between(lo, hi).all(),
              f'observed [{s.min():.1f}, {s.max():.1f}]')

    section('Categorical distributions')
    for col, order in ORDERS.items():
        observed = df[col].value_counts(normalize=True, dropna=True)
        # Both directions matter. An earlier version only checked for UNEXPECTED levels,
        # which let a silently missing SelfRatedHealth class through unnoticed.
        check(f'{col} uses only its defined levels',
              set(observed.index) <= set(order),
              ', '.join(f'{k} {v:.1%}' for k, v in observed.reindex(order).dropna().items()))
        check(f'{col} has every defined level present',
              set(order) <= set(observed.index),
              'all present' if set(order) <= set(observed.index)
              else f'MISSING: {sorted(set(order) - set(observed.index))}')

    section('Missing values')
    missing = df.isna().sum()
    total = int(missing.sum())
    if expect_missing:
        check('development set carries missing values', total > 0, f'{total} cells')
        rates = {c: missing[c] / len(df) for c in
                 ['Avg Sleep Hours Per Night', 'Avg Resting Heart Rate', 'Stress Level',
                  'Health Issues']}
        print('    ' + '  '.join(f'{c}: {r:.1%}' for c, r in rates.items()))
        # Missingness in the device-sourced fields should rise with age.
        bands = pd.cut(df['Age'].where(df['Age'] <= 100), [0, 29, 44, 59, 999])
        by_band = df.groupby(bands, observed=True)['Avg Sleep Hours Per Night'].apply(
            lambda s: s.isna().mean())
        print(f'    sleep-hours missingness by age band: '
              + ', '.join(f'{i}: {v:.1%}' for i, v in by_band.items()))
        check('missingness rises with age (a signal, not noise)',
              by_band.is_monotonic_increasing, f'{by_band.iloc[0]:.1%} -> {by_band.iloc[-1]:.1%}')
    else:
        check('TEST SET HAS NO MISSING VALUES', total == 0, f'{total} cells')

    section('Data quality issues')
    n_age = int(((df['Age'] > 100) | (df['Age'] < 13)).sum())
    n_bmi = int((df['BMI'] > 60).sum())
    n_dupes = int(df.duplicated().sum())
    print(f'    age anomalies {n_age} ({n_age/len(df):.2%})   '
          f'BMI anomalies {n_bmi} ({n_bmi/len(df):.2%})   '
          f'duplicate rows {n_dupes} ({n_dupes/len(df):.2%})')
    check('age anomalies are exactly recoverable (drop the trailing digit)',
          all(int(str(int(v))[:-1]) == int(str(int(v))[:-1]) and 13 <= int(str(int(v))[:-1]) <= 100
              for v in df.loc[df['Age'] > 100, 'Age']),
          f'{n_age} rows')
    check('BMI anomalies are exactly recoverable (divide by 10)',
          df.loc[df['BMI'] > 60, 'BMI'].div(10).between(16, 45).all(), f'{n_bmi} rows')
    check('duplicates are present for students to find', n_dupes > 0, f'{n_dupes} rows')

    section('Correlations (anomalies excluded)')
    clean = df[(df['Age'] <= 100) & (df['BMI'] <= 60)]
    pairs = [
        ('BMI', ordinal(clean['Physical Activity Level'], ORDERS['Physical Activity Level']), 'neg'),
        ('Avg Resting Heart Rate', ordinal(clean['Physical Activity Level'], ORDERS['Physical Activity Level']), 'neg'),
        ('Daily Coffees', clean['Caffeine Intake'], 'pos'),
    ]
    for col, other, direction in pairs:
        r = clean[col].corr(other)
        ok = (r < -0.15) if direction == 'neg' else (r > 0.15)
        check(f'{col} correlates {direction} as designed', ok, f'r = {r:+.3f}')

    sleep_stress = ordinal(clean['Sleep Quality'], ORDERS['Sleep Quality']).corr(
        ordinal(clean['Stress Level'], ORDERS['Stress Level']))
    check('Sleep Quality correlates negatively with Stress Level', sleep_stress < -0.15,
          f'r = {sleep_stress:+.3f}')
    srh_health = ordinal(clean['SelfRatedHealth'], ORDERS['SelfRatedHealth']).corr(
        ordinal(clean['Health Issues'], ORDERS['Health Issues']))
    check('SelfRatedHealth correlates negatively with Health Issues', srh_health < -0.15,
          f'r = {srh_health:+.3f}')

    section('Targets')
    srh = df['SelfRatedHealth'].value_counts(normalize=True).reindex(ORDERS['SelfRatedHealth'])
    print('    SelfRatedHealth: ' + ', '.join(f'{k} {v:.1%}' for k, v in srh.items()))
    print(f'    HighHealthNeeds positive rate: {df["HighHealthNeeds"].mean():.1%}')
    good_plus = df.groupby('Country')['SelfRatedHealth'].apply(
        lambda s: s.isin(['Good', 'Very Good', 'Excellent']).mean())
    print('    "good or better" by country: '
          + ', '.join(f'{k} {v:.1%}' for k, v in good_plus.items()))
    check('country ordering of self-rated health is preserved '
          '(Norway best, UK worst)',
          good_plus['Norway'] > good_plus['France'] and good_plus['Italy'] > good_plus['UK'],
          ' > '.join(good_plus.sort_values(ascending=False).index))

    if expect_missing:   # the calibrated country targets apply to the unshifted cohort
        for country, target in COUNTRY_TARGETS_GOOD_PLUS.items():
            check(f'{country} "good or better" within 5pp of {target:.1%}',
                  abs(good_plus[country] - target) <= 0.05,
                  f'{good_plus[country]:.1%}')


def validate_shift(dev, test):
    print('\n' + '=' * 78)
    print('DEVELOPMENT vs TEST -- distribution shift')
    print('=' * 78)

    section('Intended shifts are present')
    uk_dev, uk_test = (dev['Country'] == 'UK').mean(), (test['Country'] == 'UK').mean()
    check('UK is over-represented in the test cohort', uk_test > uk_dev + 0.15,
          f'{uk_dev:.1%} -> {uk_test:.1%}')

    age_dev = dev['Age'].where(dev['Age'] <= 100).mean()
    age_test = test['Age'].where(test['Age'] <= 100).mean()
    check('the test cohort is older', age_test > age_dev + 2.0,
          f'{age_dev:.1f} -> {age_test:.1f} years')

    vape_dev = (dev['Smoking Status'] == 'Vaper').mean()
    vape_test = (test['Smoking Status'] == 'Vaper').mean()
    check('vaping is far more common in the test cohort', vape_test > vape_dev * 2.5,
          f'{vape_dev:.1%} -> {vape_test:.1%}')

    pos_dev, pos_test = dev['HighHealthNeeds'].mean(), test['HighHealthNeeds'].mean()
    check('the positive rate rises (the apparent label shift)', pos_test > pos_dev + 0.03,
          f'{pos_dev:.1%} -> {pos_test:.1%}')

    check('the test cohort carries more data-entry corruption',
          (test['Age'] > 100).mean() > (dev['Age'] > 100).mean() * 1.8,
          f'age anomalies {(dev["Age"] > 100).mean():.2%} -> {(test["Age"] > 100).mean():.2%}')
    dup_uk = (test[test.duplicated()]['Country'] == 'UK').mean()
    check('test duplicates are UK-weighted, not a random sample', dup_uk > 0.5,
          f'{dup_uk:.0%} of duplicated rows are UK')

    section('The apparent label shift is compositional')
    # Reweight the development set's own within-group rates to the test cohort's mix.
    # If the relationship between features and outcome had changed, this would not
    # move towards the observed test rate.
    band = lambda d: pd.cut(d['Age'].where(d['Age'] <= 100), [0, 30, 45, 60, 100])
    keys_dev = [dev['Country'], band(dev), dev['Smoking Status']]
    keys_test = [test['Country'], band(test), test['Smoking Status']]
    dev_rates = dev.groupby(keys_dev, observed=True)['HighHealthNeeds'].mean()
    test_mix = test.groupby(keys_test, observed=True).size() / len(test)
    shared = dev_rates.index.intersection(test_mix.index)
    expected = float((dev_rates[shared] * test_mix[shared]).sum())
    coverage = float(test_mix[shared].sum())

    print(f'    development rate                        {pos_dev:.3f}')
    print(f'    reweighted to the test cohort\'s mix     {expected:.3f} '
          f'(covering {coverage:.0%} of test rows)')
    print(f'    observed test rate                      {pos_test:.3f}')
    check('reweighting recovers most of the rise, so the shift is in WHO was recruited',
          expected > pos_dev + 0.5 * (pos_test - pos_dev),
          f'{expected - pos_dev:+.3f} of the {pos_test - pos_dev:+.3f} observed rise')

    section('Within-group relationships are stable')
    for country in sorted(dev['Country'].unique()):
        d = dev[dev['Country'] == country]
        t = test[test['Country'] == country]
        for lo, hi in [(30, 45), (45, 60)]:
            dm = d[(d['Age'] > lo) & (d['Age'] <= hi)]['HighHealthNeeds'].mean()
            tm = t[(t['Age'] > lo) & (t['Age'] <= hi)]['HighHealthNeeds'].mean()
            print(f'    {country:<7} age {lo}-{hi}:  dev {dm:.3f}  test {tm:.3f}  ({tm - dm:+.3f})')
    # A coarse stability check; validate_ml_ladder.py's calibration test is the strict one.
    joint = []
    for country in dev['Country'].unique():
        for lo, hi in [(18, 30), (30, 45), (45, 60), (60, 100)]:
            d = dev[(dev['Country'] == country) & (dev['Age'] > lo) & (dev['Age'] <= hi)]
            t = test[(test['Country'] == country) & (test['Age'] > lo) & (test['Age'] <= hi)]
            if len(d) >= 100 and len(t) >= 50:
                joint.append(t['HighHealthNeeds'].mean() - d['HighHealthNeeds'].mean())
    check('within country x age band, the outcome rate barely moves',
          abs(float(np.mean(joint))) <= 0.06,
          f'mean difference {np.mean(joint):+.3f} across {len(joint)} cells')


def main():
    parser = argparse.ArgumentParser(description='Validate the coffee & health v3 dataset')
    parser.add_argument('--dev', default='coffee_health_v3_dev.csv')
    parser.add_argument('--test', default='coffee_health_v3_test.csv')
    args = parser.parse_args()

    dev = pd.read_csv(args.dev)
    test = pd.read_csv(args.test)

    validate_cohort(dev, 'development set', expect_missing=True)
    validate_cohort(test, 'held-out test set', expect_missing=False)
    validate_shift(dev, test)

    failed = [r for r in RESULTS if not r[0]]
    print('\n' + '=' * 78)
    print(f'{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed.')
    if failed:
        print('\nFailed:')
        for _, description, detail in failed:
            print(f'  - {description}' + (f' ({detail})' if detail else ''))
        return 1
    print('Both cohorts match the specification.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
