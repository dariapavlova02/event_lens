import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

data_dir = (Path(__file__).resolve().parents[1] / 'data')

features_file = data_dir / 'btc_enhanced_features.csv'
market_file = data_dir / 'btc_market_merged.csv'
univariate_results_file = data_dir / 'univariate_analysis_results.json'

print('=' * 80)
print('DIAGNOSTIC ANALYSIS: WHY MODELS FAIL ON TEST SET')
print('=' * 80)
print()

print('Loading data...')
features_df = pd.read_csv(features_file, index_col=0, parse_dates=True)
market_df = pd.read_csv(market_file, index_col=0, parse_dates=True)

with open(univariate_results_file, 'r') as f:
    univariate_results = json.load(f)

merged = features_df.join(market_df[['Close', 'price_return_1h', 'volatility', 'log_price']], how='inner')
merged = merged.rename(columns={'price_return_1h': 'returns'})
merged_clean = merged.dropna()

train_size = int(0.8 * len(merged_clean))
train = merged_clean.iloc[:train_size].copy()
test = merged_clean.iloc[train_size:].copy()

print(f'Train: {len(train):,} hours ({train.index.min()} → {train.index.max()})')
print(f'Test:  {len(test):,} hours ({test.index.min()} → {test.index.max()})')

print()
print('=' * 80)
print('HYPOTHESIS 1: Feature Distribution Shift')
print('=' * 80)
print()
print('Checking if top features have different distributions in train vs test...')
print()

feature_cols = [col for col in features_df.columns
                if col not in ['Close', 'returns', 'volatility', 'log_price']]

top_features_volatility = []
for col, res in univariate_results['correlations']['volatility'].items():
    if col in feature_cols and res['train']['n'] >= 30:
        r = abs(res['train']['r_pearson'])
        top_features_volatility.append((col, r))

top_features_volatility.sort(key=lambda x: x[1], reverse=True)
top_10_features = [f[0] for f in top_features_volatility[:10]]

print('Top 10 features by univariate correlation:')
for i, (feat, r) in enumerate(top_features_volatility[:10], 1):
    train_mean = train[feat].mean()
    test_mean = test[feat].mean()
    train_std = train[feat].std()
    test_std = test[feat].std()

    mean_shift = ((test_mean - train_mean) / train_mean * 100) if train_mean != 0 else 0
    std_shift = ((test_std - train_std) / train_std * 100) if train_std != 0 else 0

    print(f'{i:2d}. {feat:40s} r={r:.3f}')
    print(f'    Train: μ={train_mean:8.4f} σ={train_std:8.4f}')
    print(f'    Test:  μ={test_mean:8.4f} σ={test_std:8.4f}')
    print(f'    Shift: μ={mean_shift:+6.1f}% σ={std_shift:+6.1f}%')
    print()

print()
print('=' * 80)
print('HYPOTHESIS 2: Feature-Target Relationship Changes')
print('=' * 80)
print()
print('Checking if correlations are different in train vs test...')
print()

print(f'{"Feature":40s} {"Train r":>10s} {"Test r":>10s} {"Δ":>10s} {"Sign flip?":>12s}')
print('-' * 80)

correlation_shifts = []

for feat in top_10_features:
    train_corr = train[feat].corr(train['volatility'])
    test_corr = test[feat].corr(test['volatility'])

    delta = test_corr - train_corr
    sign_flip = (train_corr * test_corr) < 0

    correlation_shifts.append({
        'feature': feat,
        'train_r': train_corr,
        'test_r': test_corr,
        'delta': delta,
        'sign_flip': sign_flip
    })

    print(f'{feat:40s} {train_corr:10.4f} {test_corr:10.4f} {delta:+10.4f} {"YES ⚠️" if sign_flip else "No":>12s}')

print()
n_flips = sum(1 for c in correlation_shifts if c['sign_flip'])
print(f'Sign flips: {n_flips}/{len(top_10_features)} features ({n_flips/len(top_10_features)*100:.1f}%)')

print()
print('=' * 80)
print('HYPOTHESIS 3: Simple Univariate Models Performance')
print('=' * 80)
print()
print('Testing if even simple univariate models fail on test set...')
print()

print(f'{"Feature":40s} {"Train R²":>10s} {"Test R²":>10s} {"Gap":>10s}')
print('-' * 80)

univariate_performance = []

for feat in top_10_features:
    X_train = train[[feat]].values
    y_train = train['volatility'].values
    X_test = test[[feat]].values
    y_test = test['volatility'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_scaled, y_train)

    train_r2 = r2_score(y_train, model.predict(X_train_scaled))
    test_r2 = r2_score(y_test, model.predict(X_test_scaled))
    gap = train_r2 - test_r2

    univariate_performance.append({
        'feature': feat,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'gap': gap
    })

    print(f'{feat:40s} {train_r2:10.4f} {test_r2:10.4f} {gap:+10.4f}')

print()
n_negative = sum(1 for p in univariate_performance if p['test_r2'] < 0)
print(f'Negative test R²: {n_negative}/{len(top_10_features)} features')

print()
print('=' * 80)
print('HYPOTHESIS 4: Data Leakage Check')
print('=' * 80)
print()
print('Checking if features contain future information...')
print()

suspicious_features = []
for feat in top_10_features:
    if 'ma_6h' in feat or 'ma_24h' in feat or 'rolling' in feat.lower():
        suspicious_features.append(feat)
        print(f'⚠️  {feat} - contains moving average (potential look-ahead)')

if not suspicious_features:
    print('✓ No obvious data leakage detected')

print()
print('=' * 80)
print('HYPOTHESIS 5: Target Variable Predictability')
print('=' * 80)
print()
print('Checking autocorrelation of volatility (is it predictable at all?)...')
print()

for lag in [1, 2, 3, 6, 12, 24]:
    train_autocorr = train['volatility'].autocorr(lag=lag)
    test_autocorr = test['volatility'].autocorr(lag=lag)

    print(f'Lag {lag:2d}h: Train autocorr={train_autocorr:+.4f}, Test autocorr={test_autocorr:+.4f}')

print()
print('=' * 80)
print('HYPOTHESIS 6: Sample Size in Test Set')
print('=' * 80)
print()

print(f'Test set size: {len(test):,} samples')
print(f'Number of features: {len(top_10_features)}')
print(f'Samples per feature: {len(test) / len(top_10_features):.1f}')
print()

if len(test) < 500:
    print('⚠️  Test set may be too small for reliable evaluation')
elif len(test) / len(top_10_features) < 30:
    print('⚠️  Low samples-per-feature ratio')
else:
    print('✓ Sample size appears adequate')

print()
print('=' * 80)
print('HYPOTHESIS 7: Non-linear Relationship')
print('=' * 80)
print()
print('Checking if relationship is non-linear...')
print()

for feat in top_10_features[:3]:
    train_linear_corr = abs(train[feat].corr(train['volatility']))
    train_spearman = abs(train[feat].corr(train['volatility'], method='spearman'))

    print(f'{feat:40s}')
    print(f'  Pearson (linear):  {train_linear_corr:.4f}')
    print(f'  Spearman (rank):   {train_spearman:.4f}')
    print(f'  Difference:        {abs(train_spearman - train_linear_corr):.4f}')
    print()

print()
print('=' * 80)
print('HYPOTHESIS 8: Volatility Regime Analysis')
print('=' * 80)
print()
print('Analyzing volatility regimes in train vs test...')
print()

train_vol = train['volatility']
test_vol = test['volatility']

train_quantiles = train_vol.quantile([0.25, 0.5, 0.75])
test_quantiles = test_vol.quantile([0.25, 0.5, 0.75])

print('Volatility distribution:')
print(f'  Train - 25%: {train_quantiles[0.25]:.6f}, 50%: {train_quantiles[0.5]:.6f}, 75%: {train_quantiles[0.75]:.6f}')
print(f'  Test  - 25%: {test_quantiles[0.25]:.6f}, 50%: {test_quantiles[0.5]:.6f}, 75%: {test_quantiles[0.75]:.6f}')
print()

train_high_vol = (train_vol > train_vol.quantile(0.75)).sum()
test_high_vol = (test_vol > train_vol.quantile(0.75)).sum()

print(f'High volatility periods (>75th percentile):')
print(f'  Train: {train_high_vol} ({train_high_vol/len(train)*100:.1f}%)')
print(f'  Test:  {test_high_vol} ({test_high_vol/len(test)*100:.1f}%)')
print()

if test_high_vol / len(test) < train_high_vol / len(train) * 0.5:
    print('⚠️  Test period has much lower volatility - models trained on high volatility may not work')

print()
print('=' * 80)
print('HYPOTHESIS 9: Rolling Window Validation')
print('=' * 80)
print()
print('Testing model on different time windows...')
print()

n_windows = 5
window_size = len(test) // n_windows

feat = 'total_forwards'
X_train = train[[feat]].values
y_train = train['volatility'].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train)

print(f'Using feature: {feat}')
print()
print(f'{"Window":>8s} {"Period":>25s} {"Test R²":>10s}')
print('-' * 50)

for i in range(n_windows):
    start_idx = i * window_size
    end_idx = (i + 1) * window_size if i < n_windows - 1 else len(test)

    test_window = test.iloc[start_idx:end_idx]

    X_test_window = test_window[[feat]].values
    y_test_window = test_window['volatility'].values

    X_test_scaled = scaler.transform(X_test_window)
    test_r2 = r2_score(y_test_window, model.predict(X_test_scaled))

    period = f'{test_window.index[0].strftime("%Y-%m-%d")} to {test_window.index[-1].strftime("%Y-%m-%d")}'
    print(f'{i+1:8d} {period:>25s} {test_r2:10.4f}')

print()
print('=' * 80)
print('DIAGNOSTIC SUMMARY')
print('=' * 80)
print()

print('Key Findings:')
print()
print('1. Feature distribution shifts:')
print(f'   - Check mean/std changes above')
print()
print('2. Correlation stability:')
print(f'   - {n_flips}/{len(top_10_features)} features changed sign')
print()
print('3. Univariate performance:')
print(f'   - {n_negative}/{len(top_10_features)} features have negative test R²')
print()
print('4. Volatility regime:')
print(f'   - Test period has {(test_vol.mean()/train_vol.mean()-1)*100:+.1f}% mean volatility change')
print()
print('Next steps based on findings:')
print('  - If sign flips: relationship changed, need different approach')
print('  - If regime shift: consider regime-specific models')
print('  - If all univariate fail: fundamental predictability issue')
print()
