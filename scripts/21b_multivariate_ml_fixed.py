#!/usr/bin/env python3
"""
Script 21b: Multivariate ML Models (Correct Validation)
=======================================================
METHODOLOGICAL FIX: Pure Walk-Forward Cross-Validation

Previous issue (Script 21):
- Single 80/20 holdout + CV on train only
- Test R² depends on single period
- No uncertainty estimate

Fixed approach:
- Pure 5-fold Walk-Forward CV
- Each fold has its own train/test
- Report: Mean R² ± Std (statistically robust)

Reference:
- Tashman (2000) "Out-of-sample tests of forecasting accuracy"
- Bergmeir & Benítez (2012) "On the use of cross-validation for time series"
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

data_dir = (Path(__file__).resolve().parents[1] / 'data')
results_dir = (Path(__file__).resolve().parents[1] / 'results')

features_file = data_dir / 'btc_enhanced_features.csv'
market_file = data_dir / 'btc_market_merged.csv'
univariate_results_file = data_dir / 'univariate_analysis_results.json'
output_file = data_dir / 'multivariate_ml_results_v2.json'

print('=' * 80)
print('STUDY 3b: MULTIVARIATE ML (CORRECT WALK-FORWARD CV)')
print('=' * 80)
print()
print('Methodology:')
print('  ✅ Pure Walk-Forward CV (no single holdout)')
print('  ✅ 5 independent test periods')
print('  ✅ Report: Mean ± Std (statistically robust)')
print('  ✅ Scaler fit on train only per fold')
print()

# Load data
print(f'📊 Loading data...')
features_df = pd.read_csv(features_file, index_col=0, parse_dates=True)
market_df = pd.read_csv(market_file, index_col=0, parse_dates=True)

with open(univariate_results_file, 'r') as f:
    univariate_results = json.load(f)

merged = features_df.join(market_df[['Close', 'price_return_1h', 'volatility', 'log_price']], how='inner')
merged = merged.rename(columns={'price_return_1h': 'returns'})
merged_clean = merged.dropna()

print(f'   Total hours: {len(merged_clean):,}')
print(f'   Period: {merged_clean.index.min()} → {merged_clean.index.max()}')

# Feature selection (use univariate results)
feature_cols = [col for col in features_df.columns
                if col not in ['Close', 'returns', 'volatility', 'log_price']]

print(f'\n🔍 Selecting features based on univariate analysis...')

significant_features = {}
for target in ['returns', 'volatility']:
    corr_results = univariate_results['correlations'][target]

    sig_feats = [
        col for col, res in corr_results.items()
        if res.get('fdr_rejected', False) and res['train']['n'] >= 30
    ]

    sig_feats = [f for f in sig_feats if f in feature_cols]
    significant_features[target] = sig_feats

    print(f'   {target}: {len(sig_feats)} significant features')

# Walk-Forward CV
outer_tscv = TimeSeriesSplit(n_splits=5)
inner_tscv = TimeSeriesSplit(n_splits=3)

def walk_forward_cv(X, y, model_name, use_cv_alpha=False):
    """
    Pure Walk-Forward Cross-Validation

    Returns dict with mean ± std of metrics across 5 folds
    """
    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_tscv.split(X)):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Scaler: fit ONLY on train fold
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Model with optional nested CV for alpha
        if use_cv_alpha and model_name == 'lasso':
            model = LassoCV(cv=inner_tscv, alphas=np.logspace(-4, 1, 50), max_iter=10000)
        elif model_name == 'lasso':
            model = Lasso(alpha=0.01, max_iter=5000)
        elif model_name == 'ridge':
            model = Ridge(alpha=1.0)
        elif model_name == 'elasticnet':
            model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        # Metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Direction accuracy (for returns)
        direction_acc = ((y_test > 0) == (y_pred > 0)).mean() if y_test.std() > 0 else 0.5

        # Feature importance
        n_features = int(np.sum(model.coef_ != 0)) if hasattr(model, 'coef_') else X_train.shape[1]

        fold_results.append({
            'fold': fold_idx + 1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'test_period': f"{X_test.index.min().date()} → {X_test.index.max().date()}",
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'direction_acc': direction_acc,
            'n_features': n_features,
            'alpha': model.alpha_ if hasattr(model, 'alpha_') else None
        })

    # Summary
    r2_values = [f['r2'] for f in fold_results]
    rmse_values = [f['rmse'] for f in fold_results]

    return {
        'model': model_name,
        'mean_r2': float(np.mean(r2_values)),
        'std_r2': float(np.std(r2_values)),
        'mean_rmse': float(np.mean(rmse_values)),
        'std_rmse': float(np.std(rmse_values)),
        'mean_direction_acc': float(np.mean([f['direction_acc'] for f in fold_results])),
        'folds': fold_results
    }

# Run evaluation
results = {}

for target in ['returns', 'volatility']:
    print(f'\n{"="*80}')
    print(f'TARGET: {target.upper()}')
    print(f'{"="*80}')

    features_to_use = significant_features[target]

    if len(features_to_use) == 0:
        print(f'  No significant features found, skipping...')
        continue

    print(f'  Using {len(features_to_use)} features')

    X = merged_clean[features_to_use]
    y = merged_clean[target]

    results[target] = {}

    for model_name in ['lasso', 'ridge', 'elasticnet']:
        print(f'\n  {model_name.upper()}:')

        # Use nested CV for alpha selection (for Lasso)
        use_cv = (model_name == 'lasso')

        cv_result = walk_forward_cv(X, y, model_name, use_cv_alpha=use_cv)
        results[target][model_name] = cv_result

        print(f'    R²: {cv_result["mean_r2"]:.4f} ± {cv_result["std_r2"]:.4f}')
        print(f'    RMSE: {cv_result["mean_rmse"]:.6f} ± {cv_result["std_rmse"]:.6f}')

        if target == 'returns':
            print(f'    Direction Accuracy: {cv_result["mean_direction_acc"]:.2%}')

        # Show fold-by-fold
        print(f'    Folds:')
        for fold in cv_result['folds']:
            print(f'      Fold {fold["fold"]}: R²={fold["r2"]:.4f}, '
                  f'test_size={fold["test_size"]}, '
                  f'period={fold["test_period"]}')

# Baseline comparison
print(f'\n{"="*80}')
print(f'BASELINE: Simple Weighted Sentiment')
print(f'{"="*80}')

for target in ['returns', 'volatility']:
    if 'sentiment_baseline' not in merged_clean.columns:
        continue

    X_baseline = merged_clean[['sentiment_baseline']]
    y_baseline = merged_clean[target]

    baseline_result = walk_forward_cv(X_baseline, y_baseline, 'ridge', use_cv_alpha=False)
    results[f'{target}_baseline'] = baseline_result

    print(f'\n  {target.upper()}:')
    print(f'    R²: {baseline_result["mean_r2"]:.4f} ± {baseline_result["std_r2"]:.4f}')

# Save results
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f'\n✅ Results saved: {output_file}')

# Final comparison table
print(f'\n{"="*80}')
print(f'FINAL COMPARISON')
print(f'{"="*80}')

print('\nVolatility Prediction:')
print(f'{"Model":<15} {"Mean R²":>12} {"± Std":>10} {"Improvement":>12}')
print('-' * 50)

baseline_r2 = results.get('volatility_baseline', {}).get('mean_r2', 0)
for model_name in ['lasso', 'ridge', 'elasticnet']:
    if 'volatility' in results and model_name in results['volatility']:
        r = results['volatility'][model_name]
        imp = r['mean_r2'] - baseline_r2
        print(f'{model_name:<15} {r["mean_r2"]:>12.4f} {r["std_r2"]:>10.4f} {imp:>+12.4f}')

print(f'{"baseline":<15} {baseline_r2:>12.4f} {"":>10} {0:>+12.4f}')

print('\nReturns Prediction:')
print(f'{"Model":<15} {"Mean R²":>12} {"± Std":>10} {"Direction Acc":>14}')
print('-' * 55)

for model_name in ['lasso', 'ridge', 'elasticnet']:
    if 'returns' in results and model_name in results['returns']:
        r = results['returns'][model_name]
        print(f'{model_name:<15} {r["mean_r2"]:>12.4f} {r["std_r2"]:>10.4f} {r["mean_direction_acc"]:>14.2%}')

print(f'\n{"="*80}')
print('VALIDATION METHODOLOGY')
print('=' * 80)
print('''
✅ Pure Walk-Forward CV (5 folds)
   - No single holdout test set
   - Each fold has independent train/test split
   - Results: Mean ± Std (statistically meaningful)

✅ Nested CV for Hyperparameter Selection (Lasso)
   - Inner CV: 3-fold TimeSeriesSplit
   - Prevents look-ahead bias in alpha selection

✅ Scaler Fit on Train Only
   - Each fold: scaler.fit(train), scaler.transform(test)
   - No information leakage from test to train

Reference:
   Bergmeir & Benítez (2012) "On the use of cross-validation
   for time series predictor evaluation"
''')
