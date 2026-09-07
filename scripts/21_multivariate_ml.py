import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

data_dir = (Path(__file__).resolve().parents[1] / 'data')

features_file = data_dir / 'btc_enhanced_features.csv'
market_file = data_dir / 'btc_market_merged.csv'
univariate_results_file = data_dir / 'univariate_analysis_results.json'
output_file = data_dir / 'multivariate_ml_results.json'

print('=' * 80)
print('STUDY 3: MULTIVARIATE ML MODELS')
print('=' * 80)
print()
print('Models:')
print('  - Lasso (L1 regularization, feature selection)')
print('  - Ridge (L2 regularization, coefficient shrinkage)')
print('  - ElasticNet (L1 + L2, balanced approach)')
print()
print('Validation:')
print('  ✓ 5-fold walk-forward time-series cross-validation')
print('  ✓ Out-of-sample test set (20%)')
print('  ✓ Feature importance via coefficients')
print('  ✓ Comparison vs baseline (simple weighted sentiment)')
print()

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

train_size = int(0.8 * len(merged_clean))
train = merged_clean.iloc[:train_size].copy()
test = merged_clean.iloc[train_size:].copy()

print(f'\n📊 Train/test split:')
print(f'   Train: {len(train):,} hours ({train.index.min()} → {train.index.max()})')
print(f'   Test:  {len(test):,} hours ({test.index.min()} → {test.index.max()})')

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

def evaluate_model(model, X_train, y_train, X_test, y_test, scaler=None):
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    return {
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'train_rmse': float(train_rmse),
        'test_rmse': float(test_rmse),
        'train_mae': float(train_mae),
        'test_mae': float(test_mae),
        'n_features_used': int(np.sum(model.coef_ != 0)) if hasattr(model, 'coef_') else len(X_train.columns)
    }

def cross_validate_model(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_scores = []
    scaler = StandardScaler()

    for train_idx, val_idx in tscv.split(X):
        X_tr = X.iloc[train_idx]
        y_tr = y.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        model.fit(X_tr_scaled, y_tr)
        y_pred = model.predict(X_val_scaled)

        r2 = r2_score(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        cv_scores.append({'r2': r2, 'rmse': rmse})

    return {
        'mean_r2': float(np.mean([s['r2'] for s in cv_scores])),
        'std_r2': float(np.std([s['r2'] for s in cv_scores])),
        'mean_rmse': float(np.mean([s['rmse'] for s in cv_scores])),
        'std_rmse': float(np.std([s['rmse'] for s in cv_scores]))
    }

def get_top_features(model, feature_names, top_n=10):
    if not hasattr(model, 'coef_'):
        return []

    coef_abs = np.abs(model.coef_)
    top_indices = np.argsort(coef_abs)[::-1][:top_n]

    return [
        {
            'feature': feature_names[idx],
            'coefficient': float(model.coef_[idx]),
            'abs_coefficient': float(coef_abs[idx])
        }
        for idx in top_indices
        if coef_abs[idx] > 0
    ]

print()
print('=' * 80)
print('BASELINE: Simple Weighted Sentiment')
print('=' * 80)
print()

baseline_results = {}

for target in ['returns', 'volatility']:
    if 'sentiment_baseline' in train.columns:
        X_baseline_train = train[['sentiment_baseline']].copy()
        y_baseline_train = train[target].copy()
        X_baseline_test = test[['sentiment_baseline']].copy()
        y_baseline_test = test[target].copy()

        from sklearn.linear_model import LinearRegression
        baseline_model = LinearRegression()

        baseline_perf = evaluate_model(
            baseline_model,
            X_baseline_train,
            y_baseline_train,
            X_baseline_test,
            y_baseline_test
        )

        baseline_results[target] = baseline_perf

        print(f'{target}:')
        print(f'   Train R²: {baseline_perf["train_r2"]:.4f}')
        print(f'   Test R²:  {baseline_perf["test_r2"]:.4f}')
        print(f'   Test RMSE: {baseline_perf["test_rmse"]:.4f}')
        print()

results = {'baseline': baseline_results}

print()
print('=' * 80)
print('MULTIVARIATE MODELS')
print('=' * 80)
print()

models_config = {
    'lasso': Lasso(alpha=0.001, max_iter=5000),
    'ridge': Ridge(alpha=1.0),
    'elasticnet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)
}

for target in ['returns', 'volatility']:
    print(f'Target: {target.upper()}')
    print('-' * 80)

    features_to_use = significant_features[target]

    if len(features_to_use) == 0:
        print(f'  No significant features found, skipping...\n')
        continue

    X_train = train[features_to_use].copy()
    y_train = train[target].copy()
    X_test = test[features_to_use].copy()
    y_test = test[target].copy()

    print(f'  Using {len(features_to_use)} features')
    print()

    results[target] = {}

    for model_name, model in models_config.items():
        print(f'  {model_name.upper()}:')

        scaler = StandardScaler()
        perf = evaluate_model(model, X_train, y_train, X_test, y_test, scaler)

        cv_scores = cross_validate_model(model, X_train, y_train, n_splits=5)

        scaler_final = StandardScaler()
        X_train_scaled = scaler_final.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        top_features = get_top_features(model, features_to_use, top_n=10)

        results[target][model_name] = {
            'performance': perf,
            'cross_validation': cv_scores,
            'top_features': top_features
        }

        print(f'    Train R²: {perf["train_r2"]:.4f}')
        print(f'    Test R²:  {perf["test_r2"]:.4f}  (CV: {cv_scores["mean_r2"]:.4f} ± {cv_scores["std_r2"]:.4f})')
        print(f'    Test RMSE: {perf["test_rmse"]:.4f}')
        print(f'    Features used: {perf["n_features_used"]} / {len(features_to_use)}')

        if target in baseline_results:
            improvement = (perf["test_r2"] - baseline_results[target]["test_r2"]) * 100
            print(f'    Improvement over baseline: {improvement:+.2f} percentage points')

        print()

    print()

print()
print('=' * 80)
print('DIRECTIONAL ACCURACY (Returns only)')
print('=' * 80)
print()

if 'returns' in results:
    X_train_returns = train[significant_features['returns']].copy()
    y_train_returns = train['returns'].copy()
    X_test_returns = test[significant_features['returns']].copy()
    y_test_returns = test['returns'].copy()

    for model_name, model in models_config.items():
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_returns)
        X_test_scaled = scaler.transform(X_test_returns)

        model.fit(X_train_scaled, y_train_returns)
        y_pred = model.predict(X_test_scaled)

        y_true_direction = (y_test_returns > 0).astype(int)
        y_pred_direction = (y_pred > 0).astype(int)

        accuracy = (y_true_direction == y_pred_direction).mean()

        results['returns'][model_name]['directional_accuracy'] = float(accuracy)

        print(f'{model_name.upper()}: {accuracy:.2%} (baseline: 50%)')

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print()
print(f'✅ Results saved: {output_file}')
print(f'   Size: {output_file.stat().st_size / 1024:.1f} KB')

print()
print('=' * 80)
print('TOP FEATURES BY MODEL')
print('=' * 80)
print()

for target in ['returns', 'volatility']:
    if target not in results:
        continue

    print(f'{target.upper()}:')
    print()

    for model_name in ['lasso', 'ridge', 'elasticnet']:
        if model_name not in results[target]:
            continue

        print(f'  {model_name.upper()}:')
        top_feats = results[target][model_name]['top_features'][:5]
        for feat in top_feats:
            print(f'    {feat["feature"]:40s} coef={feat["coefficient"]:+.6f}')
        print()

print()
print('=' * 80)
print('STUDY 3 COMPLETE')
print('=' * 80)
print()
print('Key achievements:')
print('  ✓ Multivariate models trained with regularization')
print('  ✓ 5-fold time-series cross-validation performed')
print('  ✓ Out-of-sample validation on 20% hold-out test set')
print('  ✓ Feature importance extracted from model coefficients')
print('  ✓ Directional accuracy measured for returns prediction')
print()
print('Next: python3 scripts/22_event_study.py')
