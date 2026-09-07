#!/usr/bin/env python3
"""
Script 41: 1-Hour Ahead Volatility Prediction (Full Analysis)
=============================================================
Focused analysis on 1-hour prediction horizon - the optimal timeframe
discovered in multi-horizon testing.

Compares:
- Feature sets: Baseline, Sentiment, Graph, Combined
- Models: Lasso, Ridge, RF, GradientBoosting
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
RESULTS_DIR = (Path(__file__).resolve().parents[1] / 'results')

HORIZON = 1  # 1 hour ahead

print("="*70)
print("1-HOUR AHEAD VOLATILITY PREDICTION")
print("="*70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n📊 Loading data...")

# Sentiment features
features_df = pd.read_csv(DATA_DIR / 'btc_enhanced_features.csv', index_col=0, parse_dates=True)
features_df.index = features_df.index.tz_localize(None)

# Market data
market_df = pd.read_csv(DATA_DIR / 'btc_market_merged.csv', index_col=0, parse_dates=True)
market_df.index = market_df.index.tz_localize(None)

# Hourly graph features
graph_hourly = pd.read_csv(DATA_DIR / 'graph_features_hourly.csv', index_col=0, parse_dates=True)
graph_hourly.index = graph_hourly.index.floor('H').tz_localize(None)
graph_hourly = graph_hourly[~graph_hourly.index.duplicated(keep='last')]
graph_hourly.columns = ['graph_' + col for col in graph_hourly.columns]

# Merge all
market_subset = market_df[['volatility', 'price_return_1h']].rename(columns={'price_return_1h': 'returns'})
full_df = features_df.join(market_subset, how='inner')
full_df = full_df.join(graph_hourly, how='inner').dropna()

# Create target: 1-hour ahead volatility
full_df['Target'] = full_df['volatility'].shift(-HORIZON)
full_df = full_df.dropna()

print(f"   Samples: {len(full_df)}")
print(f"   Date range: {full_df.index.min()} to {full_df.index.max()}")

# =============================================================================
# 2. DEFINE FEATURE SETS
# =============================================================================
features_baseline = ['volatility', 'returns']
features_sentiment = [c for c in features_df.columns if c not in ['Close', 'log_price', 'volatility', 'returns']]
features_graph = [c for c in graph_hourly.columns]

# Remove overlap
features_sentiment = [f for f in features_sentiment if f not in features_baseline]

feature_sets = {
    'Baseline': features_baseline,
    'Sentiment': features_baseline + features_sentiment,
    'Graph': features_baseline + features_graph,
    'Combined': features_baseline + features_sentiment + features_graph
}

print(f"\n   Feature sets:")
for name, feats in feature_sets.items():
    print(f"      {name}: {len(feats)} features")

# =============================================================================
# 3. EVALUATION
# =============================================================================
outer_tscv = TimeSeriesSplit(n_splits=5)
inner_tscv = TimeSeriesSplit(n_splits=3)

def evaluate_all_models(df, features, set_name):
    """Evaluate multiple models for a feature set"""
    X = df[features]
    y = df['Target']

    models = {
        'Lasso': lambda: LassoCV(cv=inner_tscv, alphas=np.logspace(-4, 1, 50), max_iter=10000, random_state=42),
        'Ridge': lambda: RidgeCV(cv=inner_tscv, alphas=np.logspace(-4, 4, 50)),
        'RF': lambda: RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1),
        'GBM': lambda: GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    }

    results = {m: {'r2': [], 'rmse': [], 'mae': []} for m in models}

    for train_idx, test_idx in outer_tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for model_name, model_fn in models.items():
            model = model_fn()
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)

            results[model_name]['r2'].append(r2_score(y_test, y_pred))
            results[model_name]['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            results[model_name]['mae'].append(mean_absolute_error(y_test, y_pred))

    # Aggregate
    final = {}
    for model_name, metrics in results.items():
        final[model_name] = {
            'r2_mean': np.mean(metrics['r2']),
            'r2_std': np.std(metrics['r2']),
            'rmse_mean': np.mean(metrics['rmse']),
            'mae_mean': np.mean(metrics['mae'])
        }
    return final

# =============================================================================
# 4. RUN EXPERIMENTS
# =============================================================================
print("\n" + "="*70)
print(f"RUNNING 1-HOUR PREDICTION EXPERIMENTS")
print("="*70)

all_results = {}

for set_name, features in feature_sets.items():
    print(f"\n{set_name} ({len(features)} features)...")
    valid_features = [f for f in features if f in full_df.columns]
    res = evaluate_all_models(full_df, valid_features, set_name)
    all_results[set_name] = res

    for model, metrics in res.items():
        print(f"   {model}: R² = {metrics['r2_mean']:.4f} ± {metrics['r2_std']:.4f}")

# =============================================================================
# 5. FINAL COMPARISON TABLE
# =============================================================================
print("\n" + "="*70)
print("FINAL RESULTS: 1-Hour Ahead Volatility Prediction")
print("="*70)

print(f"\n{'Set':<12} {'Model':<8} {'R² Mean':<12} {'± Std':<10} {'RMSE':<12}")
print("-" * 55)

best_r2 = -999
best_config = None

for set_name in ['Baseline', 'Sentiment', 'Graph', 'Combined']:
    for model in ['Lasso', 'Ridge', 'RF', 'GBM']:
        m = all_results[set_name][model]
        print(f"{set_name:<12} {model:<8} {m['r2_mean']:<12.4f} {m['r2_std']:<10.4f} {m['rmse_mean']:<12.6f}")

        if m['r2_mean'] > best_r2:
            best_r2 = m['r2_mean']
            best_config = (set_name, model)

print(f"\n🏆 Best: {best_config[0]} + {best_config[1]} with R² = {best_r2:.4f}")

# Save results
output = {
    'horizon': HORIZON,
    'samples': len(full_df),
    'results': all_results,
    'best_config': {'feature_set': best_config[0], 'model': best_config[1], 'r2': best_r2}
}

with open(RESULTS_DIR / 'hourly_1h_prediction_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✅ Results saved to: results/hourly_1h_prediction_results.json")
