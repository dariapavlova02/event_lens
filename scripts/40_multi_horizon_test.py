#!/usr/bin/env python3
"""
Script 40: Hourly Multi-Horizon Comparison
==========================================
Tests multiple prediction horizons to find optimal timeframe.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')

print("="*70)
print("MULTI-HORIZON COMPARISON: Finding Optimal Prediction Window")
print("="*70)

# Load data
print("📊 Loading data...")
features_df = pd.read_csv(DATA_DIR / 'btc_enhanced_features.csv', index_col=0, parse_dates=True)
features_df.index = features_df.index.tz_localize(None)

market_df = pd.read_csv(DATA_DIR / 'btc_market_merged.csv', index_col=0, parse_dates=True)
market_df.index = market_df.index.tz_localize(None)

graph_hourly = pd.read_csv(DATA_DIR / 'graph_features_hourly.csv', index_col=0, parse_dates=True)
graph_hourly.index = graph_hourly.index.floor('H').tz_localize(None)
graph_hourly = graph_hourly[~graph_hourly.index.duplicated(keep='last')]
graph_hourly.columns = ['graph_' + col for col in graph_hourly.columns]

# Merge
market_subset = market_df[['volatility', 'price_return_1h']].rename(columns={'price_return_1h': 'returns'})
full_df = features_df.join(market_subset, how='inner')
full_df = full_df.join(graph_hourly, how='inner').dropna()

print(f"   Total samples before target: {len(full_df)}")

# Define features
features_baseline = ['volatility', 'returns']
features_graph = [c for c in graph_hourly.columns]

# Evaluation function
outer_tscv = TimeSeriesSplit(n_splits=5)

def evaluate_horizon(df, features, horizon_hours):
    """Evaluate model at specific horizon"""
    df_copy = df.copy()
    df_copy['target'] = df_copy['volatility'].shift(-horizon_hours)
    df_copy = df_copy.dropna()

    X = df_copy[features]
    y = df_copy['target']

    if len(X) < 100:
        return None

    r2_scores = []
    for train_idx, test_idx in outer_tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        inner_cv = TimeSeriesSplit(n_splits=3)
        lasso = LassoCV(cv=inner_cv, random_state=42).fit(X_train_s, y_train)
        r2_scores.append(r2_score(y_test, lasso.predict(X_test_s)))

    return np.mean(r2_scores)

# Test horizons
horizons = [1, 2, 3, 6, 12, 24, 48, 72]

print("\n" + "="*50)
print("TESTING PREDICTION HORIZONS")
print("="*50)

print(f"\n{'Horizon':<12} {'Baseline R²':<15} {'+Graph R²':<15} {'Δ':<10}")
print("-" * 55)

results = []
for h in horizons:
    r2_base = evaluate_horizon(full_df, features_baseline, h)
    r2_graph = evaluate_horizon(full_df, features_baseline + features_graph, h)

    if r2_base is not None and r2_graph is not None:
        delta = r2_graph - r2_base
        print(f"{h}h{'':<10} {r2_base:<15.4f} {r2_graph:<15.4f} {delta:+.4f}")
        results.append({'horizon': h, 'baseline': r2_base, 'graph': r2_graph, 'delta': delta})

# Find best
if results:
    best = max(results, key=lambda x: x['graph'])
    print(f"\n🏆 Best horizon: {best['horizon']}h with Graph R² = {best['graph']:.4f}")
