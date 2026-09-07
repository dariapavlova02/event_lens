#!/usr/bin/env python3
"""
Script 39: Hourly Comparison (Linear vs Non-linear)
===================================================
Tests if non-linear models (RF, XGBoost) can extract signal from:
1. Hourly Sentiment (noisy but abundant)
2. Daily Graph (interpolated to hourly)

Hypothesis:
RF/XGBoost might find complex interactions that Lasso missed,
leveraging the larger sample size (N=6700 vs N=1400).
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
RESULTS_DIR = (Path(__file__).resolve().parents[1] / 'results')

print("="*70)
print("HOURLY COMPARISON: Can RF/XGBoost beat the Baseline?")
print("="*70)

# 1. Load Hourly Features
print("📊 Loading data...")
features_df = pd.read_csv(DATA_DIR / 'btc_enhanced_features.csv', index_col=0, parse_dates=True)
features_df.index = features_df.index.tz_localize(None)

# Load Market Data (for target)
market_df = pd.read_csv(DATA_DIR / 'btc_market_merged.csv', index_col=0, parse_dates=True)
market_df.index = market_df.index.tz_localize(None)

# 2. Load Hourly Graph Features (NEW - real hourly data from Neo4j)
graph_hourly = pd.read_csv(DATA_DIR / 'graph_features_hourly.csv', index_col=0, parse_dates=True)
# Round to hourly precision (graph has :55:03 minutes, market has :00:00)
graph_hourly.index = graph_hourly.index.floor('H')
graph_hourly.index = graph_hourly.index.tz_localize(None)
# De-duplicate after rounding (keep last value per hour)
graph_hourly = graph_hourly[~graph_hourly.index.duplicated(keep='last')]
graph_hourly.columns = ['graph_' + col if not col.startswith('graph_') else col for col in graph_hourly.columns]

print(f"   Hourly Graph Features: {len(graph_hourly)} rows")

# 3. Merge Sentiment + Market + Graph
# Select and rename market columns
market_subset = market_df[['volatility', 'price_return_1h']].rename(columns={'price_return_1h': 'returns'})

# First join market data
full_df = features_df.join(market_subset, how='inner')
# Then join graph features
full_df = full_df.join(graph_hourly, how='inner').dropna()

# Create forward target (5-day = 120 hours, same as daily experiment)
full_df['Target_Vol_5d'] = full_df['volatility'].shift(-120)
full_df = full_df.dropna()

print(f"   Total samples: {len(full_df)}")

# 4. Define features
# Filter out target columns and market raw data from features
features_sent = [c for c in features_df.columns if c not in ['Close', 'log_price', 'returns', 'volatility', 'Target_Vol_24h']]
features_graph = [c for c in graph_hourly.columns]
features_baseline = ['volatility', 'returns']

# Ensure no overlap
features_sent = [f for f in features_sent if f not in features_baseline]

# 6. Evaluation Function
outer_tscv = TimeSeriesSplit(n_splits=5)
# Reduce inner CV size for speed on large N
inner_tscv = TimeSeriesSplit(n_splits=3)

def evaluate_models_hourly(df, features, name):
    X = df[features]
    y = df['Target_Vol_5d']

    results = {'Lasso': [], 'RF': [], 'GBM': []}

    for fold_idx, (train_idx, test_idx) in enumerate(outer_tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scaling
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # 1. Lasso (with temporal inner CV to prevent look-ahead bias)
        inner_tscv = TimeSeriesSplit(n_splits=3)
        lasso = LassoCV(cv=inner_tscv, random_state=42).fit(X_train_s, y_train)
        y_pred = lasso.predict(X_test_s)
        results['Lasso'].append(r2_score(y_test, y_pred))

        # 2. Random Forest (Optimized for speed)
        rf = RandomForestRegressor(n_estimators=50, max_depth=7, min_samples_leaf=20, n_jobs=-1, random_state=42)
        rf.fit(X_train_s, y_train)
        results['RF'].append(r2_score(y_test, rf.predict(X_test_s)))

        # 3. GBM
        gb = GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
        gb.fit(X_train_s, y_train)
        results['GBM'].append(r2_score(y_test, gb.predict(X_test_s)))

    # Aggregate
    final = {}
    for model, scores in results.items():
        final[model] = {
            'mean_r2': np.mean(scores),
            'std_r2': np.std(scores)
        }
    return final

# 7. Run Comparison
print("\n" + "="*50)
print("RUNNING HOURLY EXPERIMENTS")
print("="*50)

print("\n1. Baseline (Past Vol + Returns)")
res_base = evaluate_models_hourly(full_df, features_baseline, "Baseline")
print(f"   Lasso: {res_base['Lasso']['mean_r2']:.4f}")
print(f"   RF:    {res_base['RF']['mean_r2']:.4f}")

print("\n2. Sentiment (68 hourly features)")
res_sent = evaluate_models_hourly(full_df, features_baseline + features_sent, "Sentiment")
print(f"   Lasso: {res_sent['Lasso']['mean_r2']:.4f}")
print(f"   RF:    {res_sent['RF']['mean_r2']:.4f}")

print("\n3. Graph (Interpolated Daily)")
res_graph = evaluate_models_hourly(full_df, features_baseline + features_graph, "Graph")
print(f"   Lasso: {res_graph['Lasso']['mean_r2']:.4f}")
print(f"   RF:    {res_graph['RF']['mean_r2']:.4f}")

print("\n4. Combined")
res_comb = evaluate_models_hourly(full_df, features_baseline + features_sent + features_graph, "Combined")
print(f"   Lasso: {res_comb['Lasso']['mean_r2']:.4f}")
print(f"   RF:    {res_comb['RF']['mean_r2']:.4f}")
print(f"   GBM:   {res_comb['GBM']['mean_r2']:.4f}")

# Final Table
print("\n" + "="*50)
print("FINAL HOURLY RESULTS")
print("="*50)
print(f"{'Feature Set':<15} {'Lasso':<10} {'RF':<10} {'GBM':<10}")
print("-" * 45)
for name, res in [('Baseline', res_base), ('Sentiment', res_sent), ('Graph', res_graph), ('Combined', res_comb)]:
    print(f"{name:<15} {res['Lasso']['mean_r2']:<10.4f} {res['RF']['mean_r2']:<10.4f} {res['GBM']['mean_r2']:<10.4f}")
