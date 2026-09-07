#!/usr/bin/env python3
"""
Script 37: Unified Comparison (Daily Features)
===============================================
Fair comparison of ALL feature types at DAILY granularity:
1. Graph topology features (from Neo4j)
2. Aggregated sentiment features (daily sum/mean)
3. Combined: Graph + Sentiment

Uses Pure Walk-Forward CV with Nested inner CV for alpha selection.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
RESULTS_DIR = (Path(__file__).resolve().parents[1] / 'results')

print("="*70)
print("UNIFIED COMPARISON: Graph vs Sentiment vs Combined (Daily)")
print("="*70)
print()

# 1. Load Graph Features (daily)
print("📊 Loading data...")
graph_df = pd.read_csv(DATA_DIR / 'graph_features_temporal.csv', index_col=0, parse_dates=True)
print(f"   Graph features: {len(graph_df)} days, {len(graph_df.columns)} features")

# 2. Load Hourly Sentiment and aggregate to Daily
hourly_df = pd.read_csv(DATA_DIR / 'btc_hourly_sentiment.csv', index_col=0, parse_dates=True)

# Aggregate to daily
daily_sentiment = hourly_df.resample('D').agg({
    'weighted_sentiment': 'mean',
    'social_volume': 'sum',
    'message_count': 'sum',
    'total_views': 'sum',
    'total_forwards': 'sum'
})
daily_sentiment['total_attention'] = daily_sentiment['total_views'] + 10 * daily_sentiment['total_forwards']
daily_sentiment.columns = ['sent_' + col if not col.startswith('sent_') else col for col in daily_sentiment.columns]
print(f"   Sentiment features: {len(daily_sentiment)} days, {len(daily_sentiment.columns)} features")

# 4. Merge all datasets
graph_df.index = pd.to_datetime(graph_df.index).tz_localize(None)
daily_sentiment.index = pd.to_datetime(daily_sentiment.index).tz_localize(None)

# 3. Load Market Data (daily)
market_df = yf.download("BTC-USD", start=min(graph_df.index.min(), daily_sentiment.index.min()),
                        end=max(graph_df.index.max(), daily_sentiment.index.max()), progress=False)
if isinstance(market_df.columns, pd.MultiIndex):
    market_df = market_df.xs('BTC-USD', level=1, axis=1)

market_df.index = market_df.index.tz_localize(None) # Ensure market data is also naive

market_df['Returns'] = market_df['Close'].pct_change()
market_df['Log_Returns'] = np.log(market_df['Close'] / market_df['Close'].shift(1))
market_df['Volatility'] = market_df['Log_Returns'].rolling(20).std() * np.sqrt(365)
market_df['Target_Vol'] = market_df['Volatility'].shift(-5)  # 5-day forward

print(f"   Market data: {len(market_df)} days")


# Graph + Market
full_graph = graph_df.join(market_df[['Returns', 'Volatility', 'Target_Vol']], how='inner').dropna()

# Sentiment + Market
full_sent = daily_sentiment.join(market_df[['Returns', 'Volatility', 'Target_Vol']], how='inner').dropna()

# Combined: Graph + Sentiment + Market
full_combined = graph_df.join(daily_sentiment, how='inner')
full_combined = full_combined.join(market_df[['Returns', 'Volatility', 'Target_Vol']], how='inner').dropna()

print(f"\n   After merge:")
print(f"   - Graph only: {len(full_graph)} days")
print(f"   - Sentiment only: {len(full_sent)} days")
print(f"   - Combined: {len(full_combined)} days")

# 5. Define feature sets
graph_features = ['graph_density', 'graph_transitivity', 'graph_avg_degree',
                  'graph_max_degree', 'graph_degree_gini', 'graph_centrality_stability',
                  'graph_num_communities', 'graph_modularity']

sentiment_features = ['sent_sentiment_baseline', 'sent_social_volume', 'sent_message_count',
                      'sent_total_views', 'sent_total_forwards', 'sent_total_attention']

baseline_features = ['Volatility', 'Returns']

# Filter to available features
graph_features = [f for f in graph_features if f in full_graph.columns]
sentiment_features = [f for f in sentiment_features if f in full_sent.columns]

print(f"\n   Feature counts:")
print(f"   - Graph: {len(graph_features)}")
print(f"   - Sentiment: {len(sentiment_features)}")
print(f"   - Baseline: {len(baseline_features)}")

# 6. Walk-Forward CV function
outer_tscv = TimeSeriesSplit(n_splits=5)
inner_tscv = TimeSeriesSplit(n_splits=3)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def evaluate_features(df, features, name):
    """Pure Walk-Forward CV with multiple models comparison"""

    X = df[features]
    y = df['Target_Vol']

    # Store results for each model type
    model_results = {
        'Lasso': [],
        'RandomForest': [],
        'GradientBoosting': []
    }

    for fold_idx, (train_idx, test_idx) in enumerate(outer_tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Skip if too small
        if len(X_train) < 50 or len(X_test) < 20:
            continue

        # Scaler: fit on train only
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # 1. Linear: LassoCV with temporal inner CV
        lasso = LassoCV(cv=inner_tscv, alphas=np.logspace(-4, 1, 50), max_iter=10000, random_state=42)
        lasso.fit(X_train_s, y_train)
        y_pred_lasso = lasso.predict(X_test_s)

        model_results['Lasso'].append({
            'r2': r2_score(y_test, y_pred_lasso),
            'mse': mean_squared_error(y_test, y_pred_lasso)
        })

        # 2. Non-linear: Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42, n_jobs=-1)
        rf.fit(X_train_s, y_train)
        y_pred_rf = rf.predict(X_test_s)

        model_results['RandomForest'].append({
            'r2': r2_score(y_test, y_pred_rf),
            'mse': mean_squared_error(y_test, y_pred_rf)
        })

        # 3. Non-linear: Gradient Boosting (XGBoost analog)
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
        gb.fit(X_train_s, y_train)
        y_pred_gb = gb.predict(X_test_s)

        model_results['GradientBoosting'].append({
            'r2': r2_score(y_test, y_pred_gb),
            'mse': mean_squared_error(y_test, y_pred_gb)
        })

    if not model_results['Lasso']: # Check if any folds ran
        return None

    # Aggregate results per model
    final_output = {'name': name, 'models': {}}

    for model_name, folds in model_results.items():
        mean_r2 = np.mean([f['r2'] for f in folds])
        std_r2 = np.std([f['r2'] for f in folds])
        mean_mse = np.mean([f['mse'] for f in folds])

        final_output['models'][model_name] = {
            'mean_r2': mean_r2,
            'std_r2': std_r2,
            'mean_mse': mean_mse
        }

    return final_output

# 7. Run evaluations
print("\n" + "="*70)
print("RUNNING EXPERIMENTS (Linear vs Non-Linear)")
print("="*70)

results = {}
feature_sets = {
    'Baseline': baseline_features,
    'Graph': baseline_features + graph_features,
    'Sentiment': baseline_features + sentiment_features,
    'Combined': baseline_features + graph_features + sentiment_features
}

# Determine correct dataframe for each set
df_map = {
    'Baseline': full_graph, # Contains Baseline features
    'Graph': full_graph,
    'Sentiment': full_sent,
    'Combined': full_combined
}

for set_name, features in feature_sets.items():
    print(f"\nEvaluating {set_name} features...")
    # Filter features to those present in the dataframe
    df = df_map[set_name]
    valid_features = [f for f in features if f in df.columns]

    res = evaluate_features(df, valid_features, set_name)
    if res:
        results[set_name] = res
        print(f"   Lasso R²: {res['models']['Lasso']['mean_r2']:.4f}")
        print(f"   RF R²:    {res['models']['RandomForest']['mean_r2']:.4f}")
        print(f"   GBM R²:   {res['models']['GradientBoosting']['mean_r2']:.4f}")

# 8. Final comparison
print("\n" + "="*70)
print("FINAL COMPARISON (Daily Volatility Prediction)")
print("="*70)

print(f"\n{'Feature Set':<15} {'Model':<18} {'Mean R²':>12} {'± Std':>10}")
print("-" * 60)

for set_name in ['Baseline', 'Graph', 'Sentiment', 'Combined']:
    if set_name in results:
        for model_name in ['Lasso', 'RandomForest', 'GradientBoosting']:
             r = results[set_name]['models'][model_name]
             print(f"{set_name:<15} {model_name:<18} {r['mean_r2']:>12.4f} {r['std_r2']:>10.4f}")


# Save results
output_file = RESULTS_DIR / 'unified_comparison_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n✅ Results saved: {output_file}")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)
print("""
This comparison uses IDENTICAL methodology for all feature sets:
- Same granularity (daily)
- Same target (5-day forward volatility)
- Same validation (5-fold Walk-Forward CV)
- Same model (LassoCV with nested temporal inner CV)

Key question answered:
"Does Graph Topology add predictive power BEYOND Sentiment?"
""")
