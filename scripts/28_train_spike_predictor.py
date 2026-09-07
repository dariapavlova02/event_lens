#!/usr/bin/env python3
"""
Script 28: Train Volatility Spike Predictor (Graph vs Baseline)
===============================================================

Trains classifiers to predict volatility regime changes/spikes.
Compares models using only Traditional Features vs Graph Features.

Target: Volatility Spike > 2 sigma (Binary Classification)
Validation: Rolling Walk-Forward (Time Series Split)

Models:
1. Baseline: Social Volume + Sentiment
2. Graph-Enhanced: + Modularity, Centrality Gini, Stability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.inspection import permutation_importance

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
RESULTS_DIR = (Path(__file__).resolve().parents[1] / 'results')

print("=" * 70)
print("TRAINING VOLATILITY SPIKE PREDICTOR")
print("=" * 70)

# ============================================================
# 1. Load Data
# ============================================================

# Graph features (Daily)
print("\nLoading graph features...")
graph_df = pd.read_csv(DATA_DIR / 'graph_features_temporal.csv', index_col=0, parse_dates=True)
graph_df.index = pd.to_datetime(graph_df.index)

# Market data
print("Loading market data...")
# Market data has integer index issue, let's reconstruct dates
market_raw = pd.read_csv(DATA_DIR / 'market_data_multi_entity.csv') # No index col first

# Recover dates from graph_df or TBSA
# Assuming market data aligns with our period.
# Better approach: Download BTC data fresh for this period or try to parse 'Date' if exists
import yfinance as yf
print("   Downloading fresh BTC data to align dates...")
btc = yf.download("BTC-USD", start=graph_df.index.min(), end=graph_df.index.max() + pd.Timedelta(days=5), progress=False)

if isinstance(btc.columns, pd.MultiIndex):
    btc = btc.xs('BTC-USD', level=1, axis=1) # Handle yfinance multi-index

btc['returns'] = btc['Close'].pct_change()
btc['volatility'] = btc['returns'].rolling(20).std()
mean_vol = btc['volatility'].mean()
std_vol = btc['volatility'].std()
btc['btc_vol_spike_2sigma'] = (btc['volatility'] > (mean_vol + 2 * std_vol)).astype(int)

market_df = btc[['volatility', 'btc_vol_spike_2sigma']].copy()
market_df = market_df.rename(columns={'volatility': 'Bitcoin_volatility'})

# Align
data = graph_df.join(market_df, how='inner') # Inner join on Date index

# Shift features (Predict NEXT day spike)
# X(t) predicts y(t+1)
target_col = 'btc_vol_spike_2sigma'
data['target'] = data[target_col].shift(-1)
data.dropna(inplace=True)

print(f"Data shape: {data.shape}")
print(f"Spike prevalence: {data['target'].mean():.2%}")

# ============================================================
# 2. Feature Sets
# ============================================================

# Baseline Features (Volume-based)
baseline_features = ['graph_nodes', 'graph_edges', 'graph_avg_degree']

# Graph Structural Features (The "Unique" Contribution)
graph_features = [
    'graph_density',
    'graph_transitivity',
    'graph_degree_gini',
    'graph_centrality_stability',
    'graph_num_communities',
    'graph_modularity',
    'graph_assortativity'
]

all_features = baseline_features + graph_features

# ============================================================
# 3. Model Training & Walk-Forward Validation
# ============================================================

def train_evaluate(features, name):
    print(f"\nTraining {name} Model ({len(features)} features)...")

    X = data[features]
    y = data['target']

    tscv = TimeSeriesSplit(n_splits=5)

    scores = {'precision': [], 'recall': [], 'f1': [], 'auc': []}

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # Metrics
        scores['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        scores['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        scores['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        try:
            scores['auc'].append(roc_auc_score(y_test, y_prob))
        except:
            scores['auc'].append(0.5)

    print(f"   AVG F1: {np.mean(scores['f1']):.3f}")
    print(f"   AVG AUC: {np.mean(scores['auc']):.3f}")

    # Train on full dataset for feature importance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model.fit(X_scaled, y)

    return np.mean(scores['f1']), np.mean(scores['auc']), model

# Run Benchmark
f1_base, auc_base, model_base = train_evaluate(baseline_features, "BASELINE (Volume)")
f1_graph, auc_graph, model_graph = train_evaluate(all_features, "GRAPH-ENHANCED")

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"Baseline AUC: {auc_base:.3f}")
print(f"Graph AUC:    {auc_graph:.3f}")
print(f"Improvement:  {((auc_graph - auc_base) / auc_base):.1%}")

# ============================================================
# 4. Feature Importance
# ============================================================

print("\nFeature Importance (Graph Model):")
importances = model_graph.feature_importances_
indices = np.argsort(importances)[::-1]

top_features = []
for f in range(X_train.shape[1]):
    idx = indices[f]
    print(f"   {f+1}. {all_features[idx]:30s} {importances[idx]:.4f}")
    top_features.append({'feature': all_features[idx], 'importance': importances[idx]})

# Save Importance Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=pd.DataFrame(top_features))
plt.title('Feature Importance for Volatility Prediction')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'feature_importance_graph.png')
print(f"\nSaved feature importance plot to {RESULTS_DIR}/feature_importance_graph.png")

# Save detailed results
results = {
    'baseline': {'f1': f1_base, 'auc': auc_base},
    'graph': {'f1': f1_graph, 'auc': auc_graph},
    'top_features': top_features
}
import json
with open(DATA_DIR / 'prediction_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nDONE. Ready for thesis!")
