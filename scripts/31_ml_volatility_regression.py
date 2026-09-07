#!/usr/bin/env python3
"""
Script 31: ML Regression for Volatility Prediction (Baseline vs Graph)
======================================================================
Predicts continuous Bitcoin volatility (next 5 days average).
Compares R-squared of Baseline (Price/Vol only) vs Graph-Enhanced models.

Models: Gradient Boosting (XGBoost/LightGBM style using sklearn HistGradientBoosting)
Validation: 5-fold Time Series Split (Strictly Out-of-Sample)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from statsmodels.tsa.api import VAR
import yfinance as yf

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
RESULTS_DIR = (Path(__file__).resolve().parents[1] / 'results')

print("="*70)
print("ML REGRESSION: VOLATILITY PREDICTION COMPARISON")
print("="*70)

# 1. Load Data
# --------------------
print("Loading data...")
graph_df = pd.read_csv(DATA_DIR / 'graph_features_temporal.csv', index_col=0, parse_dates=True)
market_df = yf.download("BTC-USD", start=graph_df.index.min(), end=graph_df.index.max(), progress=False)

if isinstance(market_df.columns, pd.MultiIndex):
    market_df = market_df.xs('BTC-USD', level=1, axis=1)

# Feature Engineering
graph_df.index = pd.to_datetime(graph_df.index)
market_df['Returns'] = market_df['Close'].pct_change()
market_df['Log_Returns'] = np.log(market_df['Close'] / market_df['Close'].shift(1))
market_df['Volatility'] = market_df['Log_Returns'].rolling(20).std() * np.sqrt(365) # Annualized

# Target: Next 5 days average volatility (smooth out noise)
market_df['Target_Vol'] = market_df['Volatility'].shift(-5)

# Join
full_df = graph_df.join(market_df[['Returns', 'Volatility', 'Target_Vol']], how='inner').dropna()

# 2. Define Feature Sets
# ----------------------
# Baseline: Just Market Data + Simple Volume (what a trader sees)
baseline_features = [
    'Volatility',
    'Returns',
    'graph_nodes', # Proxy for Volume
    'graph_edges'
]

# Graph: Adding our unique structural signals
graph_features = baseline_features + [
    'graph_density',
    'graph_transitivity',
    'graph_degree_gini',
    'graph_centrality_stability',
    'graph_modularity'
]

print(f"Data Samples: {len(full_df)}")
print(f"Features Baseline: {len(baseline_features)}")
print(f"Features Graph: {len(graph_features)}")

# 3. Train & Evaluate (Walk-Forward)
# ----------------------------------
tscv = TimeSeriesSplit(n_splits=5)

results = []

def evaluate_model(features, name):
    mse_scores = []
    mae_scores = []
    r2_scores = []

    X = full_df[features]
    y = full_df['Target_Vol']

    # Store predictions for plotting
    all_preds = []
    all_dates = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Model
        model = HistGradientBoostingRegressor(max_iter=100, max_depth=5, random_state=42)
        model.fit(X_train_s, y_train)

        # Predict
        y_pred = model.predict(X_test_s)

        # Metrics
        mse_scores.append(mean_squared_error(y_test, y_pred))
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

        all_preds.extend(y_pred)
        all_dates.extend(full_df.index[test_idx])

    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)

    print(f"\nModel: {name}")
    print(f"   MSE: {avg_mse:.6f}")
    print(f"   MAE: {np.mean(mae_scores):.6f}")
    print(f"   R2 (OOS): {avg_r2:.4f}")

    return {
        'name': name,
        'mse': avg_mse,
        'r2': avg_r2,
        'mae': np.mean(mae_scores),
        'dates': all_dates,
        'preds': all_preds
    }

res_base = evaluate_model(baseline_features, "Baseline (Market + Volume)")
res_graph = evaluate_model(graph_features, "Graph-Enhanced")

# 4. Visualization & Comparison
# -----------------------------
improvement = (res_base['mse'] - res_graph['mse']) / res_base['mse']
print("\n" + "="*50)
print(f"RESULTS: Graph Feature Value-Add")
print("="*50)
print(f"MSE Improvement: {improvement:.2%} (Lower is better)")
print(f"R2 Gap: {res_graph['r2'] - res_base['r2']:.4f}")

# Plot Prediction Comparison (Last Fold)
plt.figure(figsize=(12, 6))

# Get last N points (test set size approx)
n_points = len(res_base['dates'])
dates = res_base['dates']

# Align indices
true_vals = full_df.loc[dates, 'Target_Vol']

plt.plot(dates, true_vals, label='Actual Volatility', color='black', alpha=0.5, linewidth=2)
plt.plot(dates, res_base['preds'], label=f"Baseline (R2={res_base['r2']:.2f})", linestyle='--', color='blue')
plt.plot(dates, res_graph['preds'], label=f"Graph-Enhanced (R2={res_graph['r2']:.2f})", color='red', linewidth=1.5)

plt.title(f"Volatility Prediction: Baseline vs Graph-Enhanced (MSE Improved by {improvement:.1%})")
plt.ylabel('BTC Volatility (5-day fwd)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

outfile = RESULTS_DIR / 'ml_model_comparison.png'
plt.savefig(outfile, dpi=300)
print(f"\nSaved plot to {outfile}")

# 5. Impulse Response Function (VAR Analysis Bonus)
# -------------------------------------------------
print("\ngenerating Impulse Response (VAR)...")
try:
    # Select key variables
    var_cols = ['Volatility', 'Returns', 'graph_density']
    var_data = full_df[var_cols].diff().dropna() # Use diff for stationarity

    model = VAR(var_data)
    results = model.fit(maxlags=7)

    irf = results.irf(10)

    plt.figure(figsize=(10, 6))
    irf.plot(orth=True, impulse='graph_density', response='Volatility', figsize=(10,6))
    plt.title("Impact of Graph Density Shock on Volatility (VAR IRF)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'var_irf_density_volatility.png')
    print("Saved IRF plot.")

except Exception as e:
    print(f"VAR failed: {e}")
