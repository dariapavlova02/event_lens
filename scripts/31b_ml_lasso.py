#!/usr/bin/env python3
"""
Script 31b: ML Regression with LASSO (Linear vs Non-Linear)
===========================================================
Boosting overfitted. Let's try LASSO (Linear + Regularization).
Since our robust regression (OLS) showed significance, LASSO should work better.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')

print("="*70)
print("ML REGRESSION: LASSO (Reduced Overfitting)")
print("="*70)

# Load Data (Same as before)
graph_df = pd.read_csv(DATA_DIR / 'graph_features_temporal.csv', index_col=0, parse_dates=True)
market_df = yf.download("BTC-USD", start=graph_df.index.min(), end=graph_df.index.max(), progress=False)
if isinstance(market_df.columns, pd.MultiIndex):
    market_df = market_df.xs('BTC-USD', level=1, axis=1)

graph_df.index = pd.to_datetime(graph_df.index)
market_df['Returns'] = market_df['Close'].pct_change()
market_df['Log_Returns'] = np.log(market_df['Close'] / market_df['Close'].shift(1))
market_df['Volatility'] = market_df['Log_Returns'].rolling(20).std() * np.sqrt(365)
market_df['Target_Vol'] = market_df['Volatility'].shift(-5)

full_df = graph_df.join(market_df[['Returns', 'Volatility', 'Target_Vol']], how='inner').dropna()

baseline_features = ['Volatility', 'Returns', 'graph_nodes', 'graph_edges']
graph_features = baseline_features + ['graph_density', 'graph_transitivity', 'graph_degree_gini', 'graph_centrality_stability', 'graph_modularity']

tscv = TimeSeriesSplit(n_splits=5)

def evaluate_linear(features, name):
    mse_scores, r2_scores = [], []
    X, y = full_df[features], full_df['Target_Vol']

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # LassoCV automatically finds best alpha
        model = LassoCV(cv=5, random_state=42)
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    print(f"\nModel: {name}")
    print(f"   MSE: {np.mean(mse_scores):.6f}")
    print(f"   R2 (OOS): {np.mean(r2_scores):.4f}")

    return np.mean(mse_scores), np.mean(r2_scores)

mse_base, r2_base = evaluate_linear(baseline_features, "Baseline (Lasso)")
mse_graph, r2_graph = evaluate_linear(graph_features, "Graph-Enhanced (Lasso)")

improvement = (mse_base - mse_graph) / mse_base
print("\n" + "="*50)
print(f"Lasso Improvement: {improvement:.2%}")
print(f"R2 Gain: {r2_graph - r2_base:.4f}")
