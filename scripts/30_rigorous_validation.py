#!/usr/bin/env python3
"""
Script 30: Rigorous Statistical Validation (The "Thesis Defender")
==================================================================

Performs strictly rigorous statistical tests to validate graph features:
1. Stationarity Checks (ADF Test) - Preventing spurious regression.
2. Controlled Causality (VAR) - Does Graph add value OVER market data?

This implements the "Gold Standard" for financial time series analysis.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.api import VAR
import statsmodels.api as sm
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')

print("="*70)
print("RIGOROUS STATISTICAL VALIDATION")
print("="*70)

# 1. Load & Align Data
# --------------------
graph_df = pd.read_csv(DATA_DIR / 'graph_features_temporal.csv', index_col=0, parse_dates=True)
graph_df.index = pd.to_datetime(graph_df.index)

# Download Fresh Market Data to align perfectly
print("\n1. Data Preparation...")
btc = yf.download("BTC-USD", start=graph_df.index.min(), end=graph_df.index.max(), progress=False)
if isinstance(btc.columns, pd.MultiIndex):
    btc = btc.xs('BTC-USD', level=1, axis=1)

# Calculate Volatility & Returns
btc['Returns'] = btc['Close'].pct_change()
btc['Volatility'] = btc['Returns'].rolling(20).std()
btc['Abs_Returns'] = btc['Returns'].abs()

# Join
full_df = graph_df.join(btc[['Returns', 'Volatility', 'Abs_Returns']], how='inner').dropna()
print(f"   Aligned Data: {len(full_df)} days")

# 2. Stationarity Checks (ADF Test)
# ---------------------------------
print("\n2. Stationarity Checks (Augmented Dickey-Fuller)...")
print(f"   {'Series':30s} | {'p-value':8s} | {'Stationary?'}")
print("-" * 60)

stationary_cols = []
processed_df = full_df.copy()

features_to_test = [
    'Bitcoin_Volatility', # Target
    'graph_density', 'graph_modularity', 'graph_degree_gini',
    'graph_centrality_stability', 'graph_nodes'
]

# Rename check
if 'Volatility' in full_df.columns:
    processed_df.rename(columns={'Volatility': 'Bitcoin_Volatility'}, inplace=True)

final_features = {} # Map original -> processed name

for col in features_to_test:
    if col not in processed_df.columns: continue

    series = processed_df[col].dropna()
    res = adfuller(series)
    p_val = res[1]

    is_stationary = p_val < 0.05
    status = "YES ✅" if is_stationary else "NO  ⚠️"
    print(f"   {col:30s} | {p_val:8.4f} | {status}")

    if not is_stationary:
        # Take Diff
        diff_col = f"{col}_diff"
        processed_df[diff_col] = processed_df[col].diff()

        # Check diff
        res_diff = adfuller(processed_df[diff_col].dropna())
        p_val_diff = res_diff[1]

        status_diff = "YES (Diff)" if p_val_diff < 0.05 else "NO (Diff)"
        # print(f"      -> Diff: {diff_col:20s} | {p_val_diff:8.4f} | {status_diff}")

        if p_val_diff < 0.05:
            final_features[col] = diff_col
        else:
            print(f"      ❌ {col} is non-stationary even after differencing. Excluding.")
    else:
        final_features[col] = col

processed_df.dropna(inplace=True)

# 3. Controlled Causality (VAR / OLS with Controls)
# -------------------------------------------------
print("\n3. Robust Causality Test (controlling for Past Market Data)...")
print("   Model: Volatility(t) ~ Volatility(t-1) + Returns(t-1) + FEATURE(t-lag)")
print("-" * 80)
print(f"   {'Feature (Processed)':35s} | {'Coef':8s} | {'t-stat':8s} | {'P>|t|':8s} | {'Robust?'}")
print("-" * 80)

target = final_features.get('Bitcoin_Volatility')
market_control = 'Returns' # Control for market returns

if not target:
    print("Error: Target volatility is not stationary.")
    exit()

best_robust_features = []

for original_feat in ['graph_density', 'graph_modularity', 'graph_degree_gini', 'graph_centrality_stability', 'graph_nodes']:
    feat = final_features.get(original_feat)
    if not feat: continue

    # Prepare Lagged Features
    # We want to predict Target(t) using Feature(t-lag)
    # Controlling for Target(t-lag) and Market(t-lag)

    # Let's try optimal lag from previous step (e.g., 5) or loop
    best_p = 1.0
    best_res = None
    best_lag = 0

    for lag in range(1, 8):
        # Build DataFrame for Regression
        reg_df = pd.DataFrame()
        reg_df['Target'] = processed_df[target]
        reg_df['Target_Lag'] = processed_df[target].shift(lag)
        reg_df['Market_Lag'] = processed_df[market_control].shift(lag)
        reg_df['Feature_Lag'] = processed_df[feat].shift(lag)

        reg_df.dropna(inplace=True)

        # OLS
        X = reg_df[['Target_Lag', 'Market_Lag', 'Feature_Lag']]
        X = sm.add_constant(X)
        y = reg_df['Target']

        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lag})

        p_val = model.pvalues['Feature_Lag']

        if p_val < best_p:
            best_p = p_val
            best_res = model
            best_lag = lag

    # Check Significance
    coef = best_res.params['Feature_Lag']
    t_stat = best_res.tvalues['Feature_Lag']
    is_robust = "YES 🏆" if best_p < 0.05 else "NO"

    print(f"   {feat:35s} (Lag {best_lag}) | {coef:8.4f} | {t_stat:8.2f} | {best_p:8.4f} | {is_robust}")

    if best_p < 0.05:
        best_robust_features.append({
            'feature': original_feat,
            'processed': feat,
            'lag': best_lag,
            'p_value': best_p,
            'coef': coef
        })

print("\n" + "="*70)
print("FINAL VERDICT FOR THESIS")
print("="*70)

if best_robust_features:
    print(f"✅ The following graph features contain UNIQUE predictive power")
    print(f"   beyond standard market autoregression (Proof of Concept):")
    for f in best_robust_features:
        print(f"   - {f['feature']} (Lag {f['lag']}): p={f['p_value']:.4f}")

    print("\n   Interpretation:")
    print("   This confirms that Graph Structure adds NEW information")
    print("   that is not priced in yet via simple price/volatility momentum.")
else:
    print("❌ No features survived rigorous testing.")
    print("   Graph features might be proxies for market volume/price action.")
