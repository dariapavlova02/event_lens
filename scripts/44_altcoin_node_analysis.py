#!/usr/bin/env python3
"""
Script 44: Altcoin Node-Level Analysis (ETH & SOL)
==================================================
Tests if specific node centrality metrics for Ethereum and Solana
predict their respective volatility better than generic global density.

Hypothesis:
Altcoins have more specific on-chain/social activity (e.g. DeFi, NFT mints)
that might be better captured by their specific node centrality than generic BTC-dominated graph metrics.
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
HORIZON = 1  # 1 hour ahead predictive horizon

def load_and_prep_data(coin_name, node_col):
    print(f"\n💎 Processing {coin_name}...")

    # 1. Market Data
    market = pd.read_csv(DATA_DIR / 'altcoin_market_hourly.csv', index_col=0, parse_dates=True)
    # market.index is already timezone-aware from yfinance (UTC).
    # ensure it's timezone-naive for easier merging with graph data if needed, or keep aware.
    # Graph data usually comes in as UTC. Let's make both timezone-naive to avoid issues.
    if market.index.tz is not None:
        market.index = market.index.tz_localize(None)

    # Select specific columns for this coin
    vol_col = f'{coin_name}_volatility'
    ret_col = f'{coin_name}_returns'

    if vol_col not in market.columns:
        print(f"❌ Missing volatility data for {coin_name}")
        return None

    # 2. Graph Global
    global_graph = pd.read_csv(DATA_DIR / 'graph_features_hourly.csv', index_col=0, parse_dates=True)
    global_graph.index = global_graph.index.floor('H').tz_localize(None)
    global_graph = global_graph[~global_graph.index.duplicated(keep='last')]

    # 3. Node Specific
    node_graph = pd.read_csv(DATA_DIR / 'node_features_hourly.csv', index_col=0, parse_dates=True)
    node_graph.index = pd.to_datetime(node_graph.index).floor('H').tz_localize(None)
    node_graph = node_graph[~node_graph.index.duplicated(keep='last')]

    # Select specific node centrality
    if node_col not in node_graph.columns:
        print(f"❌ Missing node feature {node_col}")
        return None

    # Merge
    df = market[[vol_col, ret_col]].join(global_graph, how='inner')
    df = df.join(node_graph[[node_col, 'max_degree_centrality']], how='inner', rsuffix='_agg')

    # Target: Future Absolute Return (Instantaneous Volatility)
    # This removes autocorrelation bias. If graph predicts this, it's real alpha.
    df['Target_AbsRet'] = df[ret_col].abs().shift(-HORIZON)

    # Baseline Features: Current volatility proxy
    df['Current_AbsRet'] = df[ret_col].abs()
    df['Vol_24h'] = df[vol_col] # Keep rolling vol as a feature (past info), not target

    df = df.dropna()

    print(f"   Samples: {len(df)}")

    return df, 'Current_AbsRet', 'Vol_24h', node_col

def evaluate_models(df, coin_name, abs_ret_col, vol_feat_col, node_col):
    features = {
        'Baseline': [abs_ret_col, vol_feat_col],
        'Global Graph': [abs_ret_col, vol_feat_col] + ['graph_density', 'graph_modularity'],
        'Specific Node': [abs_ret_col, vol_feat_col, node_col],
        'Combined': [abs_ret_col, vol_feat_col, node_col, 'graph_density']
    }

    # Check if global cols exist, otherwise use all from global df except excluded
    global_cols = [c for c in df.columns if c not in [abs_ret_col, vol_feat_col, node_col, 'max_degree_centrality', 'Target_AbsRet']]
    features['Global Graph'] = [abs_ret_col, vol_feat_col] + global_cols[:5] # Take top 5 global stats

    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    for set_name, feats in features.items():
        # Ensure features exist
        valid_feats = [f for f in feats if f in df.columns]

        X = df[valid_feats]
        y = df['Target_AbsRet']

        r2_scores = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = LassoCV(cv=TimeSeriesSplit(3), random_state=42)
            model.fit(X_train_s, y_train)

            pred = model.predict(X_test_s)
            r2_scores.append(r2_score(y_test, pred))

        results[set_name] = np.mean(r2_scores)

    return results

def main():
    print("="*60)
    print("ALTCOIN STRICT PREDICTION CHECK (Target: Next Hour Abs Return)")
    print("="*60)

    coins = [
        ('Ethereum', 'deg_Ethereum'),
        ('Solana', 'deg_Solana')
    ]

    final_summary = []

    for coin, node_feat in coins:
        data_tuple = load_and_prep_data(coin, node_feat)
        if not data_tuple:
            continue

        df, abs_ret, vol_feat, n_col = data_tuple
        res = evaluate_models(df, coin, abs_ret, vol_feat, n_col)

        print(f"\n📊 Results for {coin} (R² on AbsRet):")
        print(f"{'Feature Set':<20} {'R² Score':<10} {'Delta vs Base':<10}")
        print("-" * 45)

        base = res['Baseline']
        for k, v in res.items():
            delta = v - base
            print(f"{k:<20} {v:.4f}     {delta:+.4f}")

        node_delta = res['Specific Node'] - base
        final_summary.append({
            'Coin': coin,
            'Baseline': base,
            'Node_R2': res['Specific Node'],
            'Lift': node_delta
        })

    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    for item in final_summary:
        status = "✅ LIFT" if item['Lift'] > 0.001 else "❌ NO LIFT"
        print(f"{item['Coin']:<10} | Base: {item['Baseline']:.4f} | Node: {item['Node_R2']:.4f} | {status} ({item['Lift']:+.4f})")

if __name__ == "__main__":
    main()
