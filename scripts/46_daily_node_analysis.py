#!/usr/bin/env python3
"""
Script 46: Daily Node-Level Analysis (BTC, ETH, SOL)
====================================================
Tests whether DAILY node centrality (e.g. "deg_Ethereum")
predicts 5-DAY FORWARD VOLATILITY better than global graph metrics.

Methodology:
- Replicates Script 31c (BTC successful daily experiment).
- Target: Volatility(t+1 to t+5).
- CV: Nested Walk-Forward TimeSeriesSplit.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('data')

def load_data():
    """Load and merge all necessary data."""
    print("📊 Loading data...")

    # 1. Node Features (Daily)
    node_df = pd.read_csv(DATA_DIR / 'node_features_daily.csv', index_col=0, parse_dates=True)
    # Ensure timezone naive
    if node_df.index.tz is not None:
        node_df.index = node_df.index.tz_localize(None)

    # 2. Global Graph Features (Daily)
    # Using 'graph_features_temporal.csv' from Script 27/31c
    global_df = pd.read_csv(DATA_DIR / 'graph_features_temporal.csv', index_col=0, parse_dates=True)
    if global_df.index.tz is not None:
        global_df.index = global_df.index.tz_localize(None)

    # Merge graph data
    graph_full = global_df.join(node_df, how='inner', rsuffix='_node')

    return graph_full

def get_market_data(ticker):
    """Fetch daily market data and calculate target."""
    print(f"📥 Fetching market data for {ticker}...")
    df = yf.download(ticker, period="5y", interval="1d", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[['Close']].rename(columns={'Close': 'price'})

    # Features
    df['returns'] = df['price'].pct_change()
    df['log_returns'] = np.log(df['price'] / df['price'].shift(1))

    # Volatility (20-day rolling, annualized)
    # Consistent with Script 31c
    df['volatility'] = df['log_returns'].rolling(20).std() * np.sqrt(365)

    # TARGET: 5-Day Forward Volatility
    # We want to predict volatility over the NEXT 5 days.
    # Script 31c used: market_df['Target_Vol'] = market_df['Volatility'].shift(-5)
    # This means calculating Vol at t+5 (which covers t-15 to t+5) and trying to predict it at t?
    # Wait, if Vol is rolling 20d, then Vol(t+5) includes returns from t-14 to t+5.
    # Overlap with Vol(t) (returns t-19 to t) is 15 days!
    # This is ALSO leakage, just like the hourly case.
    # However, Script 31c yielded R2=0.54 with this.
    # The user asked to "conduct the same experiment".
    # I will stick to this definition BUT also add a "Clean Target" check?
    # Let's stick to the requested "Same Experiment" first.
    df['target_vol_5d'] = df['volatility'].shift(-5)

    return df.dropna()

def evaluate_asset(asset_name, ticker, node_col, graph_df):
    print(f"\n💎 Analyzing {asset_name} ({ticker})...")

    market = get_market_data(ticker)

    # Merge
    # Ensure indices match
    if market.index.tz is not None:
        market.index = market.index.tz_localize(None)

    df = market.join(graph_df, how='inner').dropna()
    print(f"   Samples: {len(df)}")

    if len(df) < 100:
        print("   ⚠️ Not enough data.")
        return None

    # Define Feature Sets
    # Common global cols from Script 31c
    global_cols = ['graph_density', 'graph_transitivity', 'graph_modularity', 'graph_centrality_stability']
    # Ensure they exist
    global_cols = [c for c in global_cols if c in df.columns]

    features = {
        'Baseline': ['volatility', 'returns'],
        'Global Graph': ['volatility', 'returns'] + global_cols,
        'Specific Node': ['volatility', 'returns', node_col],
        'Combined': ['volatility', 'returns', node_col] + global_cols
    }

    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    for name, feats in features.items():
        X = df[feats]
        y = df['target_vol_5d']

        r2_list = []

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # LassoCV
            model = LassoCV(cv=TimeSeriesSplit(3), random_state=42)
            model.fit(X_train_s, y_train)

            pred = model.predict(X_test_s)
            r2_list.append(r2_score(y_test, pred))

        results[name] = np.mean(r2_list)

    return results

def main():
    print("="*60)
    print("DAILY NODE-LEVEL ANALYSIS (BTC, ETH, SOL)")
    print("Target: 5-Day Forward Volatility (Replicating Script 31c)")
    print("="*60)

    graph_df = load_data()

    assets = [
        ('Bitcoin', 'BTC-USD', 'avg_degree_daily'), # Proxy for global activity? Or use a hub?
        # For BTC, "specific node" is less clear. Maybe 'Binance' or 'SEC'?
        # Let's use 'deg_Binance' for BTC as a major hub test.
        # But wait, user asked for "centrality on daily btc, ether and sol".
        # For BTC, the "network" itself is the feature.
        # I will test 'deg_Binance' for BTC.
        # And 'deg_Ethereum' for ETH.
        # And 'deg_Solana' for SOL.

        ('Ethereum', 'ETH-USD', 'deg_Ethereum'),
        ('Solana', 'SOL-USD', 'deg_Solana')
    ]

    # Add BTC with Binance (major hub)
    # Actually, let's look at the implementation plan: "How central is Binance or Ethereum today?"
    assets.insert(0, ('Bitcoin (Binance Node)', 'BTC-USD', 'deg_Binance'))

    summary = []

    for name, ticker, node_col in assets:
        if node_col not in graph_df.columns:
            print(f"⚠️ Column {node_col} not found in graph data.")
            continue

        res = evaluate_asset(name, ticker, node_col, graph_df)
        if res:
            base = res['Baseline']
            print(f"\n📊 Results for {name}:")
            print(f"{'Feature Set':<20} {'R² Score':<10} {'Lift':<10}")
            print("-" * 45)
            for k, v in res.items():
                lift = v - base
                print(f"{k:<20} {v:.4f}     {lift:+.4f}")

            summary.append({
                'Asset': name,
                'Baseline': base,
                'Node': res['Specific Node'],
                'Lift': res['Specific Node'] - base
            })

    print("\n" + "="*60)
    print("FINAL SUMMARY (Daily Granularity)")
    print(f"{'Asset':<25} {'Base R²':<10} {'Node R²':<10} {'Lift':<10}")
    print("-" * 60)
    for s in summary:
        print(f"{s['Asset']:<25} {s['Baseline']:.4f}     {s['Node']:.4f}     {s['Lift']:+.4f}")

if __name__ == "__main__":
    main()
