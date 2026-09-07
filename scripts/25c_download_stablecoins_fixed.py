#!/usr/bin/env python3
"""
Script 25c: Download Stablecoin Data (Fixed with correct DefiLlama API)
=======================================================================
Uses correct DefiLlama endpoints from official API docs.
"""

import pandas as pd
import numpy as np
import requests
import json
from pathlib import Path
from datetime import datetime

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')

print("=" * 70)
print("DOWNLOADING STABLECOIN DATA")
print("=" * 70)

depeg_events = []  # Initialize early

# ============================================================
# 1. Get current stablecoin list
# ============================================================

print("\n📊 Step 1: Getting stablecoin list...")
print("-" * 70)

stablecoins = []
try:
    url = "https://stablecoins.llama.fi/stablecoins"
    response = requests.get(url, timeout=30)

    if response.status_code == 200:
        data = response.json()
        stablecoins = data.get('peggedAssets', [])
        print(f"   ✓ Found {len(stablecoins)} stablecoins")

        # Show top 10 by circulating supply
        sorted_stable = sorted(stablecoins, key=lambda x: x.get('circulating', {}).get('peggedUSD', 0) if isinstance(x.get('circulating'), dict) else 0, reverse=True)
        print("\n   Top 10 by market cap:")
        for i, s in enumerate(sorted_stable[:10], 1):
            circ = s.get('circulating', {})
            if isinstance(circ, dict):
                supply = circ.get('peggedUSD', 0) / 1e9
            else:
                supply = 0
            print(f"   {i}. {s.get('symbol', 'N/A'):8s} - ${supply:.2f}B")
    else:
        print(f"   ⚠️ Failed: {response.status_code}")
except Exception as e:
    print(f"   ⚠️ Error: {str(e)[:50]}")

# ============================================================
# 2. Get historical total stablecoin supply
# ============================================================

print("\n📈 Step 2: Getting historical total stablecoin supply...")
print("-" * 70)

total_supply_df = pd.DataFrame()
try:
    url = "https://stablecoins.llama.fi/stablecoincharts/all"
    response = requests.get(url, timeout=30)

    if response.status_code == 200:
        data = response.json()

        records = []
        for entry in data:
            record = {'date': pd.to_datetime(int(entry['date']), unit='s')}

            # Extract totalCirculating
            if 'totalCirculating' in entry:
                tc = entry['totalCirculating']
                if isinstance(tc, dict):
                    record['total_stablecoin_supply'] = tc.get('peggedUSD', 0)
                elif isinstance(tc, (int, float)):
                    record['total_stablecoin_supply'] = tc

            records.append(record)

        total_supply_df = pd.DataFrame(records).set_index('date').sort_index()
        print(f"   ✓ Total supply history: {len(total_supply_df)} days")
        print(f"   ✓ Period: {total_supply_df.index.min()} → {total_supply_df.index.max()}")
    else:
        print(f"   ⚠️ Failed: {response.status_code}")
except Exception as e:
    print(f"   ⚠️ Error: {str(e)[:50]}")

# ============================================================
# 3. Get historical stablecoin prices (for depeg detection)
# ============================================================

print("\n💵 Step 3: Getting historical stablecoin prices (depeg detection)...")
print("-" * 70)

prices_df = pd.DataFrame()
try:
    url = "https://stablecoins.llama.fi/stablecoinprices"
    response = requests.get(url, timeout=30)

    if response.status_code == 200:
        data = response.json()

        # Convert to DataFrame
        records = []
        for entry in data:
            record = {'date': pd.to_datetime(int(entry['date']), unit='s')}
            if 'prices' in entry:
                for symbol, price in entry['prices'].items():
                    if price is not None:
                        record[f'price_{symbol}'] = price
            records.append(record)

        prices_df = pd.DataFrame(records).set_index('date').sort_index()

        print(f"   ✓ Price history: {len(prices_df)} days")
        print(f"   ✓ Stablecoins tracked: {len([c for c in prices_df.columns if c.startswith('price_')])}")

        # Detect major depegs (price < 0.95 or > 1.05)
        for col in prices_df.columns:
            if col.startswith('price_'):
                symbol = col.replace('price_', '')
                series = prices_df[col].dropna()
                depegs = series[(series < 0.95) | (series > 1.05)]
                if len(depegs) > 0:
                    for date, price in depegs.items():
                        depeg_events.append({
                            'date': date,
                            'symbol': symbol,
                            'price': price,
                            'deviation': abs(price - 1.0)
                        })

        if depeg_events:
            depeg_df = pd.DataFrame(depeg_events).sort_values('deviation', ascending=False)
            print(f"\n   🚨 Major depeg events detected: {len(depeg_events)}")
            print("   Top 5 depegs:")
            for _, row in depeg_df.head(5).iterrows():
                print(f"      {row['date'].strftime('%Y-%m-%d')} - {row['symbol']}: ${row['price']:.4f}")
    else:
        print(f"   ⚠️ Failed: {response.status_code}")
except Exception as e:
    print(f"   ⚠️ Error: {str(e)[:50]}")

# ============================================================
# 4. Get chain-specific stablecoin data (Ethereum)
# ============================================================

print("\n🔗 Step 4: Getting Ethereum stablecoin supply...")
print("-" * 70)

eth_supply_df = pd.DataFrame()
try:
    url = "https://stablecoins.llama.fi/stablecoincharts/Ethereum"
    response = requests.get(url, timeout=30)

    if response.status_code == 200:
        data = response.json()

        records = []
        for entry in data:
            record = {'date': pd.to_datetime(int(entry['date']), unit='s')}
            if 'totalCirculating' in entry:
                tc = entry['totalCirculating']
                if isinstance(tc, dict):
                    record['eth_stablecoin_supply'] = tc.get('peggedUSD', 0)
            records.append(record)

        eth_supply_df = pd.DataFrame(records).set_index('date').sort_index()
        print(f"   ✓ Ethereum stablecoin supply: {len(eth_supply_df)} days")
    else:
        print(f"   ⚠️ Failed: {response.status_code}")
except Exception as e:
    print(f"   ⚠️ Error: {str(e)[:50]}")

# ============================================================
# 5. Combine all stablecoin data
# ============================================================

print("\n🔗 Step 5: Combining stablecoin data...")
print("-" * 70)

# Start with total supply
combined = total_supply_df.copy() if len(total_supply_df) > 0 else pd.DataFrame()

# Add Ethereum supply
if len(eth_supply_df) > 0:
    combined = combined.join(eth_supply_df, how='outer')

# Add prices
if len(prices_df) > 0:
    # Keep only major stablecoins prices
    major_stables = ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'FRAX', 'USDD', 'LUSD']
    price_cols = [f'price_{s}' for s in major_stables if f'price_{s}' in prices_df.columns]
    if price_cols:
        combined = combined.join(prices_df[price_cols], how='outer')

# Calculate derived metrics
if 'total_stablecoin_supply' in combined.columns:
    combined['stablecoin_supply_change_1d'] = combined['total_stablecoin_supply'].pct_change()
    combined['stablecoin_supply_change_7d'] = combined['total_stablecoin_supply'].pct_change(7)
    combined['stablecoin_supply_ma7'] = combined['total_stablecoin_supply'].rolling(7).mean()

# Calculate depeg score (average deviation from $1.00)
price_cols = [c for c in combined.columns if c.startswith('price_')]
if price_cols:
    combined['avg_depeg_score'] = combined[price_cols].apply(lambda x: abs(x - 1.0).mean(), axis=1)
    combined['max_depeg'] = combined[price_cols].apply(lambda x: abs(x - 1.0).max(), axis=1)

print(f"   ✓ Combined shape: {combined.shape}")

# ============================================================
# 6. Save data
# ============================================================

print("\n💾 Step 6: Saving data...")
print("-" * 70)

if len(combined) > 0:
    output_file = DATA_DIR / 'stablecoin_data.csv'
    combined.to_csv(output_file)

    print(f"   ✓ Saved: {output_file.name}")
    print(f"   ✓ Size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"   ✓ Rows: {len(combined)}")
    print(f"   ✓ Columns: {len(combined.columns)}")
else:
    print("   ⚠️ No data to save")

# Also save depeg events for analysis
if depeg_events:
    depeg_output = DATA_DIR / 'stablecoin_depeg_events.csv'
    pd.DataFrame(depeg_events).to_csv(depeg_output, index=False)
    print(f"   ✓ Depeg events: {depeg_output.name} ({len(depeg_events)} events)")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("STABLECOIN DATA DOWNLOAD COMPLETE")
print("=" * 70)

if len(combined) > 0:
    print("\n📊 Available metrics:")
    for col in combined.columns[:15]:  # Show first 15
        print(f"   • {col}")
    if len(combined.columns) > 15:
        print(f"   ... and {len(combined.columns) - 15} more")

print("\n🎯 Use cases:")
print("   • total_stablecoin_supply → Market liquidity indicator")
print("   • stablecoin_supply_change → Flow signals (bullish/bearish)")
print("   • price_* → Depeg detection (USDC March 2023, UST May 2022)")
print("   • avg_depeg_score → Systemic stress indicator")
