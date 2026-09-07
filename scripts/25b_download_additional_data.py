#!/usr/bin/env python3
"""
Script 25b: Download Additional Market Data
============================================
Downloads DeFi TVL, stablecoin supply, and creates known events file.

Data sources:
- DefiLlama: DeFi TVL, stablecoin supply (free API)
- Manual: Known major crypto events

Output:
- data/defi_tvl.csv
- data/stablecoin_supply.csv
- data/known_events.json
"""

import pandas as pd
import numpy as np
import requests
import json
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')

print("=" * 70)
print("DOWNLOADING ADDITIONAL MARKET DATA")
print("=" * 70)

# ============================================================
# PART 1: DeFi TVL from DefiLlama
# ============================================================

print("\n📊 Part 1: Downloading DeFi TVL...")
print("-" * 70)

# Top chains/protocols to track
CHAINS = ['Ethereum', 'BSC', 'Solana', 'Arbitrum', 'Polygon', 'Avalanche', 'Optimism', 'Base']

PROTOCOLS = {
    'aave': 'Aave',
    'uniswap': 'Uniswap',
    'lido': 'Lido',
    'makerdao': 'MakerDAO',
    'curve-dex': 'Curve',
    'compound-v2': 'Compound',
    'pancakeswap': 'PancakeSwap',
    'gmx': 'GMX',
    'instadapp': 'Instadapp',
}

# Download chain TVL
chain_tvl = {}
print("   Downloading chain TVL...")
for chain in tqdm(CHAINS, desc="   Chains"):
    try:
        url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'], unit='s')
            df = df.set_index('date')
            chain_tvl[chain] = df['tvl']
        time.sleep(0.3)
    except Exception as e:
        print(f"   ⚠️ Failed {chain}: {str(e)[:30]}")

# Download protocol TVL
protocol_tvl = {}
print("   Downloading protocol TVL...")
for slug, name in tqdm(PROTOCOLS.items(), desc="   Protocols"):
    try:
        url = f"https://api.llama.fi/protocol/{slug}"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'tvl' in data:
                tvl_data = data['tvl']
                df = pd.DataFrame(tvl_data)
                df['date'] = pd.to_datetime(df['date'], unit='s')
                df = df.set_index('date')
                protocol_tvl[name] = df['totalLiquidityUSD']
        time.sleep(0.3)
    except Exception as e:
        print(f"   ⚠️ Failed {name}: {str(e)[:30]}")

# Combine TVL data
tvl_combined = pd.DataFrame(chain_tvl)
tvl_combined.columns = [f"tvl_{c}" for c in tvl_combined.columns]

protocol_df = pd.DataFrame(protocol_tvl)
protocol_df.columns = [f"tvl_{c}" for c in protocol_df.columns]

tvl_combined = tvl_combined.join(protocol_df, how='outer')

# Add total DeFi TVL
try:
    url = "https://api.llama.fi/v2/historicalChainTvl"
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        total_data = response.json()
        total_df = pd.DataFrame(total_data)
        total_df['date'] = pd.to_datetime(total_df['date'], unit='s')
        total_df = total_df.set_index('date')
        tvl_combined['tvl_total'] = total_df['tvl']
except:
    pass

# Save TVL data
tvl_output = DATA_DIR / 'defi_tvl.csv'
tvl_combined.to_csv(tvl_output)
print(f"   ✓ Saved: {tvl_output.name}")
print(f"   ✓ Shape: {tvl_combined.shape}")
print(f"   ✓ Period: {tvl_combined.index.min()} → {tvl_combined.index.max()}")

# ============================================================
# PART 2: Stablecoin Supply
# ============================================================

print("\n💵 Part 2: Downloading Stablecoin Supply...")
print("-" * 70)

STABLECOINS = {
    'tether': 'USDT',
    'usd-coin': 'USDC',
    'dai': 'DAI',
    'first-digital-usd': 'FDUSD',
    'trueusd': 'TUSD',
}

stablecoin_data = {}

for slug, name in tqdm(STABLECOINS.items(), desc="   Stablecoins"):
    try:
        url = f"https://stablecoins.llama.fi/stablecoin/{slug}"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if 'tokens' in data:
                tokens = data['tokens']
                # Get total circulating
                records = []
                for entry in tokens:
                    if 'circulating' in entry and 'peggedUSD' in entry['circulating']:
                        records.append({
                            'date': pd.to_datetime(entry['date'], unit='s'),
                            'supply': entry['circulating']['peggedUSD']
                        })
                if records:
                    df = pd.DataFrame(records).set_index('date')
                    stablecoin_data[name] = df['supply']
        time.sleep(0.3)
    except Exception as e:
        print(f"   ⚠️ Failed {name}: {str(e)[:30]}")

# Also get total stablecoin market cap
try:
    url = "https://stablecoins.llama.fi/stablecoincharts/all"
    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        data = response.json()
        records = []
        for entry in data:
            if 'totalCirculatingUSD' in entry:
                records.append({
                    'date': pd.to_datetime(entry['date'], unit='s'),
                    'total_stablecoin_supply': entry['totalCirculatingUSD']['peggedUSD']
                })
        if records:
            total_df = pd.DataFrame(records).set_index('date')
            stablecoin_data['Total'] = total_df['total_stablecoin_supply']
except Exception as e:
    print(f"   ⚠️ Failed total supply: {str(e)[:30]}")

# Combine stablecoin data
stablecoin_df = pd.DataFrame(stablecoin_data)
stablecoin_df.columns = [f"stablecoin_{c}" for c in stablecoin_df.columns]

# Save
stablecoin_output = DATA_DIR / 'stablecoin_supply.csv'
stablecoin_df.to_csv(stablecoin_output)
print(f"   ✓ Saved: {stablecoin_output.name}")
print(f"   ✓ Shape: {stablecoin_df.shape}")

# ============================================================
# PART 3: Known Major Events
# ============================================================

print("\n📅 Part 3: Creating Known Events Database...")
print("-" * 70)

KNOWN_EVENTS = [
    # 2021
    {"date": "2021-05-19", "event": "China crypto ban", "category": "REGULATION",
     "entities": ["Bitcoin", "Ethereum"], "impact": 9, "sentiment": -1},
    {"date": "2021-09-07", "event": "El Salvador BTC legal tender", "category": "REGULATION",
     "entities": ["Bitcoin"], "impact": 8, "sentiment": 1},
    {"date": "2021-11-10", "event": "BTC ATH $69k", "category": "MARKET",
     "entities": ["Bitcoin"], "impact": 8, "sentiment": 1},

    # 2022
    {"date": "2022-01-22", "event": "Russia proposes crypto ban", "category": "REGULATION",
     "entities": ["Bitcoin", "Ethereum"], "impact": 7, "sentiment": -1},
    {"date": "2022-05-09", "event": "Terra/LUNA collapse begins", "category": "HACK",
     "entities": ["Terra", "LUNA", "UST"], "impact": 10, "sentiment": -1},
    {"date": "2022-05-12", "event": "UST depeg complete, LUNA crashes 99%", "category": "HACK",
     "entities": ["Terra", "LUNA", "UST"], "impact": 10, "sentiment": -1},
    {"date": "2022-06-13", "event": "Celsius halts withdrawals", "category": "HACK",
     "entities": ["Celsius"], "impact": 9, "sentiment": -1},
    {"date": "2022-06-18", "event": "BTC drops below $20k (first since 2020)", "category": "MARKET",
     "entities": ["Bitcoin"], "impact": 8, "sentiment": -1},
    {"date": "2022-07-06", "event": "Three Arrows Capital files bankruptcy", "category": "HACK",
     "entities": ["Three Arrows Capital"], "impact": 8, "sentiment": -1},
    {"date": "2022-09-15", "event": "Ethereum Merge (PoS transition)", "category": "TECHNICAL",
     "entities": ["Ethereum"], "impact": 9, "sentiment": 1},
    {"date": "2022-11-02", "event": "CoinDesk reveals Alameda balance sheet", "category": "HACK",
     "entities": ["FTX", "Alameda", "FTT"], "impact": 8, "sentiment": -1},
    {"date": "2022-11-06", "event": "CZ announces selling FTT", "category": "HACK",
     "entities": ["FTX", "Binance", "FTT"], "impact": 9, "sentiment": -1},
    {"date": "2022-11-08", "event": "FTX halts withdrawals", "category": "HACK",
     "entities": ["FTX", "FTT"], "impact": 10, "sentiment": -1},
    {"date": "2022-11-11", "event": "FTX files bankruptcy", "category": "HACK",
     "entities": ["FTX", "FTT", "Alameda"], "impact": 10, "sentiment": -1},

    # 2023
    {"date": "2023-02-13", "event": "SEC sues Paxos over BUSD", "category": "REGULATION",
     "entities": ["Paxos", "BUSD", "Binance", "SEC"], "impact": 7, "sentiment": -1},
    {"date": "2023-03-08", "event": "Silvergate Bank liquidation", "category": "HACK",
     "entities": ["Silvergate"], "impact": 7, "sentiment": -1},
    {"date": "2023-03-10", "event": "Silicon Valley Bank collapse", "category": "HACK",
     "entities": ["USDC", "Circle"], "impact": 8, "sentiment": -1},
    {"date": "2023-03-11", "event": "USDC depeg to $0.87", "category": "HACK",
     "entities": ["USDC", "Circle"], "impact": 8, "sentiment": -1},
    {"date": "2023-03-27", "event": "CFTC sues Binance", "category": "REGULATION",
     "entities": ["Binance", "CFTC"], "impact": 8, "sentiment": -1},
    {"date": "2023-06-05", "event": "SEC sues Binance", "category": "REGULATION",
     "entities": ["Binance", "SEC"], "impact": 9, "sentiment": -1},
    {"date": "2023-06-06", "event": "SEC sues Coinbase", "category": "REGULATION",
     "entities": ["Coinbase", "SEC"], "impact": 9, "sentiment": -1},
    {"date": "2023-07-13", "event": "Ripple wins SEC lawsuit (partially)", "category": "REGULATION",
     "entities": ["Ripple", "XRP", "SEC"], "impact": 9, "sentiment": 1},
    {"date": "2023-08-29", "event": "Grayscale wins SEC lawsuit", "category": "REGULATION",
     "entities": ["Grayscale", "SEC", "Bitcoin"], "impact": 8, "sentiment": 1},
    {"date": "2023-11-21", "event": "Binance settles with DOJ, CZ resigns", "category": "REGULATION",
     "entities": ["Binance", "CZ"], "impact": 9, "sentiment": -1},

    # 2024
    {"date": "2024-01-10", "event": "SEC approves Bitcoin spot ETFs", "category": "REGULATION",
     "entities": ["Bitcoin", "SEC", "BlackRock", "Fidelity", "Grayscale"], "impact": 10, "sentiment": 1},
    {"date": "2024-03-05", "event": "BTC hits new ATH $69k (surpasses 2021)", "category": "MARKET",
     "entities": ["Bitcoin"], "impact": 8, "sentiment": 1},
    {"date": "2024-03-14", "event": "BTC ATH $73,737", "category": "MARKET",
     "entities": ["Bitcoin"], "impact": 8, "sentiment": 1},
    {"date": "2024-04-20", "event": "Bitcoin 4th halving", "category": "TECHNICAL",
     "entities": ["Bitcoin"], "impact": 8, "sentiment": 1},
    {"date": "2024-05-23", "event": "SEC approves Ethereum spot ETFs", "category": "REGULATION",
     "entities": ["Ethereum", "SEC"], "impact": 9, "sentiment": 1},
    {"date": "2024-07-23", "event": "Ethereum ETFs begin trading", "category": "MARKET",
     "entities": ["Ethereum"], "impact": 8, "sentiment": 1},
    {"date": "2024-11-05", "event": "Trump wins US election (crypto-friendly)", "category": "REGULATION",
     "entities": ["Bitcoin", "Ethereum"], "impact": 9, "sentiment": 1},
    {"date": "2024-11-22", "event": "BTC crosses $99k", "category": "MARKET",
     "entities": ["Bitcoin"], "impact": 8, "sentiment": 1},
    {"date": "2024-12-05", "event": "BTC crosses $100k for first time", "category": "MARKET",
     "entities": ["Bitcoin"], "impact": 9, "sentiment": 1},

    # 2025 (if applicable)
    {"date": "2025-01-20", "event": "Trump inauguration", "category": "REGULATION",
     "entities": ["Bitcoin", "Ethereum"], "impact": 7, "sentiment": 1},
]

# Save known events
events_output = DATA_DIR / 'known_events.json'
with open(events_output, 'w') as f:
    json.dump(KNOWN_EVENTS, f, indent=2)

print(f"   ✓ Saved: {events_output.name}")
print(f"   ✓ Total events: {len(KNOWN_EVENTS)}")
print(f"   ✓ Categories: {set(e['category'] for e in KNOWN_EVENTS)}")

# Create events DataFrame for easy merging
events_df = pd.DataFrame(KNOWN_EVENTS)
events_df['date'] = pd.to_datetime(events_df['date'])
events_df.to_csv(DATA_DIR / 'known_events.csv', index=False)
print(f"   ✓ Also saved: known_events.csv")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ADDITIONAL DATA DOWNLOAD COMPLETE")
print("=" * 70)

print("\n📊 Files created:")
print(f"   1. defi_tvl.csv - {tvl_combined.shape[0]} rows, {tvl_combined.shape[1]} columns")
print(f"   2. stablecoin_supply.csv - {stablecoin_df.shape[0]} rows, {stablecoin_df.shape[1]} columns")
print(f"   3. known_events.json - {len(KNOWN_EVENTS)} major events")
print(f"   4. known_events.csv - same as JSON but tabular")

print("\n🎯 These provide:")
print("   • DeFi TVL trends for protocol health analysis")
print("   • Stablecoin supply for market flow signals")
print("   • Known events for pattern validation and interpretation")

print("\n📄 Next: python3 scripts/26_build_event_graph.py")
