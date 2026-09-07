#!/usr/bin/env python3
"""
Script 25: Download Multi-Entity Market Data
=============================================
Downloads price, volume, and market data for all major entities
mentioned in the Telegram sentiment dataset.

Data sources:
- yfinance: Crypto OHLCV (hourly)
- CoinGecko: Market caps, global metrics
- alternative.me: Fear & Greed Index
- DefiLlama: DeFi TVL

Output: data/market_data_multi_entity.csv
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
OUTPUT_FILE = DATA_DIR / 'market_data_multi_entity.csv'

# Date range (matching Telegram data: 2021-08-23 → 2025-12-14)
START_DATE = '2021-08-01'
END_DATE = '2025-12-15'

# Entities to track (mapped to yfinance tickers)
CRYPTO_TICKERS = {
    # Major chains
    'Bitcoin': 'BTC-USD',
    'Ethereum': 'ETH-USD',
    'Solana': 'SOL-USD',
    'BNB Chain': 'BNB-USD',
    'Ripple': 'XRP-USD',
    'Cardano': 'ADA-USD',
    'Avalanche': 'AVAX-USD',
    'Polygon': 'MATIC-USD',
    'Chainlink': 'LINK-USD',
    'Polkadot': 'DOT-USD',
    'Litecoin': 'LTC-USD',
    'Cosmos': 'ATOM-USD',

    # DeFi protocols
    'Uniswap': 'UNI-USD',
    'Aave': 'AAVE-USD',
    'Compound': 'COMP-USD',
    'Maker': 'MKR-USD',
    'Curve': 'CRV-USD',
    'Synthetix': 'SNX-USD',

    # Layer 2
    'Arbitrum': 'ARB-USD',
    'Optimism': 'OP-USD',

    # CEX tokens
    'Binance': 'BNB-USD',  # Same as BNB Chain
    'FTX': 'FTT-USD',
    'Crypto.com': 'CRO-USD',
    'OKX': 'OKB-USD',

    # Stablecoins (for market cap tracking)
    'Tether': 'USDT-USD',
    'USDC': 'USDC-USD',
    'DAI': 'DAI-USD',

    # Meme/trending
    'Dogecoin': 'DOGE-USD',
    'Shiba Inu': 'SHIB-USD',
}

# CoinGecko IDs for market cap data
COINGECKO_IDS = {
    'bitcoin': 'Bitcoin',
    'ethereum': 'Ethereum',
    'solana': 'Solana',
    'binancecoin': 'BNB Chain',
    'ripple': 'Ripple',
    'cardano': 'Cardano',
    'avalanche-2': 'Avalanche',
    'matic-network': 'Polygon',
    'chainlink': 'Chainlink',
    'tether': 'Tether',
    'usd-coin': 'USDC',
}

print("=" * 70)
print("DOWNLOADING MULTI-ENTITY MARKET DATA")
print("=" * 70)
print(f"\nDate range: {START_DATE} → {END_DATE}")
print(f"Entities: {len(CRYPTO_TICKERS)}")
print()

# ============================================================
# STEP 1: Download OHLCV from yfinance
# ============================================================

print("📊 Step 1: Downloading OHLCV data from yfinance...")
print("-" * 70)

# Get unique tickers
unique_tickers = list(set(CRYPTO_TICKERS.values()))
print(f"   Unique tickers: {len(unique_tickers)}")

# Download in chunks to avoid rate limits
all_data = {}
failed_tickers = []

for ticker in tqdm(unique_tickers, desc="   Downloading"):
    try:
        # yfinance hourly data is limited to last 730 days
        # For longer periods, we need to use daily and interpolate
        # or download in chunks

        # Try hourly first for recent data
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            interval='1d',  # Daily for full history
            progress=False
        )

        if len(df) > 0:
            df = df.reset_index()
            df['ticker'] = ticker
            all_data[ticker] = df
        else:
            failed_tickers.append(ticker)

    except Exception as e:
        failed_tickers.append(ticker)
        print(f"   ⚠️ Failed: {ticker} - {str(e)[:50]}")

    time.sleep(0.2)  # Rate limiting

print(f"\n   ✓ Downloaded: {len(all_data)} tickers")
if failed_tickers:
    print(f"   ✗ Failed: {failed_tickers}")

# ============================================================
# STEP 2: Process and combine OHLCV data
# ============================================================

print("\n📈 Step 2: Processing OHLCV data...")
print("-" * 70)

# Combine all price data
price_series = {}
volume_series = {}

for ticker, df in all_data.items():
    if df is None or len(df) == 0:
        continue

    # Get entity name from ticker
    entity_names = [k for k, v in CRYPTO_TICKERS.items() if v == ticker]
    if not entity_names:
        continue
    entity_name = entity_names[0]

    # Handle different yfinance output formats
    if isinstance(df.columns, pd.MultiIndex):
        # Multi-ticker download format
        if 'Close' in df.columns.get_level_values(0):
            price_series[entity_name] = df['Close'][ticker] if ticker in df['Close'].columns else df['Close'].iloc[:, 0]
        if 'Volume' in df.columns.get_level_values(0):
            volume_series[entity_name] = df['Volume'][ticker] if ticker in df['Volume'].columns else df['Volume'].iloc[:, 0]
    else:
        # Single ticker download format - reset index properly
        if 'Date' in df.columns:
            df = df.set_index('Date')
        elif 'Datetime' in df.columns:
            df = df.set_index('Datetime')
        elif not isinstance(df.index, pd.DatetimeIndex):
            # If index is not datetime, skip
            continue

        if 'Close' in df.columns:
            price_series[entity_name] = df['Close']
        if 'Volume' in df.columns:
            volume_series[entity_name] = df['Volume']

# Create DataFrames from series
if price_series:
    prices_df = pd.DataFrame(price_series)
else:
    prices_df = pd.DataFrame()

if volume_series:
    volumes_df = pd.DataFrame(volume_series)
else:
    volumes_df = pd.DataFrame()

print(f"   Price data shape: {prices_df.shape}")
print(f"   Volume data shape: {volumes_df.shape}")
print(f"   Date range: {prices_df.index.min()} → {prices_df.index.max()}")

# ============================================================
# STEP 3: Calculate derived metrics
# ============================================================

print("\n📉 Step 3: Calculating derived metrics...")
print("-" * 70)

# Returns (daily)
returns_df = prices_df.pct_change()
returns_df.columns = [f"{col}_returns" for col in returns_df.columns]

# Volatility (20-day rolling std of returns)
volatility_df = prices_df.pct_change().rolling(20).std()
volatility_df.columns = [f"{col}_volatility" for col in volatility_df.columns]

# Log returns
log_returns_df = np.log(prices_df / prices_df.shift(1))
log_returns_df.columns = [f"{col}_log_returns" for col in log_returns_df.columns]

print(f"   ✓ Returns calculated")
print(f"   ✓ Volatility (20d rolling) calculated")
print(f"   ✓ Log returns calculated")

# ============================================================
# STEP 4: Market-wide metrics
# ============================================================

print("\n🌍 Step 4: Calculating market-wide metrics...")
print("-" * 70)

# BTC dominance proxy (BTC price / sum of top prices, normalized)
if 'Bitcoin' in prices_df.columns:
    top_cryptos = ['Bitcoin', 'Ethereum', 'Solana', 'BNB Chain', 'Ripple', 'Cardano']
    available_top = [c for c in top_cryptos if c in prices_df.columns]

    if len(available_top) > 1:
        # Normalize each to percent change from start
        normalized = prices_df[available_top].div(prices_df[available_top].iloc[0]) * 100
        btc_dominance_proxy = normalized['Bitcoin'] / normalized.sum(axis=1)
    else:
        btc_dominance_proxy = pd.Series(index=prices_df.index, data=1.0)
else:
    btc_dominance_proxy = pd.Series(index=prices_df.index, data=np.nan)

# Cross-correlation (average pairwise correlation of returns)
def rolling_correlation(returns_subset, window=20):
    """Calculate average pairwise correlation over rolling window"""
    corr_series = []
    for i in range(window, len(returns_subset)):
        window_data = returns_subset.iloc[i-window:i]
        corr_matrix = window_data.corr()
        # Average of off-diagonal elements
        mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        avg_corr = corr_matrix.where(mask).stack().mean()
        corr_series.append(avg_corr)

    return pd.Series(corr_series, index=returns_subset.index[window:])

# Calculate for top 5 cryptos (to avoid noise)
top_5 = ['Bitcoin', 'Ethereum', 'Solana', 'BNB Chain', 'Ripple']
available_top_5 = [c for c in top_5 if c in prices_df.columns]

if len(available_top_5) >= 3:
    returns_subset = prices_df[available_top_5].pct_change().dropna()
    avg_correlation = rolling_correlation(returns_subset, window=20)
    print(f"   ✓ Average cross-correlation calculated")
else:
    avg_correlation = pd.Series(index=prices_df.index, data=np.nan)

# Market volatility index (equal-weighted volatility of top assets)
if len(available_top_5) >= 2:
    market_vol = volatility_df[[f"{c}_volatility" for c in available_top_5 if f"{c}_volatility" in volatility_df.columns]].mean(axis=1)
else:
    market_vol = volatility_df.mean(axis=1)

print(f"   ✓ BTC dominance proxy calculated")
print(f"   ✓ Market volatility index calculated")

# ============================================================
# STEP 5: Try to get Fear & Greed Index
# ============================================================

print("\n😨 Step 5: Downloading Fear & Greed Index...")
print("-" * 70)

try:
    # alternative.me Fear & Greed API (free, no API key)
    url = "https://api.alternative.me/fng/?limit=0&format=json"
    response = requests.get(url, timeout=30)

    if response.status_code == 200:
        fng_data = response.json()['data']
        fng_df = pd.DataFrame(fng_data)
        fng_df['date'] = pd.to_datetime(fng_df['timestamp'].astype(int), unit='s')
        fng_df['fear_greed'] = fng_df['value'].astype(int)
        fng_df = fng_df.set_index('date').sort_index()
        fear_greed = fng_df['fear_greed']
        print(f"   ✓ Fear & Greed Index: {len(fear_greed)} days")
    else:
        fear_greed = pd.Series(dtype=float)
        print(f"   ⚠️ Failed to download Fear & Greed Index")
except Exception as e:
    fear_greed = pd.Series(dtype=float)
    print(f"   ⚠️ Error downloading Fear & Greed: {str(e)[:50]}")

# ============================================================
# STEP 6: Combine all data
# ============================================================

print("\n🔗 Step 6: Combining all data...")
print("-" * 70)

# Rename price columns
prices_df.columns = [f"{col}_price" for col in prices_df.columns]
volumes_df.columns = [f"{col}_volume" for col in volumes_df.columns]

# Combine everything
combined = prices_df.join(volumes_df, how='outer')
combined = combined.join(returns_df, how='outer')
combined = combined.join(volatility_df, how='outer')
combined = combined.join(log_returns_df, how='outer')

# Add market-wide metrics
combined['btc_dominance_proxy'] = btc_dominance_proxy
combined['market_volatility'] = market_vol
combined['avg_cross_correlation'] = avg_correlation

# Add Fear & Greed if available
if len(fear_greed) > 0:
    combined = combined.join(fear_greed.rename('fear_greed'), how='left')
    # Forward fill Fear & Greed (it's daily, our data might be daily too)
    if 'fear_greed' in combined.columns:
        combined['fear_greed'] = combined['fear_greed'].ffill()

print(f"   Combined shape: {combined.shape}")
print(f"   Columns: {len(combined.columns)}")

# ============================================================
# STEP 7: Add volatility events (spikes)
# ============================================================

print("\n🎯 Step 7: Identifying volatility events...")
print("-" * 70)

# BTC volatility spikes
if 'Bitcoin_volatility' in combined.columns:
    btc_vol = combined['Bitcoin_volatility']
    vol_mean = btc_vol.mean()
    vol_std = btc_vol.std()

    combined['btc_vol_spike_2sigma'] = (btc_vol > vol_mean + 2 * vol_std).astype(int)
    combined['btc_vol_spike_1_5sigma'] = (btc_vol > vol_mean + 1.5 * vol_std).astype(int)

    spike_count_2 = combined['btc_vol_spike_2sigma'].sum()
    spike_count_1_5 = combined['btc_vol_spike_1_5sigma'].sum()
    print(f"   ✓ BTC volatility spikes (2σ): {spike_count_2}")
    print(f"   ✓ BTC volatility spikes (1.5σ): {spike_count_1_5}")

# Market volatility spikes
if 'market_volatility' in combined.columns:
    mkt_vol = combined['market_volatility']
    mkt_vol_mean = mkt_vol.mean()
    mkt_vol_std = mkt_vol.std()

    combined['market_vol_spike_2sigma'] = (mkt_vol > mkt_vol_mean + 2 * mkt_vol_std).astype(int)
    print(f"   ✓ Market volatility spikes (2σ): {combined['market_vol_spike_2sigma'].sum()}")

# ============================================================
# STEP 8: Save to CSV
# ============================================================

print("\n💾 Step 8: Saving data...")
print("-" * 70)

combined.to_csv(OUTPUT_FILE)

print(f"   ✓ Saved to: {OUTPUT_FILE}")
print(f"   ✓ Size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")
print(f"   ✓ Shape: {combined.shape}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print(f"\n📊 Data Summary:")
print(f"   Period: {combined.index.min()} → {combined.index.max()}")
print(f"   Total rows: {len(combined):,}")
print(f"   Total columns: {len(combined.columns)}")

print(f"\n📈 Entities with price data:")
price_cols = [c.replace('_price', '') for c in combined.columns if c.endswith('_price')]
for entity in sorted(price_cols):
    print(f"   • {entity}")

print(f"\n🎯 Available targets:")
print(f"   • Bitcoin returns & volatility")
print(f"   • Market-wide volatility index")
print(f"   • Volatility spike events (binary)")
print(f"   • BTC dominance proxy")
print(f"   • Average cross-correlation")
if 'fear_greed' in combined.columns:
    print(f"   • Fear & Greed Index")

print(f"\n📄 Next steps:")
print(f"   python3 scripts/26_build_event_graph.py")
