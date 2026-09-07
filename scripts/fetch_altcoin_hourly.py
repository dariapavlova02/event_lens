import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')

def fetch_hourly_data(ticker_map):
    """
    Fetches max available hourly data (730 days) for given tickers.
    ticker_map: dict {'Symbol': 'Name'} e.g. {'ETH-USD': 'Ethereum'}
    """
    merged_df = pd.DataFrame()

    for symbol, name in ticker_map.items():
        print(f"📥 Fetching {symbol} ({name})...")
        try:
            # Fetch max 730d hourly data
            df = yf.download(symbol, period="730d", interval="1h", progress=False)

            if df.empty:
                print(f"⚠️ No data found for {symbol}")
                continue

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Rename columns
            df = df[['Close', 'Volume']].rename(columns={
                'Close': f'{name}_price',
                'Volume': f'{name}_volume'
            })

            # Calculate Returns and Volatility
            # Returns = Log Return
            df[f'{name}_returns'] = np.log(df[f'{name}_price'] / df[f'{name}_price'].shift(1))

            # Volatility = Rolling Std Dev of returns (24h window)
            df[f'{name}_volatility'] = df[f'{name}_returns'].rolling(window=24).std()

            if merged_df.empty:
                merged_df = df
            else:
                merged_df = merged_df.join(df, how='outer')

        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")

    # Clean up
    merged_df = merged_df.sort_index()
    # Drop rows where we don't have enough data for volatility calc
    merged_df = merged_df.dropna()

    # Save
    output_path = DATA_DIR / 'altcoin_market_hourly.csv'
    merged_df.to_csv(output_path)
    print(f"\n✅ Saved {len(merged_df)} rows to {output_path}")
    print(f"   Range: {merged_df.index.min()} to {merged_df.index.max()}")
    return merged_df

if __name__ == "__main__":
    tickers = {
        'ETH-USD': 'Ethereum',
        'SOL-USD': 'Solana'
    }
    fetch_hourly_data(tickers)
