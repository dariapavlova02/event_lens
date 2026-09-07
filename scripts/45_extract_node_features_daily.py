#!/usr/bin/env python3
"""
Script 45: Extract Daily Node Centrality
========================================
Extracts Degree Centrality for Top 20 Key Entities aggregated DAILY.
Granularity: 24h (00:00 to 00:00 next day).
Output: data/node_features_daily.csv
"""

import os
from neo4j import GraphDatabase
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

# Configuration
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")

DATA_DIR = Path("data")
OUTPUT_FILE = DATA_DIR / "node_features_daily.csv"

# Top 20 Entities (Dynamic or Fixed)
# We use the list identified in scripts/get_top_entities.py earlier
TOP_ENTITIES = [
    'Ethereum', 'Binance', 'Solana', 'Coinbase', 'Arbitrum',
    'FTX', 'Polygon', 'Uniswap', 'Tether', 'Bybit',
    'SEC', 'USDC', 'Hyperliquid', 'Optimism', 'Base',
    'Aave', 'BNB Chain', 'Ripple', 'Avalanche', 'Circle'
]

def get_date_range(driver):
    """Get min and max date from Events."""
    query = "MATCH (e:Event) RETURN min(e.timestamp) as start, max(e.timestamp) as end"
    try:
        with driver.session() as session:
            result = session.run(query).single()
            if result and result["start"] and result["end"]:
                # Neo4j datetime to python datetime
                start = pd.to_datetime(result["start"].to_native()).tz_localize(None)
                end = pd.to_datetime(result["end"].to_native()).tz_localize(None)
                return start, end
    except Exception as e:
        print(f"⚠️ Could not get date range from DB: {e}")

    print("⚠️ Using default date range (2023-2026)")
    return pd.Timestamp("2023-01-01"), pd.Timestamp("2026-01-01")

def extract_daily_features(driver, start_date, end_date):
    """
    Iterate day by day and calculate degree centrality for top entities.
    Degree = count of relationships (MENTIONS) involving the entity on that day.
    """

    # We want full days from start to end
    current_date = start_date.floor('D')
    end_date = end_date.ceil('D')

    results = []

    total_days = (end_date - current_date).days
    print(f"📅 Extracting daily features for {total_days} days...")

    # Pre-compute query template
    # Calculate degree for ALL entities first to get distribution stats,
    # then filter for specific ones.
    query = """
    MATCH (e:Entity)<-[r:MENTIONS]-(ev:Event)
    WHERE ev.timestamp >= datetime($start) AND ev.timestamp < datetime($end)
    WITH e, count(r) as degree
    ORDER BY degree DESC

    WITH collect({name: e.name, degree: degree}) as nodes,
         avg(degree) as avg_deg,
         max(degree) as max_deg,
         stDev(degree) as std_deg,
         sum(degree) as total_deg

    RETURN nodes, avg_deg, max_deg, std_deg, total_deg
    """

    with driver.session() as session:
        pbar = tqdm(total=total_days)
        while current_date < end_date:
            next_date = current_date + timedelta(days=1)

            # Ensure ISO format implies UTC for Neo4j datetime()
            iso_start = current_date.isoformat()
            if not iso_start.endswith("Z") and "+00:00" not in iso_start:
                iso_start += "Z"

            iso_end = next_date.isoformat()
            if not iso_end.endswith("Z") and "+00:00" not in iso_end:
                iso_end += "Z"

            # DEBUG: Print first iteration
            if current_date == start_date.floor('D'):
                print(f"DEBUG: Running query for {iso_start} to {iso_end}")

            result = session.run(query, start=iso_start, end=iso_end).single()

            if result and result['nodes']:
                nodes_data = result['nodes']

                # Global daily stats
                row = {
                    'timestamp': current_date,
                    'avg_degree_daily': result['avg_deg'],
                    'max_degree_daily': result['max_deg'],
                    'std_degree_daily': result['std_deg'],
                    'total_mentions_daily': result['total_deg']
                }

                # Convert nodes list to dict for fast lookup
                node_map = {n['name']: n['degree'] for n in nodes_data}

                # Extract Top 20 stats
                for entity in TOP_ENTITIES:
                    row[f'deg_{entity}'] = node_map.get(entity, 0)

                # Calculate Gini Coefficient for this day
                degrees = [n['degree'] for n in nodes_data]
                if degrees:
                    degrees = np.sort(degrees)
                    n = len(degrees)
                    index = np.arange(1, n + 1)
                    gini = ((2 * np.sum(index * degrees)) / (n * np.sum(degrees))) - (n + 1) / n
                    row['daily_gini'] = gini
                else:
                    row['daily_gini'] = 0.0

                results.append(row)
            else:
                # No events this day
                pass

            current_date = next_date
            pbar.update(1)

        pbar.close()

    return pd.DataFrame(results)

def main():
    print("🔌 Connecting to Neo4j...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("✅ Connected.")

        start, end = get_date_range(driver)
        print(f"   Data range: {start} to {end}")

        df = extract_daily_features(driver, start, end)

        if not df.empty:
            df.set_index('timestamp', inplace=True)
            df.to_csv(OUTPUT_FILE)
            print(f"\n💾 Saved {len(df)} daily records to {OUTPUT_FILE}")
            print(f"   Columns: {len(df.columns)}")
            print("   (Includes stats for: " + ", ".join(TOP_ENTITIES[:5]) + "...)")
        else:
            print("⚠️ No data extracted.")

        driver.close()

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
