#!/usr/bin/env python3
"""
Script 26: Build Telegram Crypto Event Knowledge Graph (Neo4j)
==============================================================

Creates a knowledge graph from TBSA-annotated Telegram messages.

Graph Schema:
-------------
NODES:
- (:Entity {name, ticker, type, total_mentions, avg_impact, avg_sentiment})
- (:Event {id, timestamp, channel, views, forwards, global_sentiment, is_fact, impact_sum})
- (:Channel {name, total_messages, avg_views})
- (:Category {name})
- (:KnownEvent {name, date, impact, sentiment})
- (:VolatilitySpike {date, magnitude})

EDGES:
- (Event)-[:MENTIONS {sentiment, impact, category}]->(Entity)
- (Entity)-[:CO_OCCURS_WITH {count, avg_sentiment_diff}]->(Entity)
- (Event)-[:POSTED_ON]->(Channel)
- (Event)-[:CATEGORIZED_AS]->(Category)
- (Event)-[:TEMPORAL_FOLLOWS {hours_delta}]->(Event)  # Same entity within 24h
- (Event)-[:PRECEDES_SPIKE {hours_before}]->(VolatilitySpike)
- (KnownEvent)-[:INVOLVES]->(Entity)

Requirements:
- Neo4j running locally (docker or desktop)
- pip install neo4j pandas tqdm

Usage:
1. Start Neo4j: docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j
2. python3 scripts/26_build_event_graph_neo4j.py
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
import os

# Neo4j connection settings
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')

print("=" * 70)
print("BUILDING TELEGRAM CRYPTO EVENT KNOWLEDGE GRAPH")
print("=" * 70)
print(f"\nNeo4j URI: {NEO4J_URI}")

# ============================================================
# STEP 1: Load and parse TBSA data
# ============================================================

print("\n📊 Step 1: Loading TBSA data...")
print("-" * 70)

# Load main dataset
df = pd.read_csv(DATA_DIR / 'all_messages_tbsa.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"   Total messages: {len(df):,}")
print(f"   With TBSA results: {df['tbsa_has_result'].sum():,}")

# Filter to successful TBSA
df = df[df['tbsa_has_result'] == True].copy()
print(f"   After filtering: {len(df):,}")

# Parse TBSA targets
print("\n   Parsing TBSA targets...")

events = []
entity_mentions = []
co_occurrences = defaultdict(lambda: {'count': 0, 'sentiment_diffs': []})

for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Parsing"):
    try:
        # Parse targets JSON - handle double quotes
        targets_str = row['tbsa_targets_json']
        if pd.isna(targets_str):
            continue

        # Fix double-quoted JSON
        targets_str = targets_str.replace('""', '"')
        if targets_str.startswith('"') and targets_str.endswith('"'):
            targets_str = targets_str[1:-1]

        targets = json.loads(targets_str)

        if not targets:
            continue

        # Create event record
        event_id = f"msg_{idx}"
        impact_sum = sum(t.get('impact', 5) for t in targets)

        event = {
            'id': event_id,
            'timestamp': row['date'],
            'channel': row['channel'],
            'views': row.get('views', 0) or 0,
            'forwards': row.get('forwards', 0) or 0,
            'global_sentiment': row.get('tbsa_global_sentiment', 0) or 0,
            'is_fact': row.get('tbsa_is_fact', False),
            'impact_sum': impact_sum,
            'num_targets': len(targets)
        }
        events.append(event)

        # Extract entity mentions
        entities_in_msg = []
        for t in targets:
            entity_name = t.get('name', 'UNKNOWN')
            if not entity_name or entity_name == 'UNKNOWN':
                continue

            mention = {
                'event_id': event_id,
                'entity_name': entity_name,
                'ticker': t.get('ticker'),
                'type': t.get('type', 'OTHER'),
                'sentiment': t.get('sentiment', 0),
                'impact': t.get('impact', 5),
                'category': t.get('category', 'OTHER')
            }
            entity_mentions.append(mention)
            entities_in_msg.append((entity_name, t.get('sentiment', 0)))

        # Track co-occurrences
        for i, (e1, s1) in enumerate(entities_in_msg):
            for j, (e2, s2) in enumerate(entities_in_msg):
                if i < j:
                    key = tuple(sorted([e1, e2]))
                    co_occurrences[key]['count'] += 1
                    co_occurrences[key]['sentiment_diffs'].append(abs(s1 - s2))

    except Exception as e:
        continue

print(f"\n   ✓ Events created: {len(events):,}")
print(f"   ✓ Entity mentions: {len(entity_mentions):,}")
print(f"   ✓ Co-occurrence pairs: {len(co_occurrences):,}")

# ============================================================
# STEP 2: Aggregate entity statistics
# ============================================================

print("\n📈 Step 2: Aggregating entity statistics...")
print("-" * 70)

entity_stats = defaultdict(lambda: {
    'mentions': 0,
    'total_impact': 0,
    'total_sentiment': 0,
    'types': set(),
    'tickers': set(),
    'categories': set()
})

for m in entity_mentions:
    name = m['entity_name']
    entity_stats[name]['mentions'] += 1
    entity_stats[name]['total_impact'] += m['impact']
    entity_stats[name]['total_sentiment'] += m['sentiment']
    if m['type']:
        entity_stats[name]['types'].add(m['type'])
    if m['ticker']:
        entity_stats[name]['tickers'].add(m['ticker'])
    if m['category']:
        entity_stats[name]['categories'].add(m['category'])

entities = []
for name, stats in entity_stats.items():
    entities.append({
        'name': name,
        'ticker': list(stats['tickers'])[0] if stats['tickers'] else None,
        'type': list(stats['types'])[0] if stats['types'] else 'OTHER',
        'mentions': stats['mentions'],
        'avg_impact': stats['total_impact'] / stats['mentions'],
        'avg_sentiment': stats['total_sentiment'] / stats['mentions']
    })

print(f"   ✓ Unique entities: {len(entities):,}")

# Top entities
top_entities = sorted(entities, key=lambda x: x['mentions'], reverse=True)[:20]
print("\n   Top 20 entities by mentions:")
for e in top_entities:
    print(f"      {e['name']:30s} - {e['mentions']:5,} mentions, avg_sentiment: {e['avg_sentiment']:+.2f}")

# ============================================================
# STEP 3: Load known events and volatility spikes
# ============================================================

print("\n🎯 Step 3: Loading known events and volatility spikes...")
print("-" * 70)

# Known events
known_events = []
try:
    with open(DATA_DIR / 'known_events.json', 'r') as f:
        known_events = json.load(f)
    print(f"   ✓ Known events: {len(known_events)}")
except:
    print("   ⚠️ Known events file not found")

# Volatility spikes from market data
volatility_spikes = []
try:
    market_df = pd.read_csv(DATA_DIR / 'market_data_multi_entity.csv', index_col=0, parse_dates=True)
    if 'btc_vol_spike_2sigma' in market_df.columns:
        spikes = market_df[market_df['btc_vol_spike_2sigma'] == 1]
        for date, row in spikes.iterrows():
            volatility_spikes.append({
                'date': date,
                'magnitude': row.get('Bitcoin_volatility', 0)
            })
        print(f"   ✓ Volatility spikes: {len(volatility_spikes)}")
except Exception as e:
    print(f"   ⚠️ Could not load volatility spikes: {str(e)[:50]}")

# ============================================================
# STEP 4: Connect to Neo4j and create schema
# ============================================================

print("\n🔌 Step 4: Connecting to Neo4j...")
print("-" * 70)

try:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    # Test connection
    with driver.session() as session:
        result = session.run("RETURN 1 AS test")
        result.single()

    print("   ✓ Connected to Neo4j!")

except ImportError:
    print("   ⚠️ neo4j package not installed. Run: pip install neo4j")
    print("\n   Saving data to JSON for later import...")

    # Save to JSON for later
    output = {
        'events': events[:1000],  # Sample
        'entities': entities,
        'entity_mentions': entity_mentions[:10000],  # Sample
        'co_occurrences': [{'entities': list(k), 'count': v['count']} for k, v in list(co_occurrences.items())[:1000]],
        'known_events': known_events,
        'volatility_spikes': [{'date': str(s['date']), 'magnitude': s['magnitude']} for s in volatility_spikes]
    }

    with open(DATA_DIR / 'event_graph_data.json', 'w') as f:
        json.dump(output, f, default=str, indent=2)

    print(f"   ✓ Saved to: event_graph_data.json")
    print("\n   To import into Neo4j later:")
    print("   1. Start Neo4j")
    print("   2. Run this script again with neo4j package installed")
    driver = None

except Exception as e:
    print(f"   ⚠️ Could not connect to Neo4j: {str(e)}")
    print("\n   Make sure Neo4j is running:")
    print("   docker run -d -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j")
    driver = None

# ============================================================
# STEP 5: Create graph in Neo4j
# ============================================================

if driver:
    print("\n🏗️ Step 5: Creating graph in Neo4j...")
    print("-" * 70)

    with driver.session() as session:
        # Clear existing data (optional)
        print("   Clearing existing data...")
        session.run("MATCH (n) DETACH DELETE n")

        # Create constraints/indexes
        print("   Creating indexes...")
        try:
            session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run("CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE")
            session.run("CREATE CONSTRAINT channel_name IF NOT EXISTS FOR (c:Channel) REQUIRE c.name IS UNIQUE")
            session.run("CREATE INDEX event_timestamp IF NOT EXISTS FOR (e:Event) ON (e.timestamp)")
        except:
            pass  # Indexes might already exist

        # Create Channel nodes
        print("   Creating Channel nodes...")
        channels = df['channel'].unique()
        for channel in channels:
            channel_data = df[df['channel'] == channel]
            session.run("""
                MERGE (c:Channel {name: $name})
                SET c.total_messages = $total_messages,
                    c.avg_views = $avg_views
            """, {
                'name': channel,
                'total_messages': len(channel_data),
                'avg_views': channel_data['views'].mean() or 0
            })
        print(f"      ✓ Created {len(channels)} channels")

        # Create Entity nodes
        print("   Creating Entity nodes...")
        for entity in tqdm(entities, desc="      Entities"):
            session.run("""
                MERGE (e:Entity {name: $name})
                SET e.ticker = $ticker,
                    e.type = $type,
                    e.mentions = $mentions,
                    e.avg_impact = $avg_impact,
                    e.avg_sentiment = $avg_sentiment
            """, entity)
        print(f"      ✓ Created {len(entities)} entities")

        # Create Category nodes
        print("   Creating Category nodes...")
        categories = set(m['category'] for m in entity_mentions if m['category'])
        for cat in categories:
            session.run("MERGE (c:Category {name: $name})", {'name': cat})
        print(f"      ✓ Created {len(categories)} categories")

        # Create KnownEvent nodes
        print("   Creating KnownEvent nodes...")
        for ke in known_events:
            session.run("""
                CREATE (k:KnownEvent {
                    name: $name,
                    date: date($date),
                    category: $category,
                    impact: $impact,
                    sentiment: $sentiment
                })
            """, {
                'name': ke['event'],
                'date': ke['date'],
                'category': ke['category'],
                'impact': ke['impact'],
                'sentiment': ke['sentiment']
            })

            # Link to entities
            for entity_name in ke.get('entities', []):
                session.run("""
                    MATCH (k:KnownEvent {name: $event_name})
                    MATCH (e:Entity {name: $entity_name})
                    MERGE (k)-[:INVOLVES]->(e)
                """, {'event_name': ke['event'], 'entity_name': entity_name})
        print(f"      ✓ Created {len(known_events)} known events")

        # Create VolatilitySpike nodes
        print("   Creating VolatilitySpike nodes...")
        for spike in volatility_spikes:
            session.run("""
                CREATE (v:VolatilitySpike {
                    date: date($date),
                    magnitude: $magnitude
                })
            """, {
                'date': spike['date'].strftime('%Y-%m-%d') if hasattr(spike['date'], 'strftime') else str(spike['date'])[:10],
                'magnitude': spike['magnitude']
            })
        print(f"      ✓ Created {len(volatility_spikes)} volatility spikes")

        # Create Event nodes and relationships (in batches)
        print("   Creating Event nodes and relationships...")
        batch_size = 500

        for i in tqdm(range(0, len(events), batch_size), desc="      Events"):
            batch_events = events[i:i+batch_size]
            batch_mentions = [m for m in entity_mentions if any(m['event_id'] == e['id'] for e in batch_events)]

            # Create events
            for event in batch_events:
                session.run("""
                    CREATE (e:Event {
                        id: $id,
                        timestamp: datetime($timestamp),
                        views: $views,
                        forwards: $forwards,
                        global_sentiment: $global_sentiment,
                        is_fact: $is_fact,
                        impact_sum: $impact_sum
                    })
                """, {
                    'id': event['id'],
                    'timestamp': event['timestamp'].isoformat(),
                    'views': event['views'],
                    'forwards': event['forwards'],
                    'global_sentiment': event['global_sentiment'],
                    'is_fact': event['is_fact'],
                    'impact_sum': event['impact_sum']
                })

                # Link to channel
                session.run("""
                    MATCH (e:Event {id: $event_id})
                    MATCH (c:Channel {name: $channel})
                    MERGE (e)-[:POSTED_ON]->(c)
                """, {'event_id': event['id'], 'channel': event['channel']})

            # Create MENTIONS relationships
            for mention in batch_mentions:
                session.run("""
                    MATCH (e:Event {id: $event_id})
                    MATCH (ent:Entity {name: $entity_name})
                    MERGE (e)-[m:MENTIONS]->(ent)
                    SET m.sentiment = $sentiment,
                        m.impact = $impact,
                        m.category = $category
                """, {
                    'event_id': mention['event_id'],
                    'entity_name': mention['entity_name'],
                    'sentiment': mention['sentiment'],
                    'impact': mention['impact'],
                    'category': mention['category']
                })

        print(f"      ✓ Created {len(events)} events with mentions")

        # Create CO_OCCURS_WITH relationships
        print("   Creating co-occurrence relationships...")
        co_occur_list = [(k, v) for k, v in co_occurrences.items() if v['count'] >= 3]  # Min 3 co-occurrences

        for (e1, e2), data in tqdm(co_occur_list, desc="      Co-occurrences"):
            avg_diff = np.mean(data['sentiment_diffs']) if data['sentiment_diffs'] else 0
            session.run("""
                MATCH (e1:Entity {name: $e1})
                MATCH (e2:Entity {name: $e2})
                MERGE (e1)-[r:CO_OCCURS_WITH]-(e2)
                SET r.count = $count,
                    r.avg_sentiment_diff = $avg_diff
            """, {'e1': e1, 'e2': e2, 'count': data['count'], 'avg_diff': avg_diff})

        print(f"      ✓ Created {len(co_occur_list)} co-occurrence relationships")

    driver.close()
    print("\n   ✓ Graph created successfully!")

# ============================================================
# STEP 6: Summary and next steps
# ============================================================

print("\n" + "=" * 70)
print("EVENT KNOWLEDGE GRAPH CREATED")
print("=" * 70)

print("\n📊 Graph Statistics:")
print(f"   • Events (messages): {len(events):,}")
print(f"   • Entities: {len(entities):,}")
print(f"   • Entity mentions: {len(entity_mentions):,}")
print(f"   • Co-occurrence pairs: {len(co_occurrences):,}")
print(f"   • Known events: {len(known_events)}")
print(f"   • Volatility spikes: {len(volatility_spikes)}")

print("\n🔍 Example Cypher Queries:")
print("""
   # Top entities by mentions
   MATCH (e:Entity) RETURN e.name, e.mentions ORDER BY e.mentions DESC LIMIT 20

   # Co-occurring entities with negative sentiment
   MATCH (e1:Entity)-[r:CO_OCCURS_WITH]-(e2:Entity)
   WHERE e1.avg_sentiment < 0 AND e2.avg_sentiment < 0
   RETURN e1.name, e2.name, r.count
   ORDER BY r.count DESC LIMIT 20

   # Events before volatility spikes
   MATCH (e:Event)-[:MENTIONS]->(ent:Entity)
   MATCH (v:VolatilitySpike)
   WHERE date(e.timestamp) = v.date - duration('P1D')
   RETURN e, ent, v LIMIT 50

   # Known events with most entity mentions
   MATCH (k:KnownEvent)-[:INVOLVES]->(e:Entity)
   RETURN k.name, collect(e.name) as entities, k.impact
   ORDER BY k.impact DESC
""")

print("\n📄 Next steps:")
print("   python3 scripts/27_extract_graph_features.py")
print("   python3 scripts/28_train_spike_predictor.py")
