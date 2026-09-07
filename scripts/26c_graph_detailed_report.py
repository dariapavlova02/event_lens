#!/usr/bin/env python3
"""
Script 26c: Detailed Graph Report
=================================
Generates a comprehensive analysis of the Neo4j Event Knowledge Graph.
"""

import os
from neo4j import GraphDatabase
import pandas as pd
from tabulate import tabulate

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])

driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

def run_query(query, params=None):
    with driver.session() as session:
        result = session.run(query, params)
        return [dict(record) for record in result]

print("="*70)
print("EVENT KNOWLEDGE GRAPH: DETAILED REPORT")
print("="*70)

# 1. Global Counts
print("\n📊 1. NODE & RELATIONSHIP COUNTS")
print("-" * 30)
nodes = run_query("MATCH (n) RETURN labels(n)[0] as Label, count(*) as Count ORDER BY Count DESC")
print(tabulate(nodes, headers="keys", tablefmt="simple"))

rels = run_query("MATCH ()-[r]->() RETURN type(r) as Type, count(*) as Count ORDER BY Count DESC")
print("\n")
print(tabulate(rels, headers="keys", tablefmt="simple"))

# 2. Top Entities by Mentions & Sentiment
print("\n🏆 2. TOP 20 ENTITIES (By Mentions)")
print("-" * 30)
query = """
MATCH (e:Entity)
RETURN e.name as Name,
       e.type as Type,
       e.mentions as Mentions,
       round(e.avg_sentiment, 2) as Sentiment,
       round(e.avg_impact, 2) as Impact
ORDER BY e.mentions DESC
LIMIT 20
"""
top_entities = run_query(query)
print(tabulate(top_entities, headers="keys", tablefmt="simple"))

# 3. Top Co-occurring Pairs (The "Core" of the conversation)
print("\n🔗 3. TOP 20 CO-OCCURRENCE PAIRS")
print("-" * 30)
query = """
MATCH (e1:Entity)-[r:CO_OCCURS_WITH]-(e2:Entity)
WHERE e1.name < e2.name
RETURN e1.name + ' <-> ' + e2.name as Pair,
       r.count as Co_Occurrences,
       round(r.avg_sentiment_diff, 2) as Avg_Sent_Diff
ORDER BY r.count DESC
LIMIT 20
"""
co_occurs = run_query(query)
print(tabulate(co_occurs, headers="keys", tablefmt="simple"))

# 4. Category Breakdown
print("\n📂 4. MENTIONS BY CATEGORY")
print("-" * 30)
query = """
MATCH (e:Event)-[m:MENTIONS]->(ent:Entity)
RETURN m.category as Category, count(m) as Count
ORDER BY Count DESC
"""
cats = run_query(query)
print(tabulate(cats, headers="keys", tablefmt="simple"))

# 5. Volatility Spike Analysis
print("\n📉 5. VOLATILITY SPIKES OVERVIEW")
print("-" * 30)
query = """
MATCH (v:VolatilitySpike)
RETURN substring(toString(v.date), 0, 4) as Year, count(v) as Spikes, avg(v.magnitude) as Avg_Magnitude
ORDER BY Year
"""
spikes = run_query(query)
print(tabulate(spikes, headers="keys", tablefmt="simple"))

# 6. Most "Central" Entities (PageRank - Estimated by Degree since we didn't run GDS yet)
print("\n🕸️ 6. MOST CONNECTED ENTITIES (Highest Degree)")
print("-" * 30)
query = """
MATCH (e:Entity)
RETURN e.name as Entity, size((e)--()) as Degree
ORDER BY Degree DESC
LIMIT 15
"""
central = run_query(query)
print(tabulate(central, headers="keys", tablefmt="simple"))

driver.close()
print("\n" + "="*70)
