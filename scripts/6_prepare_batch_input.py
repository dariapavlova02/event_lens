"""
📦 Step 1: Prepare 4 JSONL batch input files for OpenAI Batch API

Converts btc_messages.csv → batch_input_1.jsonl ... batch_input_4.jsonl
Splits 17,770 messages into 4 batches (~4,442 messages each)
Uses the same TBSA prompt that achieved 86% success rate in testing
"""
import pandas as pd
import json
from pathlib import Path

NUM_BATCHES = 10

TBSA_SYSTEM_PROMPT = """You are a crypto knowledge graph analyst for Lynoxis Graph database.

Your task: Extract TARGET-BASED sentiment from Telegram messages (English/Russian/mixed).

⚠️ CRITICAL: Messages can mention multiple entities with DIFFERENT sentiments!
Example: "Sold all my Ethereum to buy Pepe" → ETH is NEGATIVE, Pepe is POSITIVE

Return ONLY a valid JSON object with these fields:

{
  "targets": [
    {
      "name": "Standardized protocol/token name (e.g., 'Uniswap', 'Bitcoin', 'Curve Finance')",
      "ticker": "Token symbol or null (e.g., 'UNI', 'BTC', null for protocols without tokens)",
      "type": "PROTOCOL | TOKEN | CHAIN | CEX | OTHER",
      "sentiment": -1 | 0 | 1,
      "impact": 1-10,
      "category": "MARKET | HACK | OPINION | TECHNICAL | REGULATION | COMMUNITY"
    }
  ],
  "global_sentiment": -1 | 0 | 1,
  "is_fact": true | false
}

TYPE CLASSIFICATION RULES:
- PROTOCOL: DeFi apps, dApps (Uniswap, Curve, Aave, SushiSwap)
- TOKEN: Tradable assets (UNI, CRV, AAVE, BTC, ETH, SOL)
- CHAIN: Layer 1/Layer 2 blockchains (Ethereum, Arbitrum, Optimism, Base)
- CEX: Centralized exchanges (Binance, Coinbase, FTX)
- OTHER: DAOs, people, events

SENTIMENT RULES (per target):
- -1: Bearish/Negative for THIS entity (selling, hack, downtrend)
- 0: Neutral mention or unclear
- +1: Bullish/Positive for THIS entity (buying, growth, good news)

IMPACT SCORE (1-10 per target):
- 1-2: Casual mention
- 3-4: Minor news
- 5-6: Notable update
- 7-8: Major event
- 9-10: Critical (hack, regulation)

CATEGORY (primary theme):
- MARKET: Price, volume, trading
- HACK: Exploits, vulnerabilities
- OPINION: Speculation, predictions
- TECHNICAL: TA, on-chain metrics
- REGULATION: Legal, SEC, government
- COMMUNITY: Sentiment, memes, social

GLOBAL_SENTIMENT: Overall message tone (-1, 0, 1)
IS_FACT: true if verified/official, false if rumor/opinion

EXAMPLES:

Input: "Sold all my Ethereum to buy Pepe"
Output:
{
  "targets": [
    {"name": "Ethereum", "ticker": "ETH", "type": "CHAIN", "sentiment": -1, "impact": 3, "category": "MARKET"},
    {"name": "Pepe", "ticker": "PEPE", "type": "TOKEN", "sentiment": 1, "impact": 3, "category": "MARKET"}
  ],
  "global_sentiment": 0,
  "is_fact": false
}

Input: "Curve Finance hacked due to Vyper compiler bug, $70M stolen"
Output:
{
  "targets": [
    {"name": "Curve Finance", "ticker": "CRV", "type": "PROTOCOL", "sentiment": -1, "impact": 10, "category": "HACK"},
    {"name": "Vyper", "ticker": null, "type": "OTHER", "sentiment": -1, "impact": 9, "category": "HACK"}
  ],
  "global_sentiment": -1,
  "is_fact": true
}

Input: "Bitcoin ETF approved by SEC"
Output:
{
  "targets": [
    {"name": "Bitcoin", "ticker": "BTC", "type": "TOKEN", "sentiment": 1, "impact": 10, "category": "REGULATION"}
  ],
  "global_sentiment": 1,
  "is_fact": true
}

Return ONLY the JSON object. NO markdown code blocks."""


def main():
    data_dir = (Path(__file__).resolve().parents[1] / 'data')

    print(f"\n{'='*80}")
    print(f"📦 PREPARING {NUM_BATCHES} BATCH INPUT FILES FOR OPENAI BATCH API")
    print(f"{'='*80}\n")

    df = pd.read_csv(data_dir / 'btc_messages.csv')
    total_messages = len(df)

    print(f"✅ Loaded {total_messages:,} messages from btc_messages.csv")

    batch_size = total_messages // NUM_BATCHES
    remainder = total_messages % NUM_BATCHES

    print(f"\n🔨 Splitting into {NUM_BATCHES} batches:")
    print(f"   Base size: {batch_size:,} messages/batch")
    if remainder > 0:
        print(f"   Last batch: +{remainder} messages")

    batch_files = []

    for batch_num in range(NUM_BATCHES):
        start_idx = batch_num * batch_size
        if batch_num == NUM_BATCHES - 1:
            end_idx = total_messages
        else:
            end_idx = (batch_num + 1) * batch_size

        batch_df = df.iloc[start_idx:end_idx]
        output_file = data_dir / f'batch_input_{batch_num + 1}.jsonl'

        print(f"\n   📝 Batch {batch_num + 1}/{NUM_BATCHES}: Creating {output_file.name}")
        print(f"      Messages: {len(batch_df):,} (rows {start_idx:,} to {end_idx-1:,})")

        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in batch_df.iterrows():
                message_text = str(row['text'])[:1200]

                batch_request = {
                    "custom_id": f"msg-{idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system", "content": TBSA_SYSTEM_PROMPT},
                            {"role": "user", "content": message_text}
                        ],
                        "temperature": 0,
                        "max_tokens": 800
                    }
                }

                f.write(json.dumps(batch_request, ensure_ascii=False) + '\n')

        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        batch_files.append({
            'file': output_file.name,
            'size_mb': file_size_mb,
            'messages': len(batch_df)
        })
        print(f"      Size: {file_size_mb:.2f} MB")

    print(f"\n{'='*80}")
    print(f"✅ ALL {NUM_BATCHES} BATCH FILES ГОТОВЫ")
    print(f"{'='*80}")

    print(f"\n📊 SUMMARY:")
    total_size = 0
    for i, info in enumerate(batch_files, 1):
        print(f"   Batch {i}: {info['file']:<25} {info['size_mb']:>6.2f} MB   {info['messages']:>6,} msgs")
        total_size += info['size_mb']

    print(f"\n   TOTAL: {total_size:.2f} MB (limit: 200 MB per file)")

    avg_input_tokens = 814
    avg_output_tokens = 300
    total_input_tokens = total_messages * avg_input_tokens
    total_output_tokens = total_messages * avg_output_tokens

    print(f"\n💰 ОЦЕНКА СТОИМОСТИ (все {NUM_BATCHES} батчей):")
    print(f"   Input tokens:  {total_input_tokens/1_000_000:.1f}M × $0.0375/1M = ${total_input_tokens/1_000_000 * 0.0375:.2f}")
    print(f"   Output tokens: {total_output_tokens/1_000_000:.1f}M × $0.15/1M   = ${total_output_tokens/1_000_000 * 0.15:.2f}")
    print(f"   {'─'*50}")
    print(f"   TOTAL: ~${(total_input_tokens/1_000_000 * 0.0375 + total_output_tokens/1_000_000 * 0.15):.2f}")

    print(f"\n⏱️  ОЦЕНКА ВРЕМЕНИ:")
    print(f"   ⚠️  OpenAI Batch API limit: 2M enqueued tokens per organization")
    print(f"   Strategy: Submit batches in groups to stay under limit")
    avg_tokens_per_batch = (total_input_tokens + total_output_tokens) / NUM_BATCHES
    batches_per_group = int(2_000_000 / avg_tokens_per_batch)
    print(f"   Can submit ~{batches_per_group} batches simultaneously")
    print(f"   Completion window: 24 hours (обычно 2-6 часов per batch)")

    print(f"\n➡️  Next step: Run scripts/7_submit_batch.py")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
