"""
🔄 Step 5: Retry failed messages from batch processing

Reads tbsa_failed.csv (created by script 9)
Retries failed messages with increased max_tokens=1000
Uses OpenAI Chat Completions API (not batch, for immediate retry)
Merges successful retries back into btc_messages_tbsa.csv
"""
import os
import json
import pandas as pd
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

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


def analyze_with_tbsa_retry(client, text, max_retries=2):
    """Retries failed message with increased max_tokens"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": TBSA_SYSTEM_PROMPT},
                    {"role": "user", "content": text[:1200]}
                ],
                temperature=0,
                max_tokens=1000,
                timeout=60.0
            )

            result_text = response.choices[0].message.content.strip()
            finish_reason = response.choices[0].finish_reason

            if finish_reason == 'length':
                return {"error": "still_truncated", "content": result_text}

            if '```' in result_text:
                parts = result_text.split('```')
                for part in parts:
                    part = part.strip()
                    if part.startswith('json'):
                        part = part[4:].strip()
                    if part.startswith('{'):
                        result_text = part
                        break

            if not result_text.endswith('}'):
                return {"error": "unterminated_json", "content": result_text}

            result_json = json.loads(result_text)

            required = ['targets', 'global_sentiment', 'is_fact']
            if not all(field in result_json for field in required):
                missing = set(required) - set(result_json.keys())
                return {"error": f"missing_fields: {missing}"}

            return {
                "success": True,
                "result": result_json,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }

        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                return {"error": f"json_decode: {str(e)[:50]}"}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e)[:100]}

        time.sleep(3)

    return {"error": "max_retries_exceeded"}


def main():
    data_dir = (Path(__file__).resolve().parents[1] / 'data')
    failed_csv = data_dir / 'tbsa_failed.csv'

    print(f"\n{'='*80}")
    print(f"🔄 RETRYING FAILED MESSAGES")
    print(f"{'='*80}\n")

    if not failed_csv.exists():
        print(f"❌ ERROR: {failed_csv} not found!")
        print(f"   No failed messages to retry. Run scripts/9_process_batch_results.py first.")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in .env file")
        return

    client = OpenAI(api_key=api_key)

    df_failed = pd.read_csv(failed_csv, index_col=0)
    total_failed = len(df_failed)

    print(f"📊 Found {total_failed} failed messages to retry")
    print(f"   Error types: {dict(df_failed['error_type'].value_counts())}\n")

    retry_results = []
    retry_errors = []
    total_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    print(f"{'─'*80}")
    print(f"🚀 STARTING RETRY (max_tokens=1000)")
    print(f"{'─'*80}\n")

    for idx, row in df_failed.iterrows():
        msg_idx = idx
        text = row['text']

        print(f"   [{df_failed.index.get_loc(idx)+1}/{total_failed}] Message #{msg_idx}: {text[:60]}...")

        result = analyze_with_tbsa_retry(client, text)

        if result.get('success'):
            retry_results.append({
                'msg_idx': msg_idx,
                'tbsa_result': result['result'],
                'usage': result['usage']
            })
            total_usage['prompt_tokens'] += result['usage']['prompt_tokens']
            total_usage['completion_tokens'] += result['usage']['completion_tokens']
            total_usage['total_tokens'] += result['usage']['total_tokens']
            print(f"      ✅ SUCCESS: {len(result['result']['targets'])} targets")
        else:
            retry_errors.append({
                'msg_idx': msg_idx,
                'error_type': result.get('error', 'unknown'),
                'error': str(result)
            })
            print(f"      ❌ FAILED: {result.get('error', 'unknown')}")

        time.sleep(1)

    print(f"\n{'='*80}")
    print(f"📊 RETRY RESULTS")
    print(f"{'='*80}")

    success_count = len(retry_results)
    still_failed = len(retry_errors)

    print(f"\n   Retry success: {success_count}/{total_failed} ({success_count/total_failed*100:.1f}%)")
    print(f"   Still failed:  {still_failed}/{total_failed} ({still_failed/total_failed*100:.1f}%)")

    if success_count > 0:
        print(f"\n{'─'*80}")
        print(f"💾 MERGING SUCCESSFUL RETRIES INTO btc_messages_tbsa.csv")
        print(f"{'─'*80}\n")

        main_csv = data_dir / 'btc_messages_tbsa.csv'
        if not main_csv.exists():
            print(f"   ⚠️  {main_csv} not found, creating new file...")
            df_orig = pd.read_csv(data_dir / 'btc_messages.csv')
            df_orig['tbsa_targets_json'] = None
            df_orig['tbsa_num_targets'] = 0
            df_orig['tbsa_global_sentiment'] = None
            df_orig['tbsa_is_fact'] = None
            df_orig['tbsa_has_result'] = False
            df_orig['tbsa_target_names'] = None
        else:
            df_orig = pd.read_csv(main_csv)

        results_dict = {r['msg_idx']: r['tbsa_result'] for r in retry_results}

        for msg_idx, tbsa_result in results_dict.items():
            df_orig.at[msg_idx, 'tbsa_targets_json'] = json.dumps(tbsa_result.get('targets', []), ensure_ascii=False)
            df_orig.at[msg_idx, 'tbsa_num_targets'] = len(tbsa_result.get('targets', []))
            df_orig.at[msg_idx, 'tbsa_global_sentiment'] = tbsa_result.get('global_sentiment')
            df_orig.at[msg_idx, 'tbsa_is_fact'] = tbsa_result.get('is_fact')
            df_orig.at[msg_idx, 'tbsa_has_result'] = True

            target_names = '|'.join([t['name'] for t in tbsa_result.get('targets', [])])
            df_orig.at[msg_idx, 'tbsa_target_names'] = target_names if target_names else None

        df_orig.to_csv(main_csv, index=False)
        print(f"   ✅ Updated {success_count} messages in {main_csv}")

    if retry_errors:
        still_failed_csv = data_dir / 'tbsa_still_failed.csv'
        df_still_failed = df_failed.loc[[e['msg_idx'] for e in retry_errors]].copy()
        df_still_failed['retry_error_type'] = df_still_failed.index.map(
            lambda idx: next((e['error_type'] for e in retry_errors if e['msg_idx'] == idx), None)
        )
        df_still_failed.to_csv(still_failed_csv, index=True)
        print(f"\n   ⚠️  {still_failed} messages still failed, saved to: {still_failed_csv}")

    print(f"\n💰 RETRY COST:")
    print(f"   Prompt tokens:     {total_usage['prompt_tokens']:,}")
    print(f"   Completion tokens: {total_usage['completion_tokens']:,}")
    print(f"   Total tokens:      {total_usage['total_tokens']:,}")

    retry_cost = (total_usage['prompt_tokens'] / 1_000_000 * 0.075 +
                  total_usage['completion_tokens'] / 1_000_000 * 0.30)
    print(f"   Cost: ${retry_cost:.2f} (regular API pricing, not batch)")

    print(f"\n{'='*80}")
    print(f"✅ RETRY COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
