"""
🎯 Target-Based Sentiment Analysis (TBSA) Model Battle
Tests GPT-4o-mini vs Llama-3.1-8b on entity-level sentiment extraction

Schema v4: name, ticker, type, sentiment, impact, category
"""
import pandas as pd
import json
import os
import time
from openai import OpenAI
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"

MODELS_TO_TEST = [
    "openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct",
]

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


def analyze_with_tbsa(client, model_name, text, max_retries=2):
    """Analyzes message using TBSA schema v4"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": TBSA_SYSTEM_PROMPT},
                    {"role": "user", "content": text[:1200]}
                ],
                temperature=0,
                max_tokens=800,
                timeout=60.0
            )

            result_text = response.choices[0].message.content.strip()

            if '```' in result_text:
                parts = result_text.split('```')
                for part in parts:
                    part = part.strip()
                    if part.startswith('json'):
                        part = part[4:].strip()
                    if part.startswith('{'):
                        result_text = part
                        break

            result_json = json.loads(result_text)

            required = ['targets', 'global_sentiment', 'is_fact']
            if not all(field in result_json for field in required):
                missing = set(required) - set(result_json.keys())
                return {"error": f"Missing fields: {missing}"}

            if not isinstance(result_json['targets'], list):
                return {"error": "targets must be array"}

            for idx, target in enumerate(result_json['targets']):
                target_required = ['name', 'ticker', 'type', 'sentiment', 'impact', 'category']
                if not all(field in target for field in target_required):
                    missing = set(target_required) - set(target.keys())
                    return {"error": f"Target {idx}: missing {missing}"}

                if target['sentiment'] not in [-1, 0, 1]:
                    return {"error": f"Target {idx}: invalid sentiment {target['sentiment']}"}

                if not (1 <= target['impact'] <= 10):
                    return {"error": f"Target {idx}: invalid impact {target['impact']}"}

                valid_types = ['PROTOCOL', 'TOKEN', 'CHAIN', 'CEX', 'OTHER']
                if target['type'] not in valid_types:
                    return {"error": f"Target {idx}: invalid type {target['type']}"}

            if result_json['global_sentiment'] not in [-1, 0, 1]:
                return {"error": f"Invalid global_sentiment: {result_json['global_sentiment']}"}

            return result_json

        except json.JSONDecodeError as e:
            if attempt == max_retries - 1:
                return {"error": f"JSON parse failed: {str(e)[:50]}"}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e)[:100]}

        time.sleep(3)

    return {"error": "Max retries exceeded"}


def calculate_tbsa_metrics(result):
    """Calculates TBSA-specific metrics from result"""
    if 'error' in result:
        return {
            'num_targets': 0,
            'has_conflict': False,
            'avg_impact': 0,
            'types_found': '',
            'error': result['error']
        }

    targets = result.get('targets', [])
    num_targets = len(targets)

    sentiments = [t['sentiment'] for t in targets if 'sentiment' in t]
    has_conflict = len(set(sentiments)) > 1 if len(sentiments) > 1 else False

    impacts = [t['impact'] for t in targets if 'impact' in t]
    avg_impact = sum(impacts) / len(impacts) if impacts else 0

    types = [t['type'] for t in targets if 'type' in t]
    types_found = '|'.join(set(types)) if types else ''

    return {
        'num_targets': num_targets,
        'has_conflict': has_conflict,
        'avg_impact': avg_impact,
        'types_found': types_found,
        'error': None
    }


def test_models_on_message(client, message_text, message_id, test_category):
    """Tests both models on a single message"""
    print(f"\n{'─'*80}")
    print(f"📧 Message #{message_id} | Category: {test_category}")
    print(f"   Text preview: {message_text[:80].replace(chr(10), ' ')}...")

    results = {}

    for model in MODELS_TO_TEST:
        model_short = model.split('/')[1]
        print(f"\n   🤖 Testing {model_short}...")

        result = analyze_with_tbsa(client, model, message_text)
        results[model_short] = result

        if 'error' in result:
            print(f"      ❌ ERROR: {result['error']}")
        else:
            targets = result.get('targets', [])
            print(f"      ✅ Found {len(targets)} targets | Global sentiment: {result['global_sentiment']:+d}")
            for t in targets[:3]:
                print(f"         • {t['name']} ({t['type']}) → S:{t['sentiment']:+d} I:{t['impact']} [{t['category']}]")

        time.sleep(3)

    return results


def main():
    start_time = datetime.now()
    data_dir = (Path(__file__).resolve().parents[1] / 'data')

    print(f"\n{'='*80}")
    print(f"🎯 TARGET-BASED SENTIMENT ANALYSIS (TBSA) MODEL BATTLE")
    print(f"{'='*80}")
    print(f"Schema: v4 (name, ticker, type, sentiment, impact, category)")
    print(f"Models: {len(MODELS_TO_TEST)}")
    for m in MODELS_TO_TEST:
        print(f"  • {m}")
    print(f"{'='*80}\n")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    test_df = pd.read_csv(data_dir / 'difficult_test_cases.csv')

    print(f"📊 Test dataset: {len(test_df)} messages")
    print(f"Categories: {dict(test_df['category'].value_counts())}\n")

    detailed_results = []

    for idx, row in test_df.iterrows():
        model_results = test_models_on_message(
            client,
            row['text'],
            idx + 1,
            row['category']
        )

        base_row = {
            'message_id': idx + 1,
            'test_category': row['category'],
            'date': row['date'],
            'channel': row['channel'],
            'text': str(row['text'])[:300],
        }

        for model_short, result in model_results.items():
            metrics = calculate_tbsa_metrics(result)

            base_row[f'{model_short}_num_targets'] = metrics['num_targets']
            base_row[f'{model_short}_has_conflict'] = metrics['has_conflict']
            base_row[f'{model_short}_avg_impact'] = metrics['avg_impact']
            base_row[f'{model_short}_types'] = metrics['types_found']
            base_row[f'{model_short}_global_sentiment'] = result.get('global_sentiment', None) if 'error' not in result else None
            base_row[f'{model_short}_is_fact'] = result.get('is_fact', None) if 'error' not in result else None
            base_row[f'{model_short}_error'] = metrics['error']

            targets_json = json.dumps(result.get('targets', []), ensure_ascii=False) if 'error' not in result else ''
            base_row[f'{model_short}_targets_json'] = targets_json

        detailed_results.append(base_row)

        if idx < len(test_df) - 1:
            time.sleep(2)

    detailed_df = pd.DataFrame(detailed_results)
    detailed_output = data_dir / 'tbsa_battle_results.csv'
    detailed_df.to_csv(detailed_output, index=False)

    print(f"\n{'='*80}")
    print(f"📊 TBSA METRICS COMPARISON")
    print(f"{'='*80}\n")

    summary = []

    for model in MODELS_TO_TEST:
        model_short = model.split('/')[1]

        num_targets_col = f'{model_short}_num_targets'
        conflict_col = f'{model_short}_has_conflict'
        error_col = f'{model_short}_error'

        total = len(detailed_df)
        valid = detailed_df[error_col].isna().sum()
        success_rate = (valid / total * 100) if total > 0 else 0

        avg_targets = detailed_df[num_targets_col].mean()

        conflicts = detailed_df[conflict_col].sum()
        conflict_rate = (conflicts / valid * 100) if valid > 0 else 0

        types_col = f'{model_short}_types'
        has_types = (detailed_df[types_col].notna() & (detailed_df[types_col] != '')).sum()
        type_classification_rate = (has_types / valid * 100) if valid > 0 else 0

        summary.append({
            'model': model_short,
            'success_rate': success_rate,
            'avg_targets': avg_targets,
            'conflict_rate': conflict_rate,
            'type_classification_rate': type_classification_rate,
        })

    summary_df = pd.DataFrame(summary)
    summary_output = data_dir / 'tbsa_battle_summary.csv'
    summary_df.to_csv(summary_output, index=False)

    print(f"{'Model':<30} {'Success%':<10} {'Avg Targets':<15} {'Conflict%':<12} {'Type Class%':<12}")
    print("─" * 80)
    for _, row in summary_df.iterrows():
        print(f"{row['model']:<30} {row['success_rate']:>8.1f}%  {row['avg_targets']:>13.2f}  {row['conflict_rate']:>10.1f}%  {row['type_classification_rate']:>10.1f}%")

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*80}")
    print(f"✅ TBSA BATTLE ЗАВЕРШЕН")
    print(f"{'='*80}")
    print(f"📁 Detailed results: {detailed_output}")
    print(f"📊 Summary: {summary_output}")
    print(f"⏱️  Time: {elapsed/60:.1f} минут")

    print(f"\n💡 KEY INSIGHTS:")
    print(f"   • Conflict Rate HIGH = Good (model resolves opposing sentiments)")
    print(f"   • Avg Targets shows entity extraction capability")
    print(f"   • Type Classification Rate shows semantic understanding")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
