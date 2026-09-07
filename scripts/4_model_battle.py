"""
🏆 Model Battle V2: 11 Free LLM моделей на OpenRouter
С извлечением entities и сравнением по сообщениям
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
    "meta-llama/llama-3.2-3b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "mistralai/mistral-7b-instruct",
    "anthropic/claude-3-haiku",
    "openai/gpt-4o-mini",
]

SYSTEM_PROMPT = """You are a senior DeFi risk analyst for Lynoxis security platform.

Analyze the provided Telegram message (which may be in English, Russian, or mixed slang).

Return ONLY a valid JSON object with these fields:

1. "sentiment": integer (-1 to 1)
   - -1 = Bearish/Negative/Risk
   - 0 = Neutral/Unclear
   - 1 = Bullish/Positive/Growth

2. "impact_score": integer (1 to 10)
   - 1-2: Spam, ads, irrelevant chatter
   - 3-4: Personal opinions, minor questions
   - 5-6: Notable news, protocol updates
   - 7-8: Major announcements, significant price moves
   - 9-10: Critical hacks, SEC decisions, systemic events

3. "category": string (one of)
   - "HACK" (exploits, vulnerabilities)
   - "REGULATION" (legal, SEC)
   - "TECHNICAL" (TA, on-chain metrics)
   - "COMMUNITY" (sentiment, memes)
   - "MARKET" (volumes, listings)
   - "SPAM" (ads, scams, referral links)

4. "entities": array of strings
   - Extract up to 3 key protocols, tokens, or entities mentioned (e.g., ["Curve", "CRV", "Vyper"])
   - Use standardized names (e.g., "Bitcoin" not "BTC", "Ethereum" not "ETH")
   - If none found, return []

5. "is_fact": boolean
   - true if official announcement or verified fact
   - false if rumor, opinion, or speculation

6. "reasoning": string (max 15 words explaining your decision)

Return ONLY the JSON object. Do NOT use markdown code blocks.

Example:
{"sentiment": -1, "impact_score": 9, "category": "HACK", "entities": ["Curve", "Vyper"], "is_fact": true, "reasoning": "Confirmed exploit in Curve pools due to Vyper bug"}"""


def analyze_message(client, model_name, text, max_retries=2):
    """Анализирует одно сообщение с помощью модели"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text[:800]}
                ],
                temperature=0,
                max_tokens=200,
                timeout=45.0
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

            required = ['sentiment', 'impact_score', 'category', 'entities', 'is_fact', 'reasoning']
            if all(field in result_json for field in required):
                if result_json['sentiment'] not in [-1, 0, 1]:
                    return {"error": f"Invalid sentiment: {result_json['sentiment']}"}
                if not (1 <= result_json['impact_score'] <= 10):
                    return {"error": f"Invalid impact_score: {result_json['impact_score']}"}
                if not isinstance(result_json['entities'], list):
                    return {"error": "entities must be array"}

                return result_json
            else:
                missing = set(required) - set(result_json.keys())
                return {"error": f"Missing: {missing}"}

        except json.JSONDecodeError:
            if attempt == max_retries - 1:
                return {"error": "JSON parse failed"}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": str(e)[:80]}

        time.sleep(2)

    return {"error": "Max retries"}


def test_all_models_on_message(client, message_text, message_id):
    """Тестирует модели на сообщении ПОСЛЕДОВАТЕЛЬНО"""

    def test_one(model_name):
        result = analyze_message(client, model_name, message_text)
        model_short = model_name.split('/')[1].split(':')[0]

        if 'error' in result:
            print(f"    ❌ {model_short[:20]:<20}: {result['error'][:50]}")
            return (model_short, result)
        else:
            ent = ','.join(result['entities'][:2]) if result['entities'] else 'none'
            print(f"    ✅ {model_short[:20]:<20}: S={result['sentiment']:+2d} I={result['impact_score']:2d} {result['category'][:4]} [{ent[:15]}]")
            return (model_short, result)

    print(f"\n  🔬 Тест сообщения #{message_id}...")
    results = []
    for model in MODELS_TO_TEST:
        result = test_one(model)
        results.append(result)
        time.sleep(2)

    return dict(results)


def main():
    start_time = datetime.now()
    data_dir = (Path(__file__).resolve().parents[1] / 'data')

    print("DEBUG: Creating client...")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print("DEBUG: Loading CSV...")
    test_df = pd.read_csv(data_dir / 'difficult_test_cases.csv')

    print(f"\n{'='*80}")
    print(f"🏆 БАТЛ LLM МОДЕЛЕЙ - ШИРОКИЙ ФОРМАТ")
    print(f"{'='*80}")
    print(f"Сообщений: {len(test_df)}")
    print(f"Моделей: {len(MODELS_TO_TEST)}")
    print(f"Категории: {dict(test_df['category'].value_counts())}\n")
    print("DEBUG: Starting main loop...")

    wide_results = []

    for idx, row in test_df.iterrows():
        print(f"\n{'─'*80}")
        print(f"[{idx+1}/{len(test_df)}] Категория: {row['category']} | {row['channel']}")
        text_preview = str(row['text'])[:100].replace('\n', ' ')
        print(f"  Текст: {text_preview}...")

        model_responses = test_all_models_on_message(client, row['text'], idx+1)

        wide_row = {
            'message_id': int(idx) + 1,
            'test_category': row['category'],
            'date': row['date'],
            'channel': row['channel'],
            'text': str(row['text'])[:300],
        }

        for model_short, response in model_responses.items():
            if 'error' in response:
                wide_row[f'{model_short}_sentiment'] = None
                wide_row[f'{model_short}_impact'] = None
                wide_row[f'{model_short}_category'] = None
                wide_row[f'{model_short}_entities'] = None
                wide_row[f'{model_short}_is_fact'] = None
                wide_row[f'{model_short}_reasoning'] = response['error']
            else:
                wide_row[f'{model_short}_sentiment'] = response['sentiment']
                wide_row[f'{model_short}_impact'] = response['impact_score']
                wide_row[f'{model_short}_category'] = response['category']
                wide_row[f'{model_short}_entities'] = '|'.join(response['entities']) if response['entities'] else ''
                wide_row[f'{model_short}_is_fact'] = response['is_fact']
                wide_row[f'{model_short}_reasoning'] = response['reasoning']

        wide_results.append(wide_row)

        if int(idx) < len(test_df) - 1:
            time.sleep(3)

    wide_df = pd.DataFrame(wide_results)
    wide_output = data_dir / 'model_battle_wide.csv'
    wide_df.to_csv(wide_output, index=False)

    print(f"\n{'='*80}")
    print(f"📊 АНАЛИЗ РЕЗУЛЬТАТОВ")
    print(f"{'='*80}\n")

    summary = []

    for model in MODELS_TO_TEST:
        model_short = model.split('/')[1].split(':')[0]

        sentiment_col = f'{model_short}_sentiment'
        impact_col = f'{model_short}_impact'

        if sentiment_col not in wide_df.columns:
            continue

        total = len(wide_df)
        valid = wide_df[sentiment_col].notna().sum()
        success_rate = (valid / total * 100) if total > 0 else 0

        impacts = wide_df[impact_col].dropna()
        avg_impact = impacts.mean() if len(impacts) > 0 else 0
        std_impact = impacts.std() if len(impacts) > 0 else 0

        sentiments = wide_df[sentiment_col].dropna()
        pos = (sentiments == 1).sum()
        neu = (sentiments == 0).sum()
        neg = (sentiments == -1).sum()

        entities_col = f'{model_short}_entities'
        if entities_col in wide_df.columns:
            has_entities = (wide_df[entities_col].notna() & (wide_df[entities_col] != '')).sum()
            entity_rate = (has_entities / valid * 100) if valid > 0 else 0
        else:
            entity_rate = 0

        summary.append({
            'model': model_short,
            'success_rate': success_rate,
            'avg_impact': avg_impact,
            'std_impact': std_impact,
            'entity_rate': entity_rate,
            'sent_pos': pos,
            'sent_neu': neu,
            'sent_neg': neg,
        })

    summary_df = pd.DataFrame(summary).sort_values('success_rate', ascending=False)

    print("🏆 ТОП-5 МОДЕЛЕЙ\n")
    print(f"{'Модель':<35} {'Успех':<8} {'Avg I':<8} {'Std I':<8} {'Entity%':<8}")
    print("─" * 80)

    for idx, row in summary_df.head(5).iterrows():
        print(f"{row['model']:<35} {row['success_rate']:>6.1f}%  {row['avg_impact']:>6.2f}  {row['std_impact']:>6.2f}  {row['entity_rate']:>6.1f}%")

    summary_file = data_dir / 'model_battle_summary.csv'
    summary_df.to_csv(summary_file, index=False)

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*80}")
    print(f"✅ БАТЛ ЗАВЕРШЕН")
    print(f"{'='*80}")
    print(f"📁 Широкая таблица: {wide_output}")
    print(f"📊 Сводка по моделям: {summary_file}")
    print(f"⏱️  Время: {elapsed/60:.1f} минут")
    print(f"\n💡 Лучшая модель: {summary_df.iloc[0]['model']}")
    print(f"   - Успех: {summary_df.iloc[0]['success_rate']:.1f}%")
    print(f"   - Avg Impact: {summary_df.iloc[0]['avg_impact']:.2f}")
    print(f"   - Entity extraction: {summary_df.iloc[0]['entity_rate']:.1f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
