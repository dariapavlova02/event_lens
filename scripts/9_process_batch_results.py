"""
📥 Step 4: Download and process batch results

Downloads output files from all 4 completed batches
Parses TBSA results from JSONL format
Combines results with original messages
Creates final CSV: btc_messages_tbsa.csv
Generates analysis report
"""
import os
import json
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


def parse_batch_output(output_content):
    """Parses batch output JSONL and extracts TBSA results"""
    results = []
    errors = []

    for line in output_content.strip().split('\n'):
        if not line.strip():
            continue

        try:
            batch_response = json.loads(line)
            custom_id = batch_response['custom_id']
            msg_idx = int(custom_id.split('-')[1])

            if batch_response.get('error'):
                errors.append({
                    'msg_idx': msg_idx,
                    'error_type': 'api_error',
                    'error': str(batch_response['error']),
                    'content': None
                })
                continue

            response_body = batch_response['response']['body']
            content = response_body['choices'][0]['message']['content']
            finish_reason = response_body['choices'][0].get('finish_reason', 'unknown')

            if finish_reason == 'length':
                errors.append({
                    'msg_idx': msg_idx,
                    'error_type': 'truncated',
                    'error': 'Response truncated due to max_tokens limit',
                    'content': content
                })
                continue

            content = content.strip()
            if '```' in content:
                parts = content.split('```')
                for part in parts:
                    part = part.strip()
                    if part.startswith('json'):
                        part = part[4:].strip()
                    if part.startswith('{'):
                        content = part
                        break

            if not content.endswith('}'):
                errors.append({
                    'msg_idx': msg_idx,
                    'error_type': 'unterminated_json',
                    'error': 'JSON response does not end with }',
                    'content': content
                })
                continue

            tbsa_result = json.loads(content)

            required = ['targets', 'global_sentiment', 'is_fact']
            if not all(field in tbsa_result for field in required):
                missing = set(required) - set(tbsa_result.keys())
                errors.append({
                    'msg_idx': msg_idx,
                    'error_type': 'missing_fields',
                    'error': f'Missing fields: {missing}',
                    'content': content
                })
                continue

            results.append({
                'msg_idx': msg_idx,
                'tbsa_result': tbsa_result,
                'usage': response_body.get('usage', {})
            })

        except json.JSONDecodeError as e:
            errors.append({
                'msg_idx': msg_idx if 'msg_idx' in locals() else -1,
                'error_type': 'json_decode_error',
                'error': f'JSON parse failed: {str(e)[:100]}',
                'content': content if 'content' in locals() else None
            })
        except Exception as e:
            errors.append({
                'msg_idx': msg_idx if 'msg_idx' in locals() else -1,
                'error_type': 'unknown',
                'error': str(e)[:100],
                'content': None
            })

    return results, errors


def main():
    data_dir = (Path(__file__).resolve().parents[1] / 'data')
    jobs_file = data_dir / 'batch_jobs.json'

    print(f"\n{'='*80}")
    print(f"📥 DOWNLOADING AND PROCESSING BATCH RESULTS")
    print(f"{'='*80}\n")

    if not jobs_file.exists():
        print(f"❌ ERROR: {jobs_file} not found!")
        print(f"   Run scripts/7_submit_batch.py first")
        return

    with open(jobs_file, 'r', encoding='utf-8') as f:
        batch_jobs = json.load(f)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in .env file")
        return

    client = OpenAI(api_key=api_key)

    all_results = []
    all_errors = []
    total_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

    for job in batch_jobs:
        batch_num = job['batch_number']
        batch_id = job['batch_id']

        print(f"{'─'*80}")
        print(f"📦 BATCH {batch_num}/4: {batch_id}")

        if job['status'] != 'completed':
            print(f"   ⚠️  Status: {job['status']} - skipping")
            print(f"   Run scripts/8_monitor_batch.py to check status")
            continue

        if not job.get('output_file_id'):
            print(f"   ⚠️  No output file - skipping")
            continue

        print(f"   Downloading output file: {job['output_file_id']}")

        output_content = client.files.content(job['output_file_id'])
        output_text = output_content.read().decode('utf-8')

        output_file = data_dir / f'batch_output_{batch_num}.jsonl'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)

        print(f"   ✅ Saved to: {output_file.name}")

        print(f"   Parsing TBSA results...")
        results, errors = parse_batch_output(output_text)

        print(f"   ✅ Parsed {len(results)} successful results")
        if errors:
            print(f"   ⚠️  Found {len(errors)} errors")

        all_results.extend(results)
        all_errors.extend(errors)

        for r in results:
            usage = r.get('usage', {})
            total_usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
            total_usage['completion_tokens'] += usage.get('completion_tokens', 0)
            total_usage['total_tokens'] += usage.get('total_tokens', 0)

    if not all_results:
        print(f"\n❌ No results to process - all batches incomplete or failed")
        return

    print(f"\n{'─'*80}")
    print(f"📊 COMBINING RESULTS WITH ORIGINAL MESSAGES")
    print(f"{'─'*80}\n")

    print(f"   Loading original messages...")
    df = pd.read_csv(data_dir / 'btc_messages.csv')

    results_dict = {r['msg_idx']: r['tbsa_result'] for r in all_results}

    print(f"   Extracting TBSA fields...")

    df['tbsa_targets_json'] = df.index.map(lambda idx: json.dumps(results_dict.get(idx, {}).get('targets', []), ensure_ascii=False) if idx in results_dict else None)
    df['tbsa_num_targets'] = df.index.map(lambda idx: len(results_dict.get(idx, {}).get('targets', [])) if idx in results_dict else 0)
    df['tbsa_global_sentiment'] = df.index.map(lambda idx: results_dict.get(idx, {}).get('global_sentiment'))
    df['tbsa_is_fact'] = df.index.map(lambda idx: results_dict.get(idx, {}).get('is_fact'))
    df['tbsa_has_result'] = df.index.map(lambda idx: idx in results_dict)

    df['tbsa_target_names'] = df['tbsa_targets_json'].apply(
        lambda x: '|'.join([t['name'] for t in json.loads(x)]) if x and x != '[]' else None
    )

    output_csv = data_dir / 'btc_messages_tbsa.csv'
    df.to_csv(output_csv, index=False)

    print(f"   ✅ Saved to: {output_csv}")

    print(f"\n{'='*80}")
    print(f"✅ PROCESSING COMPLETE")
    print(f"{'='*80}")

    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   Total messages:        {len(df):,}")
    print(f"   Successfully analyzed: {len(all_results):,} ({len(all_results)/len(df)*100:.1f}%)")
    print(f"   Failed/errors:         {len(all_errors):,}")
    print(f"   Avg targets/message:   {df['tbsa_num_targets'].mean():.2f}")

    sentiment_dist = df['tbsa_global_sentiment'].value_counts().sort_index()
    print(f"\n📈 GLOBAL SENTIMENT DISTRIBUTION:")
    for sent, count in sentiment_dist.items():
        if pd.notna(sent):
            label = {-1: 'BEARISH', 0: 'NEUTRAL', 1: 'BULLISH'}.get(int(sent), 'UNKNOWN')
            print(f"   {label} ({int(sent):+d}): {count:,} ({count/len(df)*100:.1f}%)")

    fact_dist = df['tbsa_is_fact'].value_counts()
    print(f"\n📰 FACT vs OPINION:")
    for is_fact, count in fact_dist.items():
        if pd.notna(is_fact):
            label = 'FACT' if is_fact else 'OPINION'
            print(f"   {label}: {count:,} ({count/len(df)*100:.1f}%)")

    print(f"\n💰 TOKEN USAGE:")
    print(f"   Prompt tokens:     {total_usage['prompt_tokens']:,}")
    print(f"   Completion tokens: {total_usage['completion_tokens']:,}")
    print(f"   Total tokens:      {total_usage['total_tokens']:,}")

    actual_cost = (total_usage['prompt_tokens'] / 1_000_000 * 0.0375 +
                   total_usage['completion_tokens'] / 1_000_000 * 0.15)
    print(f"\n💵 ACTUAL COST: ${actual_cost:.2f}")

    if all_errors:
        errors_file = data_dir / 'tbsa_errors.json'
        with open(errors_file, 'w', encoding='utf-8') as f:
            json.dump(all_errors, f, indent=2, ensure_ascii=False)
        print(f"\n⚠️  Errors saved to: {errors_file}")

        print(f"\n   Creating failed messages CSV for retry...")
        df_orig = pd.read_csv(data_dir / 'btc_messages.csv')
        failed_indices = [e['msg_idx'] for e in all_errors if e['msg_idx'] >= 0]

        if failed_indices:
            df_failed = df_orig.iloc[failed_indices].copy()
            df_failed['error_type'] = df_failed.index.map(
                lambda idx: next((e['error_type'] for e in all_errors if e['msg_idx'] == idx), None)
            )
            df_failed['error_message'] = df_failed.index.map(
                lambda idx: next((e['error'] for e in all_errors if e['msg_idx'] == idx), None)
            )

            failed_csv = data_dir / 'tbsa_failed.csv'
            df_failed.to_csv(failed_csv, index=True)
            print(f"   ✅ Saved {len(df_failed)} failed messages to: {failed_csv}")

            error_breakdown = pd.Series([e['error_type'] for e in all_errors]).value_counts()
            print(f"\n   Error breakdown:")
            for err_type, count in error_breakdown.items():
                print(f"      {err_type}: {count}")

    print(f"\n📁 OUTPUT FILES:")
    print(f"   Main results:  {output_csv}")
    print(f"   Raw outputs:   batch_output_1.jsonl ... batch_output_4.jsonl")

    print(f"\n➡️  Next: Analyze results with your preferred method")
    print(f"   Or create visualization script for diploma presentation")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
