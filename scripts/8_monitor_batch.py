"""
👀 Step 3: Monitor status of all batch jobs

Checks status of all 4 batches
Shows progress, completion estimates, and any errors
Auto-refreshes every 60 seconds when run with --watch flag
"""
import os
import json
import time
import argparse
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()


def get_status_emoji(status):
    """Returns emoji for batch status"""
    emoji_map = {
        'validating': '🔍',
        'failed': '❌',
        'in_progress': '⚙️',
        'finalizing': '🏁',
        'completed': '✅',
        'expired': '⏰',
        'cancelling': '🛑',
        'cancelled': '🚫'
    }
    return emoji_map.get(status, '❓')


def format_timestamp(ts):
    """Converts Unix timestamp to readable datetime"""
    if ts is None:
        return "N/A"
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')


def format_duration(seconds):
    """Formats duration in human-readable format"""
    if seconds is None:
        return "N/A"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60

    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m"
    else:
        return f"{seconds}s"


def check_batch_status(client, batch_jobs):
    """Checks status of all batches and returns updated info"""
    updated_jobs = []

    for job in batch_jobs:
        batch = client.batches.retrieve(job['batch_id'])

        updated_job = {
            'batch_number': job['batch_number'],
            'batch_id': batch.id,
            'input_file_id': job['input_file_id'],
            'status': batch.status,
            'created_at': batch.created_at,
            'in_progress_at': batch.in_progress_at,
            'completed_at': batch.completed_at,
            'failed_at': batch.failed_at,
            'expired_at': batch.expired_at,
            'expires_at': batch.expires_at,
            'output_file_id': batch.output_file_id,
            'error_file_id': batch.error_file_id,
            'request_counts': {
                'total': batch.request_counts.total,
                'completed': batch.request_counts.completed,
                'failed': batch.request_counts.failed
            }
        }

        updated_jobs.append(updated_job)

    return updated_jobs


def display_status(batch_jobs):
    """Displays formatted status table"""
    print(f"\n{'='*100}")
    print(f"👀 BATCH STATUS MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")

    all_completed = True
    any_failed = False
    total_requests = 0
    total_completed = 0
    total_failed = 0

    for job in batch_jobs:
        status = job['status']
        emoji = get_status_emoji(status)
        batch_num = job['batch_number']

        if status != 'completed':
            all_completed = False
        if status == 'failed':
            any_failed = True

        counts = job['request_counts']
        total_requests += counts['total']
        total_completed += counts['completed']
        total_failed += counts['failed']

        print(f"{'─'*100}")
        print(f"BATCH {batch_num}/4: {emoji} {status.upper()}")
        print(f"{'─'*100}")
        print(f"  Batch ID: {job['batch_id']}")
        print(f"  Created:  {format_timestamp(job['created_at'])}")

        if job['in_progress_at']:
            print(f"  Started:  {format_timestamp(job['in_progress_at'])}")

        if job['completed_at']:
            duration = job['completed_at'] - job['created_at']
            print(f"  Completed: {format_timestamp(job['completed_at'])} (took {format_duration(duration)})")
        elif job['failed_at']:
            print(f"  Failed:    {format_timestamp(job['failed_at'])}")
        elif job['expired_at']:
            print(f"  Expired:   {format_timestamp(job['expired_at'])}")
        else:
            print(f"  Expires:   {format_timestamp(job['expires_at'])}")

        if counts['total'] > 0:
            progress = (counts['completed'] / counts['total']) * 100
            print(f"\n  Progress: {counts['completed']:,}/{counts['total']:,} requests ({progress:.1f}%)")

            if counts['failed'] > 0:
                print(f"  Failed:   {counts['failed']:,} requests")

            if status == 'in_progress' and counts['completed'] > 0 and job['in_progress_at']:
                elapsed = time.time() - job['in_progress_at']
                rate = counts['completed'] / elapsed
                remaining = counts['total'] - counts['completed']
                eta_seconds = remaining / rate if rate > 0 else 0
                print(f"  Rate:     {rate:.1f} requests/sec")
                print(f"  ETA:      ~{format_duration(int(eta_seconds))}")

        if job['output_file_id']:
            print(f"\n  ✅ Output file: {job['output_file_id']}")

        if job['error_file_id']:
            print(f"  ⚠️  Error file: {job['error_file_id']}")

        print()

    print(f"{'='*100}")
    print(f"📊 OVERALL SUMMARY")
    print(f"{'='*100}")
    print(f"  Total requests:   {total_requests:,}")
    print(f"  Completed:        {total_completed:,} ({total_completed/total_requests*100:.1f}%)" if total_requests > 0 else "  Completed:        0")
    print(f"  Failed:           {total_failed:,}")

    if all_completed:
        print(f"\n  ✅ ALL BATCHES COMPLETED!")
        print(f"  ➡️  Next step: python scripts/9_process_batch_results.py")
    elif any_failed:
        print(f"\n  ⚠️  Some batches failed - check error files")
    else:
        print(f"\n  ⏳ Batches still processing...")

    print(f"{'='*100}\n")

    return all_completed


def main():
    parser = argparse.ArgumentParser(description='Monitor OpenAI Batch API jobs')
    parser.add_argument('--watch', action='store_true', help='Auto-refresh every 60 seconds')
    parser.add_argument('--interval', type=int, default=60, help='Refresh interval in seconds (default: 60)')
    args = parser.parse_args()

    data_dir = (Path(__file__).resolve().parents[1] / 'data')
    jobs_file = data_dir / 'batch_jobs.json'

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

    if args.watch:
        print(f"👀 Watch mode enabled - refreshing every {args.interval} seconds")
        print(f"   Press Ctrl+C to stop\n")

        try:
            while True:
                updated_jobs = check_batch_status(client, batch_jobs)
                all_completed = display_status(updated_jobs)

                with open(jobs_file, 'w', encoding='utf-8') as f:
                    json.dump(updated_jobs, f, indent=2, ensure_ascii=False)

                if all_completed:
                    print("✅ All batches completed - exiting watch mode")
                    break

                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n\n⏸️  Monitoring stopped by user")

    else:
        updated_jobs = check_batch_status(client, batch_jobs)
        display_status(updated_jobs)

        with open(jobs_file, 'w', encoding='utf-8') as f:
            json.dump(updated_jobs, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
