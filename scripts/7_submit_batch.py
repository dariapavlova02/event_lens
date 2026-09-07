"""
🚀 Step 2: Submit batch files to OpenAI Batch API

Usage:
  python scripts/7_submit_batch.py 1    # Submit only batch 1
  python scripts/7_submit_batch.py 5    # Submit only batch 5
  python scripts/7_submit_batch.py      # Submit ALL batches (⚠️  use with caution)

Saves batch IDs to batch_jobs.json for monitoring
"""
import os
import sys
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

NUM_BATCHES = 10


def main():
    data_dir = (Path(__file__).resolve().parents[1] / 'data')

    batch_to_submit = None
    if len(sys.argv) > 1:
        try:
            batch_to_submit = int(sys.argv[1])
            if batch_to_submit < 1 or batch_to_submit > NUM_BATCHES:
                print(f"❌ ERROR: Batch number must be between 1 and {NUM_BATCHES}")
                return
        except ValueError:
            print(f"❌ ERROR: Invalid batch number")
            print(f"Usage: python scripts/7_submit_batch.py [1-{NUM_BATCHES}]")
            return

    print(f"\n{'='*80}")
    if batch_to_submit:
        print(f"🚀 SUBMITTING BATCH {batch_to_submit}/{NUM_BATCHES}")
        batches_to_process = [batch_to_submit]
    else:
        print(f"⚠️  WARNING: SUBMITTING ALL {NUM_BATCHES} BATCHES AT ONCE!")
        print(f"   This may exceed OpenAI's 2M token limit")
        print(f"   Recommended: python scripts/7_submit_batch.py 1")
        batches_to_process = list(range(1, NUM_BATCHES + 1))
    print(f"{'='*80}\n")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in .env file")
        print("   Add: OPENAI_API_KEY=sk-proj-... to .env")
        return

    client = OpenAI(api_key=api_key)

    print(f"✅ Connected to OpenAI API\n")

    jobs_file = data_dir / 'batch_jobs.json'
    if jobs_file.exists() and jobs_file.stat().st_size > 0:
        with open(jobs_file, 'r', encoding='utf-8') as f:
            try:
                batch_jobs = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️  Warning: batch_jobs.json is corrupted, starting fresh")
                batch_jobs = []
    else:
        batch_jobs = []

    for batch_num in batches_to_process:
        input_file = data_dir / f'batch_input_{batch_num}.jsonl'

        if not input_file.exists():
            print(f"❌ ERROR: {input_file.name} not found!")
            print(f"   Run scripts/6_prepare_batch_input.py first")
            return

        file_size_mb = input_file.stat().st_size / (1024 * 1024)

        print(f"{'─'*80}")
        print(f"📤 BATCH {batch_num}/{NUM_BATCHES}: {input_file.name}")
        print(f"   Size: {file_size_mb:.2f} MB")

        print(f"\n   Step 1: Uploading file to OpenAI...")

        with open(input_file, 'rb') as f:
            uploaded_file = client.files.create(
                file=f,
                purpose='batch'
            )

        print(f"   ✅ File uploaded: {uploaded_file.id}")

        print(f"\n   Step 2: Creating batch job...")

        batch = client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "batch_number": str(batch_num),
                "total_batches": str(NUM_BATCHES),
                "description": f"TBSA analysis batch {batch_num}/{NUM_BATCHES}"
            }
        )

        print(f"   ✅ Batch created: {batch.id}")
        print(f"   Status: {batch.status}")
        print(f"   Created at: {datetime.fromtimestamp(batch.created_at)}")
        print(f"   Expires at: {datetime.fromtimestamp(batch.expires_at)}")

        batch_jobs.append({
            'batch_number': batch_num,
            'batch_id': batch.id,
            'input_file_id': uploaded_file.id,
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
        })

        print(f"   ✅ Batch {batch_num} submitted!")

    with open(jobs_file, 'w', encoding='utf-8') as f:
        json.dump(batch_jobs, f, indent=4, ensure_ascii=False)

    print(f"\n{'='*80}")
    if batch_to_submit:
        print(f"✅ BATCH {batch_to_submit} SUBMITTED SUCCESSFULLY")
    else:
        print(f"✅ ALL {NUM_BATCHES} BATCHES SUBMITTED SUCCESSFULLY")
    print(f"{'='*80}")

    print(f"\n📊 SUBMITTED BATCHES:")
    for job in batch_jobs:
        if job['batch_number'] in batches_to_process:
            print(f"   Batch {job['batch_number']}: {job['batch_id']} - {job['status']}")

    print(f"\n💾 Batch info saved to: {jobs_file}")

    print(f"\n⏱️  NEXT STEPS:")
    print(f"   1. Monitor progress: python scripts/8_monitor_batch.py --watch")
    print(f"   2. Expected completion: 2-6 hours per batch")
    if batch_to_submit and batch_to_submit < NUM_BATCHES:
        print(f"   3. After completion, submit next batch:")
        print(f"      python scripts/7_submit_batch.py {batch_to_submit + 1}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
