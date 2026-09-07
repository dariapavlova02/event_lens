"""
🤖 Полностью автоматизированная батч-обработка через OpenAI Batch API

Что делает этот скрипт:
1. Разделяет all_messages_unprocessed.csv на N батчей
2. Создает JSONL файлы для каждого батча
3. Отправляет батчи в OpenAI (по 1-2 одновременно)
4. Мониторит статус каждого батча (каждые 60 сек)
5. Скачивает результаты при завершении
6. Автоматически запускает следующий батч
7. Сохраняет промежуточные результаты
8. Продолжает с места остановки при перезапуске

Usage:
  python scripts/15_auto_process_batches.py --batches 40 --concurrent 2
  python scripts/15_auto_process_batches.py --resume  # Продолжить с места остановки
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
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

Return ONLY the JSON object. NO markdown code blocks."""


class AutoBatchProcessor:
    def __init__(self, data_dir, num_batches=40, concurrent_batches=2):
        self.data_dir = Path(data_dir)
        self.num_batches = num_batches
        self.concurrent_batches = concurrent_batches
        self.state_file = self.data_dir / 'auto_batch_state.json'
        self.jobs_file = self.data_dir / 'auto_batch_jobs.json'

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        self.client = OpenAI(api_key=api_key)
        self.state = self.load_state()
        self.jobs = self.load_jobs()

    def load_state(self):
        """Загружает состояние обработки"""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'total_batches': self.num_batches,
            'prepared_batches': [],
            'submitted_batches': [],
            'completed_batches': [],
            'failed_batches': [],
            'processed_batches': [],
            'total_messages': 0,
            'total_processed': 0,
            'total_failed': 0,
            'start_time': None,
            'last_update': None
        }

    def save_state(self):
        """Сохраняет состояние обработки"""
        self.state['last_update'] = datetime.now().isoformat()
        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False)

    def load_jobs(self):
        """Загружает информацию о батч-джобах"""
        if self.jobs_file.exists():
            with open(self.jobs_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

    def save_jobs(self):
        """Сохраняет информацию о батч-джобах"""
        with open(self.jobs_file, 'w', encoding='utf-8') as f:
            json.dump(self.jobs, f, indent=2, ensure_ascii=False)

    def prepare_batches(self):
        """Шаг 1: Подготовка JSONL файлов для всех батчей"""
        print(f"\n{'='*80}")
        print(f"📦 ШАГ 1: ПОДГОТОВКА {self.num_batches} БАТЧЕЙ")
        print(f"{'='*80}\n")

        input_csv = self.data_dir / 'all_messages_unprocessed.csv'
        if not input_csv.exists():
            raise FileNotFoundError(f"{input_csv} not found! Run script 14 first.")

        df = pd.read_csv(input_csv)
        total_messages = len(df)
        self.state['total_messages'] = total_messages

        print(f"✅ Загружено {total_messages:,} необработанных сообщений")

        batch_size = total_messages // self.num_batches
        remainder = total_messages % self.num_batches

        print(f"\n🔨 Разделение на {self.num_batches} батчей:")
        print(f"   Базовый размер: {batch_size:,} сообщений/батч")
        if remainder > 0:
            print(f"   Последний батч: +{remainder} сообщений\n")

        for batch_num in range(1, self.num_batches + 1):
            if batch_num in self.state['prepared_batches']:
                print(f"   ⏭️  Батч {batch_num}/{self.num_batches}: уже подготовлен")
                continue

            start_idx = (batch_num - 1) * batch_size
            if batch_num == self.num_batches:
                end_idx = total_messages
            else:
                end_idx = batch_num * batch_size

            batch_df = df.iloc[start_idx:end_idx]
            output_file = self.data_dir / f'auto_batch_input_{batch_num}.jsonl'

            print(f"   📝 Батч {batch_num}/{self.num_batches}: {len(batch_df):,} сообщений (строки {start_idx:,}-{end_idx-1:,})")

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
            print(f"      ✅ Сохранено: {output_file.name} ({file_size_mb:.2f} MB)")

            self.state['prepared_batches'].append(batch_num)
            self.save_state()

        print(f"\n✅ Все {self.num_batches} батчей подготовлены!")

    def submit_batch(self, batch_num):
        """Отправляет один батч в OpenAI"""
        input_file = self.data_dir / f'auto_batch_input_{batch_num}.jsonl'

        if not input_file.exists():
            raise FileNotFoundError(f"{input_file} not found!")

        print(f"\n   📤 Отправка батча {batch_num}...")

        with open(input_file, 'rb') as f:
            uploaded_file = self.client.files.create(file=f, purpose='batch')

        print(f"      ✅ Файл загружен: {uploaded_file.id}")

        batch = self.client.batches.create(
            input_file_id=uploaded_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "batch_number": str(batch_num),
                "total_batches": str(self.num_batches),
                "description": f"Auto TBSA batch {batch_num}/{self.num_batches}"
            }
        )

        print(f"      ✅ Батч создан: {batch.id}")
        print(f"      Статус: {batch.status}")

        job_info = {
            'batch_number': batch_num,
            'batch_id': batch.id,
            'input_file_id': uploaded_file.id,
            'status': batch.status,
            'created_at': batch.created_at,
            'submitted_at': datetime.now().isoformat(),
            'in_progress_at': batch.in_progress_at,
            'completed_at': batch.completed_at,
            'output_file_id': batch.output_file_id,
            'request_counts': {
                'total': batch.request_counts.total,
                'completed': batch.request_counts.completed,
                'failed': batch.request_counts.failed
            }
        }

        self.jobs.append(job_info)
        self.state['submitted_batches'].append(batch_num)
        self.save_jobs()
        self.save_state()

        return batch.id

    def check_batch_status(self, batch_id):
        """Проверяет статус одного батча"""
        batch = self.client.batches.retrieve(batch_id)
        return batch

    def update_job_status(self, batch_num, batch):
        """Обновляет статус джоба"""
        for job in self.jobs:
            if job['batch_number'] == batch_num:
                job['status'] = batch.status
                job['in_progress_at'] = batch.in_progress_at
                job['completed_at'] = batch.completed_at
                job['output_file_id'] = batch.output_file_id
                job['request_counts'] = {
                    'total': batch.request_counts.total,
                    'completed': batch.request_counts.completed,
                    'failed': batch.request_counts.failed
                }
                break
        self.save_jobs()

    def download_and_process_batch(self, batch_num):
        """Скачивает и обрабатывает результаты батча"""
        job = next((j for j in self.jobs if j['batch_number'] == batch_num), None)
        if not job:
            print(f"      ⚠️  Джоб для батча {batch_num} не найден")
            return False

        if not job.get('output_file_id'):
            print(f"      ⚠️  Нет output_file_id для батча {batch_num}")
            return False

        print(f"      📥 Скачивание результатов...")

        try:
            output_content = self.client.files.content(job['output_file_id'])
            output_text = output_content.read().decode('utf-8')

            output_file = self.data_dir / f'auto_batch_output_{batch_num}.jsonl'
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output_text)

            print(f"      ✅ Результаты сохранены: {output_file.name}")

            self.state['processed_batches'].append(batch_num)

            counts = job['request_counts']
            self.state['total_processed'] += counts['completed']
            self.state['total_failed'] += counts['failed']

            self.save_state()
            return True

        except Exception as e:
            print(f"      ❌ Ошибка при скачивании: {str(e)}")
            return False

    def monitor_active_batches(self):
        """Мониторит все активные батчи"""
        active_batches = [
            j for j in self.jobs
            if j['status'] in ['validating', 'in_progress', 'finalizing']
        ]

        if not active_batches:
            return []

        print(f"\n   🔍 Мониторинг {len(active_batches)} активных батчей...")

        completed = []

        for job in active_batches:
            batch_num = job['batch_number']
            batch = self.check_batch_status(job['batch_id'])
            self.update_job_status(batch_num, batch)

            status_emoji = {
                'validating': '🔍',
                'in_progress': '⚙️',
                'finalizing': '🏁',
                'completed': '✅',
                'failed': '❌'
            }.get(batch.status, '❓')

            counts = batch.request_counts
            progress = (counts.completed / counts.total * 100) if counts.total > 0 else 0

            print(f"      {status_emoji} Батч {batch_num}: {batch.status} - {counts.completed}/{counts.total} ({progress:.1f}%)")

            if batch.status == 'completed':
                print(f"      ✅ Батч {batch_num} завершен!")
                completed.append(batch_num)
                if batch_num not in self.state['completed_batches']:
                    self.state['completed_batches'].append(batch_num)
                    self.save_state()

                if batch_num not in self.state['processed_batches']:
                    self.download_and_process_batch(batch_num)

            elif batch.status == 'failed':
                print(f"      ❌ Батч {batch_num} провалился!")
                if batch_num not in self.state['failed_batches']:
                    self.state['failed_batches'].append(batch_num)
                    self.save_state()

        return completed

    def process_all_batches(self):
        """Главный цикл обработки всех батчей"""
        if not self.state['start_time']:
            self.state['start_time'] = datetime.now().isoformat()
            self.save_state()

        print(f"\n{'='*80}")
        print(f"🤖 ШАГ 2: АВТОМАТИЧЕСКАЯ ОТПРАВКА И МОНИТОРИНГ")
        print(f"{'='*80}\n")
        print(f"   Всего батчей: {self.num_batches}")
        print(f"   Одновременно: {self.concurrent_batches}")
        print(f"   Интервал проверки: 60 секунд\n")

        batches_to_submit = [
            i for i in range(1, self.num_batches + 1)
            if i not in self.state['submitted_batches']
        ]

        print(f"   Осталось отправить: {len(batches_to_submit)} батчей")
        print(f"   Активных батчей: {len([j for j in self.jobs if j['status'] in ['validating', 'in_progress', 'finalizing']])}")

        try:
            while batches_to_submit or len([j for j in self.jobs if j['status'] in ['validating', 'in_progress', 'finalizing']]) > 0:
                active_count = len([j for j in self.jobs if j['status'] in ['validating', 'in_progress', 'finalizing']])

                while active_count < self.concurrent_batches and batches_to_submit:
                    batch_num = batches_to_submit.pop(0)
                    print(f"\n{'─'*80}")
                    print(f"🚀 Отправка батча {batch_num}/{self.num_batches}")
                    self.submit_batch(batch_num)
                    active_count += 1

                completed = self.monitor_active_batches()

                total_completed = len(self.state['completed_batches'])
                total_failed = len(self.state['failed_batches'])
                total_processed_msgs = self.state['total_processed']
                total_messages = self.state['total_messages']

                print(f"\n   📊 Прогресс:")
                print(f"      Батчей завершено: {total_completed}/{self.num_batches}")
                print(f"      Батчей провалилось: {total_failed}")
                print(f"      Сообщений обработано: {total_processed_msgs:,}/{total_messages:,}")

                if total_completed + total_failed >= self.num_batches:
                    print(f"\n   ✅ Все батчи обработаны!")
                    break

                if batches_to_submit or active_count > 0:
                    print(f"\n   ⏳ Ожидание 60 секунд перед следующей проверкой...")
                    time.sleep(60)

        except KeyboardInterrupt:
            print(f"\n\n⏸️  Обработка остановлена пользователем")
            print(f"   Состояние сохранено в {self.state_file}")
            print(f"   Для продолжения запустите: python scripts/15_auto_process_batches.py --resume")
            return

    def combine_all_results(self):
        """Шаг 3: Объединение всех результатов в один CSV"""
        print(f"\n{'='*80}")
        print(f"📊 ШАГ 3: ОБЪЕДИНЕНИЕ ВСЕХ РЕЗУЛЬТАТОВ")
        print(f"{'='*80}\n")

        all_results = []
        all_errors = []

        for batch_num in self.state['completed_batches']:
            output_file = self.data_dir / f'auto_batch_output_{batch_num}.jsonl'
            if not output_file.exists():
                print(f"   ⚠️  Пропуск батча {batch_num} - файл не найден")
                continue

            print(f"   📖 Обработка результатов батча {batch_num}...")

            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue

                    try:
                        batch_response = json.loads(line)
                        custom_id = batch_response['custom_id']
                        msg_idx = int(custom_id.split('-')[1])

                        if batch_response.get('error'):
                            all_errors.append({'msg_idx': msg_idx, 'error': str(batch_response['error'])})
                            continue

                        response_body = batch_response['response']['body']
                        content = response_body['choices'][0]['message']['content'].strip()

                        if '```' in content:
                            parts = content.split('```')
                            for part in parts:
                                part = part.strip()
                                if part.startswith('json'):
                                    part = part[4:].strip()
                                if part.startswith('{'):
                                    content = part
                                    break

                        tbsa_result = json.loads(content)
                        all_results.append({'msg_idx': msg_idx, 'tbsa_result': tbsa_result})

                    except Exception as e:
                        all_errors.append({'msg_idx': -1, 'error': str(e)[:100]})

        print(f"\n   ✅ Всего результатов: {len(all_results):,}")
        print(f"   ⚠️  Всего ошибок: {len(all_errors):,}")

        print(f"\n   📝 Создание финального CSV...")

        df_unprocessed = pd.read_csv(self.data_dir / 'all_messages_unprocessed.csv')
        results_dict = {r['msg_idx']: r['tbsa_result'] for r in all_results}

        df_unprocessed['tbsa_targets_json'] = df_unprocessed.index.map(
            lambda idx: json.dumps(results_dict.get(idx, {}).get('targets', []), ensure_ascii=False) if idx in results_dict else None
        )
        df_unprocessed['tbsa_num_targets'] = df_unprocessed.index.map(
            lambda idx: len(results_dict.get(idx, {}).get('targets', [])) if idx in results_dict else 0
        )
        df_unprocessed['tbsa_global_sentiment'] = df_unprocessed.index.map(
            lambda idx: results_dict.get(idx, {}).get('global_sentiment')
        )
        df_unprocessed['tbsa_is_fact'] = df_unprocessed.index.map(
            lambda idx: results_dict.get(idx, {}).get('is_fact')
        )
        df_unprocessed['tbsa_has_result'] = df_unprocessed.index.map(
            lambda idx: idx in results_dict
        )

        output_csv = self.data_dir / 'all_messages_tbsa.csv'
        df_unprocessed.to_csv(output_csv, index=False)

        print(f"   ✅ Сохранено: {output_csv}")
        print(f"   Записей: {len(df_unprocessed):,}")
        print(f"   С результатами TBSA: {len(all_results):,} ({len(all_results)/len(df_unprocessed)*100:.1f}%)")

        if all_errors:
            errors_file = self.data_dir / 'auto_batch_errors.json'
            with open(errors_file, 'w', encoding='utf-8') as f:
                json.dump(all_errors, f, indent=2, ensure_ascii=False)
            print(f"\n   ⚠️  Ошибки сохранены: {errors_file}")

        print(f"\n{'='*80}")
        print(f"✅ ОБРАБОТКА ЗАВЕРШЕНА!")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Автоматическая батч-обработка через OpenAI API')
    parser.add_argument('--batches', type=int, default=40, help='Количество батчей (default: 40)')
    parser.add_argument('--concurrent', type=int, default=2, help='Одновременных батчей (default: 2)')
    parser.add_argument('--resume', action='store_true', help='Продолжить с места остановки')
    parser.add_argument('--combine-only', action='store_true', help='Только объединить готовые результаты')
    args = parser.parse_args()

    data_dir = (Path(__file__).resolve().parents[1] / 'data')

    print(f"\n{'='*80}")
    print(f"🤖 АВТОМАТИЧЕСКАЯ БАТЧ-ОБРАБОТКА")
    print(f"{'='*80}")
    print(f"Режим: {'ПРОДОЛЖЕНИЕ' if args.resume else 'НОВЫЙ ЗАПУСК'}")
    print(f"Батчей: {args.batches}")
    print(f"Одновременно: {args.concurrent}")
    print(f"{'='*80}\n")

    processor = AutoBatchProcessor(data_dir, num_batches=args.batches, concurrent_batches=args.concurrent)

    if args.combine_only:
        processor.combine_all_results()
        return

    if not args.resume:
        processor.prepare_batches()
    else:
        print(f"♻️  Продолжение с места остановки...")
        print(f"   Подготовлено батчей: {len(processor.state['prepared_batches'])}/{processor.num_batches}")
        print(f"   Отправлено батчей: {len(processor.state['submitted_batches'])}/{processor.num_batches}")
        print(f"   Завершено батчей: {len(processor.state['completed_batches'])}/{processor.num_batches}")

    processor.process_all_batches()

    if len(processor.state['completed_batches']) == processor.num_batches:
        processor.combine_all_results()
    else:
        print(f"\n⚠️  Не все батчи завершены. Для объединения результатов запустите:")
        print(f"   python scripts/15_auto_process_batches.py --combine-only")


if __name__ == "__main__":
    main()
