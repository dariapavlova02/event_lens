import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

data_dir = Path(os.environ["TELEGRAM_EXPORT_DIR"])
output_dir = (Path(__file__).resolve().parents[1] / 'data')
output_dir.mkdir(parents=True, exist_ok=True)

channels = ['WatcherGuru', 'rickler_feed', 'dlnewsinfo']
btc_keywords = ['btc', 'bitcoin', 'биткоин']

print('=== Извлечение BTC сообщений ===\n')

all_messages = []

for channel in channels:
    print(f'Обработка {channel}...')
    channel_dir = data_dir / channel
    batch_files = sorted(channel_dir.glob('batch_*.json'))

    channel_count = 0

    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            messages = json.load(f)

        for msg in messages:
            text = msg.get('text', '')
            if not text:
                continue

            if any(kw in text.lower() for kw in btc_keywords) and len(text) >= 30:
                all_messages.append({
                    'date': msg['date'],
                    'channel': msg['channel'],
                    'text': text,
                    'views': msg.get('views', 0),
                    'forwards': msg.get('forwards', 0),
                    'replies': msg.get('replies', 0)
                })
                channel_count += 1

    print(f'  Найдено: {channel_count:,} сообщений')

print(f'\n=== Создание датасета ===')
print(f'Всего сообщений: {len(all_messages):,}')

df = pd.DataFrame(all_messages)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

print(f'\nПериод данных:')
print(f'  От: {df["date"].min()}')
print(f'  До: {df["date"].max()}')
print(f'  Дней: {(df["date"].max() - df["date"].min()).days}')

print(f'\nУдаление дубликатов...')
original_count = len(df)
df = df.drop_duplicates(subset=['text'], keep='first')
duplicates_removed = original_count - len(df)
print(f'  Удалено: {duplicates_removed:,} ({duplicates_removed/original_count*100:.1f}%)')
print(f'  Осталось: {len(df):,}')

output_file = output_dir / 'btc_messages.csv'
df.to_csv(output_file, index=False)
print(f'\n✅ Сохранено: {output_file}')

print(f'\nСтатистика по каналам:')
print(df['channel'].value_counts())

print(f'\nРаспределение сообщений по дням:')
daily_counts = df.groupby(df['date'].dt.date).size()
print(f'  Медиана: {daily_counts.median():.0f} сообщений/день')
print(f'  Среднее: {daily_counts.mean():.0f} сообщений/день')
print(f'  Макс: {daily_counts.max():.0f} сообщений/день')
