import os
import json
import pandas as pd
from pathlib import Path

data_dir = Path(os.environ["TELEGRAM_EXPORT_DIR"])
output_dir = (Path(__file__).resolve().parents[1] / 'data')
channels = ['WatcherGuru', 'rickler_feed', 'dlnewsinfo']

print('=== Извлечение необработанных сообщений ===\n')

processed_df = pd.read_csv(output_dir / 'btc_messages_tbsa.csv')
processed_texts = set(processed_df['text'].values)
print(f'Шаг 1: Загружено {len(processed_texts):,} уже обработанных текстов')

all_messages = []

for channel in channels:
    print(f'\nОбработка канала {channel}...')
    channel_dir = data_dir / channel
    batch_files = sorted(channel_dir.glob('batch_*.json'))

    channel_count = 0

    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            messages = json.load(f)

        for msg in messages:
            text = msg.get('text', '')
            if text and len(text) >= 5:
                all_messages.append({
                    'date': msg['date'],
                    'channel': msg['channel'],
                    'text': text,
                    'views': msg.get('views', 0),
                    'forwards': msg.get('forwards', 0),
                    'replies': msg.get('replies', 0)
                })
                channel_count += 1

    print(f'  Найдено сообщений с текстом: {channel_count:,}')

print(f'\nШаг 2: Создание DataFrame...')
df_all = pd.DataFrame(all_messages)
print(f'  Всего сообщений: {len(df_all):,}')

print(f'\nШаг 3: Удаление дубликатов...')
original_count = len(df_all)
df_all = df_all.drop_duplicates(subset=['text'], keep='first')
duplicates_removed = original_count - len(df_all)
print(f'  Удалено дубликатов: {duplicates_removed:,}')
print(f'  Осталось уникальных: {len(df_all):,}')

print(f'\nШаг 4: Фильтрация необработанных...')
df_all['is_processed'] = df_all['text'].isin(processed_texts)
df_unprocessed = df_all[~df_all['is_processed']].drop(columns=['is_processed'])

print(f'\n✨ Необработанных сообщений: {len(df_unprocessed):,}')
print(f'   Это {len(df_unprocessed)/len(df_all)*100:.1f}% от уникальных сообщений')

df_unprocessed['date'] = pd.to_datetime(df_unprocessed['date'])
df_unprocessed = df_unprocessed.sort_values('date')

print(f'\nПериод данных:')
print(f'  От: {df_unprocessed["date"].min()}')
print(f'  До: {df_unprocessed["date"].max()}')
print(f'  Дней: {(df_unprocessed["date"].max() - df_unprocessed["date"].min()).days}')

print(f'\nРаспределение по каналам:')
for channel, count in df_unprocessed['channel'].value_counts().items():
    print(f'  {channel}: {count:,} ({count/len(df_unprocessed)*100:.1f}%)')

print(f'\nСтатистика по длине текста:')
text_lengths = df_unprocessed['text'].str.len()
print(f'  Минимум: {text_lengths.min()} символов')
print(f'  Среднее: {text_lengths.mean():.0f} символов')
print(f'  Медиана: {text_lengths.median():.0f} символов')
print(f'  Максимум: {text_lengths.max()} символов')

output_file = output_dir / 'all_messages_unprocessed.csv'
df_unprocessed.to_csv(output_file, index=False)

file_size_mb = output_file.stat().st_size / (1024 * 1024)
print(f'\n✅ Сохранено: {output_file}')
print(f'   Размер файла: {file_size_mb:.2f} MB')
print(f'   Записей: {len(df_unprocessed):,}')

print(f'\n💰 Оценка стоимости обработки TBSA API:')
cost_per_1k = 0.15
total_cost = (len(df_unprocessed) / 1000) * cost_per_1k
print(f'   ~${total_cost:.2f} (при ${cost_per_1k}/1K сообщений)')
