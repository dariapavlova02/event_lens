import pandas as pd
import json
from pathlib import Path
from datetime import datetime

data_dir = (Path(__file__).resolve().parents[1] / 'data')

input_file = data_dir / 'btc_messages_tbsa.csv'
output_file = data_dir / 'btc_hourly_sentiment.csv'

print('=== TBSA Time Series Preparation ===\n')

df = pd.read_csv(input_file)
df['date'] = pd.to_datetime(df['date'])

print(f'Загружено сообщений: {len(df):,}')
print(f'Период: {df["date"].min()} -> {df["date"].max()}')

df_success = df[df['tbsa_has_result'] == True].copy()
print(f'Успешно обработано: {len(df_success):,} ({len(df_success)/len(df)*100:.2f}%)')

print('\n📊 Парсинг BTC targets из JSON...')

btc_signals = []
parse_errors = 0
no_btc_count = 0
multi_btc_count = 0

for idx, row in df_success.iterrows():
    try:
        targets = json.loads(row['tbsa_targets_json'])

        btc_targets = [
            t for t in targets
            if t.get('ticker') == 'BTC' or 'Bitcoin' in t.get('name', '')
        ]

        if not btc_targets:
            no_btc_count += 1
            continue

        if len(btc_targets) > 1:
            multi_btc_count += 1
            target = max(btc_targets, key=lambda x: x.get('impact', 0))
        else:
            target = btc_targets[0]

        btc_signals.append({
            'timestamp': row['date'],
            'sentiment': target.get('sentiment', 0),
            'impact': target.get('impact', 1),
            'is_fact': row['tbsa_is_fact'],
            'global_sentiment': row['tbsa_global_sentiment'],
            'channel': row['channel'],
            'views': row['views'],
            'forwards': row['forwards']
        })

    except Exception as e:
        parse_errors += 1

print(f'✅ Извлечено BTC signals: {len(btc_signals):,}')
print(f'   Пропущено (нет BTC): {no_btc_count:,}')
print(f'   Множественные BTC targets: {multi_btc_count:,}')
print(f'   Ошибки парсинга: {parse_errors}')

btc_df = pd.DataFrame(btc_signals)
btc_df = btc_df.set_index('timestamp').sort_index()

print(f'\n⏱️ Агрегация по часам (1h resample)...')

def weighted_sentiment(df_group):
    if df_group['impact'].sum() == 0:
        return 0.0
    return (df_group['sentiment'] * df_group['impact']).sum() / df_group['impact'].sum()

def fact_ratio(df_group):
    if len(df_group) == 0:
        return 0.0
    return df_group['is_fact'].sum() / len(df_group)

hourly_agg = btc_df.resample('1h').apply(lambda x: pd.Series({
    'sentiment_raw_avg': x['sentiment'].mean() if len(x) > 0 else 0,
    'weighted_sentiment': weighted_sentiment(x),
    'social_volume': x['impact'].sum(),
    'fact_ratio': fact_ratio(x),
    'global_sentiment_avg': x['global_sentiment'].mean() if len(x) > 0 else 0,
    'total_views': x['views'].sum(),
    'total_forwards': x['forwards'].sum(),
    'message_count': len(x)
}))

hourly_agg = hourly_agg[hourly_agg['message_count'] > 0].copy()

hourly_agg = hourly_agg[hourly_agg['message_count'] > 0].copy()

print(f'✅ Создано часовых записей: {len(hourly_agg):,}')
print(f'   Период: {hourly_agg.index.min()} -> {hourly_agg.index.max()}')
print(f'   Дней с данными: {(hourly_agg.index.max() - hourly_agg.index.min()).days}')

print(f'\n📈 Статистика weighted sentiment:')
print(f'   Mean: {hourly_agg["weighted_sentiment"].mean():.4f}')
print(f'   Median: {hourly_agg["weighted_sentiment"].median():.4f}')
print(f'   Std: {hourly_agg["weighted_sentiment"].std():.4f}')
print(f'   Min: {hourly_agg["weighted_sentiment"].min():.4f}')
print(f'   Max: {hourly_agg["weighted_sentiment"].max():.4f}')

print(f'\n📊 Статистика social volume (impact sum):')
print(f'   Mean: {hourly_agg["social_volume"].mean():.1f}')
print(f'   Median: {hourly_agg["social_volume"].median():.1f}')
print(f'   Max: {hourly_agg["social_volume"].max():.0f}')

print(f'\n💬 Статистика message count:')
print(f'   Mean msgs/hour: {hourly_agg["message_count"].mean():.1f}')
print(f'   Max msgs/hour: {hourly_agg["message_count"].max():.0f}')
print(f'   Hours with data: {len(hourly_agg)} / {(hourly_agg.index.max() - hourly_agg.index.min()).total_seconds() / 3600:.0f} ({len(hourly_agg) / ((hourly_agg.index.max() - hourly_agg.index.min()).total_seconds() / 3600) * 100:.1f}%)')

print(f'\n📰 Fact vs Opinion ratio:')
print(f'   Average fact ratio: {hourly_agg["fact_ratio"].mean():.2%}')

hourly_agg.to_csv(output_file)
print(f'\n✅ Сохранено: {output_file}')
print(f'   Размер: {output_file.stat().st_size / 1024 / 1024:.2f} MB')

print('\n=== Скрипт 11 ЗАВЕРШЕН ===')
print(f'Следующий шаг: python3 scripts/12_market_correlation.py')
