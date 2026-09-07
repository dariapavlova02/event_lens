import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

data_dir = (Path(__file__).resolve().parents[1] / 'data')
results_dir = (Path(__file__).resolve().parents[1] / 'results')
results_dir.mkdir(parents=True, exist_ok=True)

input_file = data_dir / 'btc_messages_with_sentiment.csv'

print('=== Визуализация корреляции BTC цена vs Telegram Sentiment ===\n')

df = pd.read_csv(input_file)
df['date'] = pd.to_datetime(df['date'])

print(f'Загружено сообщений: {len(df):,}')
print(f'Период: {df["date"].min()} -> {df["date"].max()}')

print('\n📊 Агрегация sentiment по часам...')
df_hourly = df.set_index('date').resample('1H').agg({
    'sentiment_score': 'mean',
    'text': 'count'
}).rename(columns={'text': 'message_count'})

df_hourly = df_hourly[df_hourly['message_count'] > 0]

print(f'Часов с данными: {len(df_hourly)}')
print(f'Среднее сообщений/час: {df_hourly["message_count"].mean():.1f}')

start_date = df_hourly.index.min()
end_date = df_hourly.index.max()

print(f'\n💰 Скачивание BTC-USD цен ({start_date.date()} -> {end_date.date()})...')
btc_data = yf.download(
    'BTC-USD',
    start=start_date - timedelta(days=1),
    end=end_date + timedelta(days=1),
    interval='1h',
    progress=False
)

print(f'Загружено: {len(btc_data)} ценовых точек')

df_hourly['sentiment_ma24'] = df_hourly['sentiment_score'].rolling(window=24, min_periods=1).mean()

print('\n📈 Создание графика...')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))

ax1.plot(btc_data.index, btc_data['Close'], color='#FF9500', linewidth=2, label='BTC Price')
ax1.set_ylabel('Bitcoin Price (USD)', fontsize=12, fontweight='bold')
ax1.set_title('Bitcoin Price vs Telegram Sentiment Analysis', fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

color = '#FF3B30'
ax2.plot(df_hourly.index, df_hourly['sentiment_score'], color=color, alpha=0.2, label='Raw Sentiment')
ax2.plot(df_hourly.index, df_hourly['sentiment_ma24'], color=color, linewidth=2, label='Sentiment (24h MA)')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_ylabel('Sentiment Score', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')

ax2_twin = ax2.twinx()
ax2_twin.bar(df_hourly.index, df_hourly['message_count'], alpha=0.2, color='blue', width=0.04)
ax2_twin.set_ylabel('Messages/Hour', fontsize=10, color='blue')
ax2_twin.tick_params(axis='y', labelcolor='blue')

price_aligned = btc_data['Close'].reindex(df_hourly.index, method='nearest')
sentiment_aligned = df_hourly['sentiment_ma24']

valid_idx = price_aligned.notna() & sentiment_aligned.notna()
if valid_idx.sum() > 0:
    correlation = price_aligned[valid_idx].corr(sentiment_aligned[valid_idx])
    print(f'📊 Корреляция (цена vs sentiment 24h MA): {correlation:.4f}')

    ax3.scatter(sentiment_aligned[valid_idx], price_aligned[valid_idx], alpha=0.5, s=20)
    ax3.set_xlabel('Sentiment Score (24h MA)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Bitcoin Price (USD)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Correlation: {correlation:.4f}', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    z = np.polyfit(sentiment_aligned[valid_idx], price_aligned[valid_idx], 1)
    p = np.poly1d(z)
    ax3.plot(sentiment_aligned[valid_idx], p(sentiment_aligned[valid_idx]), "r--", alpha=0.8, linewidth=2)

ax3.set_xlabel('Date', fontsize=12)

plt.tight_layout()

output_plot = results_dir / 'btc_sentiment_correlation.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f'\n✅ График сохранен: {output_plot}')

stats_output = results_dir / 'correlation_stats.txt'
with open(stats_output, 'w') as f:
    f.write(f'=== BTC Sentiment Analysis Results ===\n\n')
    f.write(f'Data Period: {start_date} -> {end_date}\n')
    f.write(f'Total Messages: {len(df):,}\n')
    f.write(f'Hours with data: {len(df_hourly)}\n')
    f.write(f'Average messages/hour: {df_hourly["message_count"].mean():.1f}\n\n')
    f.write(f'Sentiment Statistics:\n')
    f.write(f'  Mean: {df["sentiment_score"].mean():.4f}\n')
    f.write(f'  Median: {df["sentiment_score"].median():.4f}\n')
    f.write(f'  Std: {df["sentiment_score"].std():.4f}\n\n')
    f.write(f'Price-Sentiment Correlation: {correlation:.4f}\n')
    f.write(f'\nInterpretation:\n')
    if correlation > 0.3:
        f.write(f'  ✅ Положительная корреляция - sentiment предсказывает движение цены\n')
    elif correlation < -0.3:
        f.write(f'  ⚠️  Отрицательная корреляция - противоположное движение\n')
    else:
        f.write(f'  ℹ️  Слабая корреляция - нужна дополнительная фильтрация данных\n')

print(f'✅ Статистика сохранена: {stats_output}')

print('\n=== АНАЛИЗ ЗАВЕРШЕН ===')
print(f'Результаты в: {results_dir}')
print(f'  - График: btc_sentiment_correlation.png')
print(f'  - Статистика: correlation_stats.txt')
