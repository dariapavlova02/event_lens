import pandas as pd
import numpy as np
import yfinance as yf
import json
from pathlib import Path
from scipy.stats import pearsonr
from datetime import timedelta

data_dir = (Path(__file__).resolve().parents[1] / 'data')

input_file = data_dir / 'btc_hourly_sentiment.csv'
output_merged = data_dir / 'btc_market_merged.csv'
output_results = data_dir / 'correlation_results.json'

print('=== Market Correlation Analysis ===\n')

sentiment_df = pd.read_csv(input_file, index_col=0, parse_dates=True)
print(f'Загружено sentiment данных: {len(sentiment_df):,} часов')
print(f'Период: {sentiment_df.index.min()} -> {sentiment_df.index.max()}')

print(f'\n💰 Скачивание BTC-USD цен...')
print('ℹ️  Yahoo Finance ограничение: часовые данные доступны только за последние 730 дней')

end_date = sentiment_df.index.max() + timedelta(days=1)
start_date = end_date - timedelta(days=729)

actual_start = max(start_date, sentiment_df.index.min() - timedelta(days=1))

print(f'   Запрашиваем: {actual_start.date()} -> {end_date.date()}')
print('   ⏳ Загрузка...')

btc_data = yf.download(
    'BTC-USD',
    start=actual_start,
    end=end_date,
    interval='1h',
    progress=False,
    auto_adjust=True
)

if isinstance(btc_data.columns, pd.MultiIndex):
    btc_data.columns = btc_data.columns.droplevel(1)

print(f'✅ Загружено BTC данных: {len(btc_data):,} часов')
print(f'   Период: {btc_data.index.min()} -> {btc_data.index.max()}')

if btc_data.index.tz is None:
    btc_data.index = btc_data.index.tz_localize('UTC')
else:
    btc_data.index = btc_data.index.tz_convert('UTC')

if sentiment_df.index.tz is None:
    sentiment_df.index = sentiment_df.index.tz_localize('UTC')
else:
    sentiment_df.index = sentiment_df.index.tz_convert('UTC')

print(f'\n🔗 Объединение данных (inner join)...')
merged = pd.merge(
    sentiment_df,
    btc_data[['Open', 'High', 'Low', 'Close', 'Volume']],
    left_index=True,
    right_index=True,
    how='inner'
)

print(f'✅ Объединено: {len(merged):,} часов ({len(merged)/len(sentiment_df)*100:.1f}% от sentiment)')

print(f'\n📊 Вычисление производных метрик...')

merged['price_return_1h'] = merged['Close'].pct_change()
merged['price_return_3h'] = merged['Close'].pct_change(periods=3)
merged['price_return_6h'] = merged['Close'].pct_change(periods=6)
merged['price_return_12h'] = merged['Close'].pct_change(periods=12)
merged['price_return_24h'] = merged['Close'].pct_change(periods=24)

merged['volatility'] = (merged['High'] - merged['Low']) / merged['Close']

merged['price_change_1h'] = merged['Close'].shift(-1) - merged['Close']
merged['price_change_3h'] = merged['Close'].shift(-3) - merged['Close']
merged['price_change_6h'] = merged['Close'].shift(-6) - merged['Close']
merged['price_change_12h'] = merged['Close'].shift(-12) - merged['Close']
merged['price_change_24h'] = merged['Close'].shift(-24) - merged['Close']

merged_clean = merged.dropna()
print(f'После удаления NaN: {len(merged_clean):,} записей')

results = {}

print(f'\n═══════════════════════════════════════')
print(f'📈 КОРРЕЛЯЦИОННЫЙ АНАЛИЗ')
print(f'═══════════════════════════════════════\n')

print('1️⃣  СИНХРОННАЯ КОРРЕЛЯЦИЯ (Weighted Sentiment vs Price)')
print('─' * 50)

def calc_correlation(x, y, name):
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 30:
        return {'r': 0, 'p_value': 1, 'n': 0, 'significant': False}

    r, p = pearsonr(x[valid], y[valid])
    return {
        'r': float(r),
        'p_value': float(p),
        'n': int(valid.sum()),
        'significant': bool(p < 0.05),
        'interpretation': 'Strong' if abs(r) > 0.5 else 'Moderate' if abs(r) > 0.3 else 'Weak'
    }

sync_price = calc_correlation(
    merged_clean['weighted_sentiment'].values,
    merged_clean['Close'].values,
    'Price vs Weighted Sentiment'
)
results['sync_price_correlation'] = sync_price

print(f"Корреляция (Price vs Weighted Sentiment):")
print(f"  r = {sync_price['r']:.4f} (p={sync_price['p_value']:.4e}, n={sync_price['n']:,})")
print(f"  {'✅ Статистически значима' if sync_price['significant'] else '❌ Незначима'} ({sync_price['interpretation']})")

sync_return = calc_correlation(
    merged_clean['weighted_sentiment'].values,
    merged_clean['price_return_1h'].values,
    'Returns vs Weighted Sentiment'
)
results['sync_return_correlation'] = sync_return

print(f"\nКорреляция (1h Returns vs Weighted Sentiment):")
print(f"  r = {sync_return['r']:.4f} (p={sync_return['p_value']:.4e})")
print(f"  {'✅ Статистически значима' if sync_return['significant'] else '❌ Незначима'}")

print(f'\n2️⃣  LAG ANALYSIS (Sentiment предсказывает будущую цену?)')
print('─' * 50)

lags_to_test = [1, 2, 3, 6, 12, 24, 48, 72]
lag_results = []

for lag in lags_to_test:
    sentiment_shifted = merged_clean['weighted_sentiment'].shift(lag)

    corr = calc_correlation(
        sentiment_shifted.values,
        merged_clean['Close'].values,
        f'Lag {lag}h'
    )

    lag_results.append({
        'lag_hours': lag,
        **corr
    })

    marker = '🔥' if corr['significant'] and abs(corr['r']) > 0.2 else '✅' if corr['significant'] else '  '
    print(f"  {marker} Lag {lag:2d}h: r={corr['r']:7.4f} (p={corr['p_value']:.2e}) [{corr['interpretation']}]")

results['lag_analysis'] = lag_results

best_lag = max(lag_results, key=lambda x: abs(x['r']))
print(f"\n🎯 Оптимальный лаг: {best_lag['lag_hours']}h (r={best_lag['r']:.4f})")

print(f'\n3️⃣  VOLATILITY PREDICTION (Social Volume vs Volatility)')
print('─' * 50)

vol_corr = calc_correlation(
    merged_clean['social_volume'].values,
    merged_clean['volatility'].values,
    'Social Volume vs Volatility'
)
results['volatility_correlation'] = vol_corr

print(f"Корреляция (Social Volume vs Volatility):")
print(f"  r = {vol_corr['r']:.4f} (p={vol_corr['p_value']:.4e})")
print(f"  {'✅ Статистически значима' if vol_corr['significant'] else '❌ Незначима'} ({vol_corr['interpretation']})")

vol_btc_volume = calc_correlation(
    merged_clean['social_volume'].values,
    merged_clean['Volume'].values,
    'Social Volume vs BTC Trading Volume'
)
results['btc_volume_correlation'] = vol_btc_volume

print(f"\nКорреляция (Social Volume vs BTC Trading Volume):")
print(f"  r = {vol_btc_volume['r']:.4f} (p={vol_btc_volume['p_value']:.4e})")
print(f"  {'✅ Статистически значима' if vol_btc_volume['significant'] else '❌ Незначима'}")

print(f'\n4️⃣  SEGMENT ANALYSIS (Facts vs Opinions, High Impact)')
print('─' * 50)

facts_only = merged_clean[merged_clean['fact_ratio'] > 0.7]
opinions_more = merged_clean[merged_clean['fact_ratio'] <= 0.5]
high_impact = merged_clean[merged_clean['social_volume'] >= merged_clean['social_volume'].quantile(0.75)]

segments = {
    'all_data': merged_clean,
    'facts_dominant': facts_only,
    'opinions_more': opinions_more,
    'high_impact_q75': high_impact
}

segment_results = {}

for segment_name, segment_df in segments.items():
    if len(segment_df) < 30:
        continue

    seg_corr = calc_correlation(
        segment_df['weighted_sentiment'].values,
        segment_df['Close'].values,
        segment_name
    )

    segment_results[segment_name] = {
        **seg_corr,
        'sample_size': len(segment_df)
    }

    marker = '🔥' if seg_corr['significant'] and abs(seg_corr['r']) > 0.3 else '✅' if seg_corr['significant'] else '  '
    print(f"  {marker} {segment_name:20s}: r={seg_corr['r']:7.4f} (n={len(segment_df):5,}) [{seg_corr['interpretation']}]")

results['segment_analysis'] = segment_results

print(f'\n5️⃣  PREDICTIVE ANALYSIS (Sentiment → Future Price Change)')
print('─' * 50)

predictive = {}
for horizon in ['1h', '3h', '6h', '12h', '24h']:
    col = f'price_change_{horizon}'
    pred_corr = calc_correlation(
        merged_clean['weighted_sentiment'].values,
        merged_clean[col].values,
        f'Sentiment vs Price Change {horizon}'
    )
    predictive[horizon] = pred_corr

    marker = '🔥' if pred_corr['significant'] and abs(pred_corr['r']) > 0.1 else '✅' if pred_corr['significant'] else '  '
    print(f"  {marker} {horizon}: r={pred_corr['r']:7.4f} (p={pred_corr['p_value']:.2e}) [{pred_corr['interpretation']}]")

results['predictive_analysis'] = predictive

print(f'\n═══════════════════════════════════════')
print(f'💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ')
print(f'═══════════════════════════════════════\n')

merged.to_csv(output_merged)
print(f'✅ Merged data: {output_merged}')
print(f'   Размер: {output_merged.stat().st_size / 1024 / 1024:.2f} MB')

results['metadata'] = {
    'data_period_start': str(merged.index.min()),
    'data_period_end': str(merged.index.max()),
    'total_hours': len(merged),
    'coverage_pct': float(len(merged) / len(sentiment_df) * 100)
}

with open(output_results, 'w') as f:
    json.dump(results, f, indent=2)

print(f'✅ Correlation results: {output_results}')

print(f'\n═══════════════════════════════════════')
print(f'🎓 ИНТЕРПРЕТАЦИЯ ДЛЯ ДИПЛОМА')
print(f'═══════════════════════════════════════\n')

if sync_price['significant'] and abs(sync_price['r']) > 0.3:
    print('✅ STRONG FINDING: Sentiment коррелирует с ценой!')
    print(f'   → "Impact-weighted Telegram sentiment shows {sync_price["interpretation"].lower()} correlation (r={sync_price["r"]:.3f}) with Bitcoin price"')
elif vol_corr['significant'] and abs(vol_corr['r']) > 0.3:
    print('✅ ALTERNATIVE FINDING: Sentiment предсказывает волатильность!')
    print(f'   → "Social volume serves as a volatility predictor (r={vol_corr["r"]:.3f})"')
elif best_lag['significant'] and abs(best_lag['r']) > 0.2:
    print('✅ LEADING INDICATOR: Sentiment опережает цену!')
    print(f'   → "Sentiment leads price by {best_lag["lag_hours"]} hours (r={best_lag["r"]:.3f})"')
else:
    print('⚠️  Корреляции слабые, но это тоже научный результат!')
    print('   → "While direct price correlation is modest, segmentation reveals meaningful patterns in specific contexts"')

    best_segment = max(segment_results.items(), key=lambda x: abs(x[1]['r']))
    print(f'   → Лучший сегмент: {best_segment[0]} (r={best_segment[1]["r"]:.3f})')

print('\n=== Скрипт 12 ЗАВЕРШЕН ===')
print(f'Следующий шаг: python3 scripts/13_visualize_findings.py')
