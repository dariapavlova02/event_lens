import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

data_dir = (Path(__file__).resolve().parents[1] / 'data')

input_file = data_dir / 'btc_messages_tbsa.csv'
output_file = data_dir / 'btc_enhanced_features.csv'

print('=' * 80)
print('STUDY 1: COMPREHENSIVE FEATURE ENGINEERING')
print('=' * 80)
print()

df = pd.read_csv(input_file)
df['date'] = pd.to_datetime(df['date'])

print(f'📊 Loaded messages: {len(df):,}')
print(f'   Period: {df["date"].min()} → {df["date"].max()}')

df_success = df[df['tbsa_has_result'] == True].copy()
print(f'   Successfully processed: {len(df_success):,} ({len(df_success)/len(df)*100:.2f}%)')

print()
print('🔍 Extracting BTC targets with full TBSA metadata...')

btc_signals = []
parse_errors = 0
no_btc_count = 0

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

        target = max(btc_targets, key=lambda x: x.get('impact', 0))

        btc_signals.append({
            'timestamp': row['date'],
            'sentiment': target.get('sentiment', 0),
            'impact': target.get('impact', 1),
            'category': target.get('category', 'UNKNOWN'),
            'is_fact': row['tbsa_is_fact'],
            'global_sentiment': row['tbsa_global_sentiment'],
            'channel': row['channel'],
            'views': row['views'],
            'forwards': row['forwards']
        })

    except Exception as e:
        parse_errors += 1

print(f'✅ Extracted BTC signals: {len(btc_signals):,}')
print(f'   Skipped (no BTC): {no_btc_count:,}')
print(f'   Parse errors: {parse_errors}')

btc_df = pd.DataFrame(btc_signals)
btc_df = btc_df.set_index('timestamp').sort_index()

print()
print(f'📋 Category distribution:')
category_counts = btc_df['category'].value_counts()
for cat, count in category_counts.items():
    print(f'   {cat:20s}: {count:6,} ({count/len(btc_df)*100:5.2f}%)')

print()
print(f'⏱️  Aggregating by hour (1h resample)...')

def weighted_sentiment(df_group):
    if df_group['impact'].sum() == 0:
        return 0.0
    return (df_group['sentiment'] * df_group['impact']).sum() / df_group['impact'].sum()

def attention_score(df_group):
    return (df_group['views'] * df_group['forwards']).sum()

def category_weighted_sentiment(df_group, category):
    cat_msgs = df_group[df_group['category'] == category]
    if len(cat_msgs) == 0 or cat_msgs['impact'].sum() == 0:
        return 0.0
    return (cat_msgs['sentiment'] * cat_msgs['impact']).sum() / cat_msgs['impact'].sum()

def category_volume(df_group, category):
    return df_group[df_group['category'] == category]['impact'].sum()

def high_impact_sentiment(df_group, threshold):
    high_impact = df_group[df_group['impact'] >= threshold]
    if len(high_impact) == 0 or high_impact['impact'].sum() == 0:
        return 0.0
    return (high_impact['sentiment'] * high_impact['impact']).sum() / high_impact['impact'].sum()

def fact_sentiment(df_group):
    facts = df_group[df_group['is_fact'] == True]
    if len(facts) == 0 or facts['impact'].sum() == 0:
        return 0.0
    return (facts['sentiment'] * facts['impact']).sum() / facts['impact'].sum()

def opinion_sentiment(df_group):
    opinions = df_group[df_group['is_fact'] == False]
    if len(opinions) == 0 or opinions['impact'].sum() == 0:
        return 0.0
    return (opinions['sentiment'] * opinions['impact']).sum() / opinions['impact'].sum()

categories = ['REGULATION', 'MARKET', 'HACK', 'OPINION', 'COMMUNITY', 'TECHNOLOGY', 'UNKNOWN']

hourly_features = btc_df.resample('1h').apply(lambda x: pd.Series({
    'sentiment_baseline': weighted_sentiment(x),
    'social_volume': x['impact'].sum(),
    'message_count': len(x),

    'sentiment_regulation': category_weighted_sentiment(x, 'REGULATION'),
    'sentiment_market': category_weighted_sentiment(x, 'MARKET'),
    'sentiment_hack': category_weighted_sentiment(x, 'HACK'),
    'sentiment_opinion': category_weighted_sentiment(x, 'OPINION'),
    'sentiment_community': category_weighted_sentiment(x, 'COMMUNITY'),
    'sentiment_technology': category_weighted_sentiment(x, 'TECHNOLOGY'),

    'volume_regulation': category_volume(x, 'REGULATION'),
    'volume_market': category_volume(x, 'MARKET'),
    'volume_hack': category_volume(x, 'HACK'),
    'volume_opinion': category_volume(x, 'OPINION'),
    'volume_community': category_volume(x, 'COMMUNITY'),
    'volume_technology': category_volume(x, 'TECHNOLOGY'),

    'sentiment_high_impact': high_impact_sentiment(x, 8),
    'sentiment_medium_impact': high_impact_sentiment(x, 5),

    'volume_high_impact': x[x['impact'] >= 8]['impact'].sum(),
    'volume_medium_impact': x[(x['impact'] >= 5) & (x['impact'] < 8)]['impact'].sum(),
    'volume_low_impact': x[x['impact'] < 5]['impact'].sum(),

    'count_high_impact': len(x[x['impact'] >= 8]),

    'sentiment_fact': fact_sentiment(x),
    'sentiment_opinion_type': opinion_sentiment(x),

    'volume_fact': x[x['is_fact'] == True]['impact'].sum(),
    'volume_opinion_type': x[x['is_fact'] == False]['impact'].sum(),

    'fact_ratio': x['is_fact'].sum() / len(x) if len(x) > 0 else 0,

    'attention_weighted_sentiment': (
        (x['sentiment'] * x['impact'] * x['views'] * (1 + x['forwards'])).sum() /
        (x['impact'] * x['views'] * (1 + x['forwards'])).sum()
        if (x['impact'] * x['views'] * (1 + x['forwards'])).sum() > 0 else 0
    ),
    'total_attention': attention_score(x),
    'total_views': x['views'].sum(),
    'total_forwards': x['forwards'].sum(),

    'sentiment_positive': (x['sentiment'] > 0).sum(),
    'sentiment_negative': (x['sentiment'] < 0).sum(),
    'sentiment_neutral': (x['sentiment'] == 0).sum(),

    'sentiment_dispersion': x['sentiment'].std() if len(x) > 1 else 0,
    'impact_mean': x['impact'].mean() if len(x) > 0 else 0,
    'impact_max': x['impact'].max() if len(x) > 0 else 0,
}))

hourly_features = hourly_features[hourly_features['message_count'] > 0].copy()

print(f'✅ Created hourly records: {len(hourly_features):,}')
print(f'   Period: {hourly_features.index.min()} → {hourly_features.index.max()}')

print()
print('🔧 Computing temporal features (momentum, changes, acceleration)...')

for col in ['sentiment_baseline', 'social_volume', 'sentiment_regulation',
            'sentiment_market', 'sentiment_high_impact', 'volume_high_impact']:
    hourly_features[f'{col}_delta_1h'] = hourly_features[col].diff(1)
    hourly_features[f'{col}_delta_3h'] = hourly_features[col].diff(3)
    hourly_features[f'{col}_ma_6h'] = hourly_features[col].rolling(6, min_periods=1).mean()
    hourly_features[f'{col}_ma_24h'] = hourly_features[col].rolling(24, min_periods=1).mean()
    hourly_features[f'{col}_momentum'] = hourly_features[col] - hourly_features[f'{col}_ma_24h']

hourly_features['acceleration_sentiment'] = hourly_features['sentiment_baseline_delta_1h'].diff(1)
hourly_features['acceleration_volume'] = hourly_features['social_volume_delta_1h'].diff(1)

print(f'✅ Temporal features computed')

print()
print('📊 Feature Summary:')
print(f'   Total features: {len(hourly_features.columns)}')
print()
print('   Feature groups:')
print(f'     - Baseline (2): sentiment, volume')
print(f'     - Category-specific (12): sentiment × 6 + volume × 6')
print(f'     - Impact-stratified (9): sentiment × 2 + volume × 3 + count × 1')
print(f'     - Fact/Opinion (5): sentiment × 2 + volume × 2 + ratio × 1')
print(f'     - Attention-weighted (4): sentiment + attention + views + forwards')
print(f'     - Polarity (3): positive + negative + neutral counts')
print(f'     - Quality (3): dispersion + impact_mean + impact_max')
print(f'     - Temporal (25): delta/ma/momentum for 5 features')

print()
print('🔍 Sample statistics for key features:')

key_features = [
    'sentiment_baseline',
    'social_volume',
    'sentiment_regulation',
    'sentiment_market',
    'sentiment_high_impact',
    'volume_high_impact',
    'sentiment_fact',
    'attention_weighted_sentiment'
]

for feat in key_features:
    if feat in hourly_features.columns:
        values = hourly_features[feat].dropna()
        print(f'\n   {feat}:')
        print(f'     Mean: {values.mean():8.4f}  Median: {values.median():8.4f}')
        print(f'     Std:  {values.std():8.4f}  Range: [{values.min():7.4f}, {values.max():7.4f}]')
        print(f'     Non-zero: {(values != 0).sum():,} ({(values != 0).sum()/len(values)*100:.1f}%)')

hourly_features.to_csv(output_file)
print()
print(f'✅ Saved: {output_file}')
print(f'   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB')
print(f'   Shape: {hourly_features.shape}')

print()
print('=' * 80)
print('STUDY 1 COMPLETE')
print('=' * 80)
print(f'Next: python3 scripts/20_univariate_analysis.py')
