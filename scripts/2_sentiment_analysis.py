import pandas as pd
from pathlib import Path
from transformers import pipeline
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

data_dir = (Path(__file__).resolve().parents[1] / 'data')
input_file = data_dir / 'btc_messages.csv'

print('=== FinBERT Sentiment Analysis ===\n')

df = pd.read_csv(input_file)
df['date'] = pd.to_datetime(df['date'])

print(f'Загружено сообщений: {len(df):,}')
print(f'Период: {df["date"].min()} -> {df["date"].max()}')

print('\n⏳ Загрузка модели FinBERT (первый запуск ~400 MB)...')
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    device=-1
)
print('✅ Модель загружена')

def get_sentiment(text):
    try:
        text_truncated = text[:512]
        result = sentiment_pipe(text_truncated)[0]

        score = result['score']

        if result['label'] == 'negative':
            return -score
        elif result['label'] == 'neutral':
            return 0.0
        else:
            return score
    except Exception as e:
        return 0.0

print(f'\n🔍 Анализ тональности {len(df):,} сообщений...')
print('(Это займет ~10-15 минут для ~18k сообщений)\n')

tqdm.pandas(desc="Sentiment Analysis")
df['sentiment_score'] = df['text'].progress_apply(get_sentiment)

print('\n=== Результаты ===')
print(f'Средний sentiment: {df["sentiment_score"].mean():.3f}')
print(f'Median sentiment: {df["sentiment_score"].median():.3f}')
print(f'Std sentiment: {df["sentiment_score"].std():.3f}')

print(f'\nРаспределение:')
print(f'  Позитивные (>0.3): {(df["sentiment_score"] > 0.3).sum():,} ({(df["sentiment_score"] > 0.3).sum()/len(df)*100:.1f}%)')
print(f'  Нейтральные (-0.3 to 0.3): {((df["sentiment_score"] >= -0.3) & (df["sentiment_score"] <= 0.3)).sum():,}')
print(f'  Негативные (<-0.3): {(df["sentiment_score"] < -0.3).sum():,} ({(df["sentiment_score"] < -0.3).sum()/len(df)*100:.1f}%)')

output_file = data_dir / 'btc_messages_with_sentiment.csv'
df.to_csv(output_file, index=False)
print(f'\n✅ Сохранено: {output_file}')

print(f'\nПримеры сообщений:')
print('\n--- ПОЗИТИВНЫЕ ---')
positive = df.nlargest(3, 'sentiment_score')
for _, row in positive.iterrows():
    print(f"Score: {row['sentiment_score']:.3f}")
    print(f"Text: {row['text'][:150]}...")
    print()

print('--- НЕГАТИВНЫЕ ---')
negative = df.nsmallest(3, 'sentiment_score')
for _, row in negative.iterrows():
    print(f"Score: {row['sentiment_score']:.3f}")
    print(f"Text: {row['text'][:150]}...")
    print()
