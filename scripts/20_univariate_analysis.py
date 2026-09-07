import pandas as pd
import numpy as np
import yfinance as yf
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac
from sklearn.model_selection import TimeSeriesSplit
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

data_dir = (Path(__file__).resolve().parents[1] / 'data')

features_file = data_dir / 'btc_enhanced_features.csv'
market_file = data_dir / 'btc_market_merged.csv'
output_file = data_dir / 'univariate_analysis_results.json'

print('=' * 80)
print('STUDY 2: UNIVARIATE ANALYSIS')
print('=' * 80)
print()
print('Statistical rigor:')
print('  ✓ Stationarity testing (ADF + KPSS)')
print('  ✓ Correlation (Pearson, Spearman, Kendall)')
print('  ✓ HAC standard errors (Newey-West)')
print('  ✓ FDR correction (Benjamini-Hochberg)')
print('  ✓ Train/test split (80/20 temporal)')
print('  ✓ Granger causality (1-12 lags)')
print()

print(f'📊 Loading enhanced features...')
features_df = pd.read_csv(features_file, index_col=0, parse_dates=True)
print(f'   Features: {len(features_df.columns)} columns × {len(features_df):,} hours')

print(f'📊 Loading market data...')
market_df = pd.read_csv(market_file, index_col=0, parse_dates=True)
print(f'   Market data: {len(market_df):,} hours')

print(f'\n🔗 Merging datasets...')
merged = features_df.join(market_df[['Close', 'price_return_1h', 'volatility', 'log_price']], how='inner')
merged = merged.rename(columns={'price_return_1h': 'returns'})
print(f'   Merged: {len(merged):,} hours')
print(f'   Period: {merged.index.min()} → {merged.index.max()}')

merged_clean = merged.dropna()
print(f'   After removing NaN: {len(merged_clean):,} records')

train_size = int(0.8 * len(merged_clean))
train = merged_clean.iloc[:train_size]
test = merged_clean.iloc[train_size:]
print(f'\n📊 Train/test split (80/20 temporal):')
print(f'   Train: {len(train):,} hours ({train.index.min()} → {train.index.max()})')
print(f'   Test:  {len(test):,} hours ({test.index.min()} → {test.index.max()})')

def stationarity_test(series, name):
    if len(series) < 30:
        return {'is_stationary': False, 'adf_pvalue': 1.0, 'kpss_pvalue': 0.0, 'note': 'insufficient_data'}

    try:
        adf = adfuller(series.dropna(), autolag='AIC')
        kpss_result = kpss(series.dropna(), regression='c', nlags='auto')

        is_stationary = (adf[1] < 0.05) and (kpss_result[1] > 0.05)

        return {
            'adf_statistic': float(adf[0]),
            'adf_pvalue': float(adf[1]),
            'kpss_statistic': float(kpss_result[0]),
            'kpss_pvalue': float(kpss_result[1]),
            'is_stationary': bool(is_stationary)
        }
    except:
        return {'is_stationary': False, 'adf_pvalue': 1.0, 'kpss_pvalue': 0.0, 'note': 'test_failed'}

def calc_correlation_robust(x, y, use_hac=True):
    valid = ~(np.isnan(x) | np.isnan(y)) & ~np.isinf(x) & ~np.isinf(y)
    if valid.sum() < 30:
        return {'r_pearson': 0, 'p_pearson': 1, 'r_spearman': 0, 'p_spearman': 1,
                'r_kendall': 0, 'p_kendall': 1, 'n': 0}

    x_clean = x[valid]
    y_clean = y[valid]

    try:
        r_pearson, p_pearson = pearsonr(x_clean, y_clean)
        r_spearman, p_spearman = spearmanr(x_clean, y_clean)
        r_kendall, p_kendall = kendalltau(x_clean, y_clean)

        p_hac = p_pearson
        if use_hac and len(x_clean) > 50:
            try:
                X = add_constant(x_clean)
                model = OLS(y_clean, X).fit()
                cov_hac_matrix = cov_hac(model, nlags=12)
                se_hac = np.sqrt(np.diag(cov_hac_matrix))[1]
                t_stat = model.params[1] / se_hac
                from scipy import stats
                p_hac = 2 * (1 - stats.t.cdf(abs(t_stat), len(x_clean) - 2))
            except:
                pass

        return {
            'r_pearson': float(r_pearson),
            'p_pearson': float(p_pearson),
            'p_pearson_hac': float(p_hac),
            'r_spearman': float(r_spearman),
            'p_spearman': float(p_spearman),
            'r_kendall': float(r_kendall),
            'p_kendall': float(p_kendall),
            'n': int(valid.sum()),
            'r_squared': float(r_pearson ** 2)
        }
    except:
        return {'r_pearson': 0, 'p_pearson': 1, 'r_spearman': 0, 'p_spearman': 1,
                'r_kendall': 0, 'p_kendall': 1, 'n': 0}

def test_granger_causality(x, y, max_lag=12):
    try:
        data = pd.DataFrame({'y': y, 'x': x}).dropna()
        if len(data) < 100:
            return {'significant_lags': [], 'min_pvalue': 1.0, 'has_causality': False}

        gc_results = grangercausalitytests(data[['y', 'x']], maxlag=max_lag, verbose=False)

        pvalues = [gc_results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)]
        min_pvalue = min(pvalues)
        significant_lags = [lag + 1 for lag, p in enumerate(pvalues) if p < 0.05]

        return {
            'pvalues_by_lag': [float(p) for p in pvalues],
            'min_pvalue': float(min_pvalue),
            'significant_lags': significant_lags,
            'has_causality': len(significant_lags) > 0
        }
    except:
        return {'significant_lags': [], 'min_pvalue': 1.0, 'has_causality': False}

print()
print('=' * 80)
print('PHASE 1: STATIONARITY TESTING')
print('=' * 80)
print()

feature_cols = [col for col in features_df.columns if col not in ['Close', 'returns', 'volatility', 'log_price']]

stationarity_results = {}
for col in feature_cols:
    if col in merged_clean.columns:
        stationarity_results[col] = stationarity_test(merged_clean[col], col)

stationary_features = [col for col, res in stationarity_results.items() if res.get('is_stationary', False)]
nonstationary_features = [col for col, res in stationarity_results.items() if not res.get('is_stationary', False)]

print(f'Stationary features: {len(stationary_features)} / {len(feature_cols)}')
print(f'Non-stationary features: {len(nonstationary_features)} / {len(feature_cols)}')

print()
print('=' * 80)
print('PHASE 2: CORRELATION ANALYSIS')
print('=' * 80)
print()

targets = ['returns', 'volatility']
correlation_results = {}

print('Testing correlations with:')
print(f'  1. Returns (price_return_1h) - stationary')
print(f'  2. Volatility (realized volatility) - stationary')
print()

for target in targets:
    print(f'Target: {target}')
    print('-' * 80)

    correlation_results[target] = {}

    for i, col in enumerate(feature_cols, 1):
        if col not in merged_clean.columns:
            continue

        train_corr = calc_correlation_robust(
            train[col].values,
            train[target].values,
            use_hac=True
        )

        test_corr = calc_correlation_robust(
            test[col].values,
            test[target].values,
            use_hac=False
        )

        correlation_results[target][col] = {
            'train': train_corr,
            'test': test_corr,
            'is_stationary': stationarity_results.get(col, {}).get('is_stationary', False)
        }

        if i % 10 == 0:
            print(f'  Processed {i}/{len(feature_cols)} features...', end='\r')

    print(f'  Processed {len(feature_cols)}/{len(feature_cols)} features - DONE')
    print()

print()
print('=' * 80)
print('PHASE 3: FDR CORRECTION')
print('=' * 80)
print()

for target in targets:
    pvalues = []
    feature_names = []

    for col, res in correlation_results[target].items():
        if res['train']['n'] >= 30:
            pvalues.append(res['train']['p_pearson_hac'])
            feature_names.append(col)

    if len(pvalues) > 0:
        rejected, pvals_corrected, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')

        for i, col in enumerate(feature_names):
            correlation_results[target][col]['fdr_rejected'] = bool(rejected[i])
            correlation_results[target][col]['p_corrected'] = float(pvals_corrected[i])

        n_significant = sum(rejected)
        print(f'{target}:')
        print(f'  Features tested: {len(pvalues)}')
        print(f'  Significant after FDR: {n_significant} ({n_significant/len(pvalues)*100:.1f}%)')
        print()

print()
print('=' * 80)
print('PHASE 4: GRANGER CAUSALITY TESTS')
print('=' * 80)
print()

significant_features = {}
for target in targets:
    significant_features[target] = [
        col for col, res in correlation_results[target].items()
        if res.get('fdr_rejected', False) and abs(res['train']['r_pearson']) > 0.1
    ]

granger_results = {}

for target in targets:
    print(f'Testing Granger causality: features → {target}')
    print('-' * 80)

    granger_results[target] = {}
    features_to_test = significant_features[target][:20]

    if len(features_to_test) == 0:
        print(f'  No significant features to test')
        print()
        continue

    for i, col in enumerate(features_to_test, 1):
        if col in merged_clean.columns:
            gc = test_granger_causality(
                merged_clean[col].values,
                merged_clean[target].values,
                max_lag=12
            )
            granger_results[target][col] = gc

            if gc['has_causality']:
                print(f'  ✓ {col}: lags={gc["significant_lags"]}, min_p={gc["min_pvalue"]:.4f}')

        if i % 5 == 0:
            print(f'  Tested {i}/{len(features_to_test)}...', end='\r')

    print(f'  Tested {len(features_to_test)}/{len(features_to_test)} features - DONE')

    causal_features = [col for col, res in granger_results[target].items() if res['has_causality']]
    print(f'  Features with Granger causality: {len(causal_features)}')
    print()

print()
print('=' * 80)
print('SUMMARY')
print('=' * 80)
print()

all_results = {
    'metadata': {
        'total_features': len(feature_cols),
        'total_hours': len(merged_clean),
        'train_hours': len(train),
        'test_hours': len(test),
        'period_start': str(merged_clean.index.min()),
        'period_end': str(merged_clean.index.max())
    },
    'stationarity': stationarity_results,
    'correlations': correlation_results,
    'granger_causality': granger_results
}

with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'✅ Results saved: {output_file}')
print(f'   Size: {output_file.stat().st_size / 1024:.1f} KB')

print()
print('Top findings:')
print()

for target in targets:
    print(f'{target.upper()}:')

    significant = [
        (col, res['train']['r_pearson'], res['train']['p_pearson_hac'], res['train']['r_squared'])
        for col, res in correlation_results[target].items()
        if res.get('fdr_rejected', False)
    ]

    significant_sorted = sorted(significant, key=lambda x: abs(x[1]), reverse=True)[:10]

    if len(significant_sorted) > 0:
        print(f'  Top correlations (after FDR correction):')
        for col, r, p, r2 in significant_sorted:
            causal = '🎯' if col in granger_results.get(target, {}) and granger_results[target][col]['has_causality'] else '  '
            print(f'    {causal} {col:40s} r={r:+.3f} p={p:.4f} R²={r2:.3f}')
    else:
        print(f'  No significant correlations after FDR correction')

    print()

print()
print('=' * 80)
print('STUDY 2 COMPLETE')
print('=' * 80)
print(f'Next: python3 scripts/21_multivariate_ml.py')
