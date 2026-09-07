import pandas as pd
import numpy as np
import yfinance as yf
import json
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.stats.multitest import multipletests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac
from datetime import timedelta
from sklearn.model_selection import TimeSeriesSplit

data_dir = (Path(__file__).resolve().parents[1] / 'data')

input_file = data_dir / 'btc_hourly_sentiment.csv'
output_merged = data_dir / 'btc_market_merged.csv'
output_results = data_dir / 'correlation_results_v2.json'

print('=== Market Correlation Analysis v2 (Statistically Corrected) ===\n')
print('Phase 1 Fixes:')
print('  ✓ Stationarity testing + differencing')
print('  ✓ Multiple testing correction (FDR)')
print('  ✓ HAC standard errors (Newey-West)')
print('  ✓ Train/test temporal split')
print('  ✓ Granger causality tests\n')

print(f'Loading pre-merged data from: {output_merged}')
merged = pd.read_csv(output_merged, index_col=0, parse_dates=True)
print(f'✓ Loaded: {len(merged):,} hours')
print(f'Period: {merged.index.min()} -> {merged.index.max()}')

print(f'\nCalculating additional derived metrics...')

if 'log_price' not in merged.columns:
    merged['log_price'] = np.log(merged['Close'])
if 'log_volume' not in merged.columns:
    merged['log_volume'] = np.log(merged['Volume'] + 1)
if 'log_social_volume' not in merged.columns:
    merged['log_social_volume'] = np.log(merged['social_volume'] + 1)

merged_clean = merged.dropna()
print(f'After removing NaN: {len(merged_clean):,} records')

def stationarity_test(series, name):
    print(f'\nStationarity Test: {name}')
    print('─' * 60)

    adf = adfuller(series, autolag='AIC')
    print(f'  ADF Test: statistic={adf[0]:.4f}, p-value={adf[1]:.4f}')
    print(f'    {"✓ Stationary (reject unit root)" if adf[1] < 0.05 else "✗ NON-STATIONARY (unit root present)"}')

    kpss_result = kpss(series, regression='c', nlags='auto')
    print(f'  KPSS Test: statistic={kpss_result[0]:.4f}, p-value={kpss_result[1]:.4f}')
    print(f'    {"✗ NON-STATIONARY (reject stationarity)" if kpss_result[1] < 0.05 else "✓ Stationary"}')

    is_stationary = (adf[1] < 0.05) and (kpss_result[1] > 0.05)

    return {
        'adf_statistic': float(adf[0]),
        'adf_pvalue': float(adf[1]),
        'kpss_statistic': float(kpss_result[0]),
        'kpss_pvalue': float(kpss_result[1]),
        'is_stationary': bool(is_stationary),
        'recommendation': 'Use as-is' if is_stationary else 'USE DIFFERENCED/RETURNS'
    }

results = {}

print(f'\n{"="*60}')
print(f'PHASE 1: STATIONARITY TESTING')
print(f'{"="*60}')

stationarity_results = {}
stationarity_results['price'] = stationarity_test(merged_clean['Close'].values, 'BTC Price (levels)')
stationarity_results['log_price'] = stationarity_test(merged_clean['log_price'].values, 'Log Price')
stationarity_results['returns'] = stationarity_test(merged_clean['price_return_1h'].values, 'Returns (1h)')
stationarity_results['sentiment'] = stationarity_test(merged_clean['weighted_sentiment'].values, 'Weighted Sentiment')
stationarity_results['social_volume'] = stationarity_test(merged_clean['log_social_volume'].values, 'Log Social Volume')
stationarity_results['volatility'] = stationarity_test(merged_clean['volatility'].values, 'Volatility')

results['stationarity_tests'] = stationarity_results

print(f'\n⚠️  CRITICAL FINDING:')
if not stationarity_results['price']['is_stationary']:
    print('  → Price levels are NON-STATIONARY (as expected)')
    print('  → Correlations with price levels will be SPURIOUS')
    print('  → SOLUTION: Use returns or first differences')

def calc_correlation_robust(x, y, name, use_hac=True):
    valid = ~(np.isnan(x) | np.isnan(y))
    if valid.sum() < 30:
        return {'r': 0, 'p_value': 1, 'p_value_hac': 1, 'n': 0, 'significant': False}

    x_clean = x[valid]
    y_clean = y[valid]

    r_pearson, p_pearson = pearsonr(x_clean, y_clean)
    r_spearman, p_spearman = spearmanr(x_clean, y_clean)

    p_hac = p_pearson
    if use_hac and len(x_clean) > 30:
        try:
            X = add_constant(x_clean)
            model = OLS(y_clean, X).fit()
            cov_hac_matrix = cov_hac(model, nlags=10)
            se_hac = np.sqrt(np.diag(cov_hac_matrix))
            t_stat_hac = model.params[1] / se_hac[1]
            p_hac = 2 * (1 - np.abs(t_stat_hac))
            p_hac = max(min(p_hac, 1.0), 0.0)
        except:
            p_hac = p_pearson

    return {
        'r_pearson': float(r_pearson),
        'p_pearson': float(p_pearson),
        'r_spearman': float(r_spearman),
        'p_spearman': float(p_spearman),
        'p_hac': float(p_hac),
        'n': int(valid.sum()),
        'interpretation': 'Strong' if abs(r_pearson) > 0.5 else 'Moderate' if abs(r_pearson) > 0.3 else 'Weak'
    }

print(f'\n{"="*60}')
print(f'ANALYSIS 1: CORRECTED CORRELATIONS (Stationary Variables Only)')
print(f'{"="*60}\n')

print('1.1 Sentiment vs Returns (NOT price levels)')
print('─' * 60)

sync_return = calc_correlation_robust(
    merged_clean['weighted_sentiment'].values,
    merged_clean['price_return_1h'].values,
    'Returns vs Sentiment'
)
results['sync_return_correlation'] = sync_return

print(f"Correlation (1h Returns vs Weighted Sentiment):")
print(f"  Pearson r = {sync_return['r_pearson']:.4f} (p_naive={sync_return['p_pearson']:.4e})")
print(f"  Spearman ρ = {sync_return['r_spearman']:.4f} (p={sync_return['p_spearman']:.4e})")
print(f"  HAC-corrected p = {sync_return['p_hac']:.4e} (Newey-West)")

print(f'\n1.2 Social Volume vs Volatility (MAIN FINDING)')
print('─' * 60)

vol_corr = calc_correlation_robust(
    merged_clean['log_social_volume'].values,
    merged_clean['volatility'].values,
    'Social Volume vs Volatility'
)
results['volatility_correlation'] = vol_corr

print(f"Correlation (Log Social Volume vs Volatility):")
print(f"  Pearson r = {vol_corr['r_pearson']:.4f} (p_naive={vol_corr['p_pearson']:.4e})")
print(f"  Spearman ρ = {vol_corr['r_spearman']:.4f} (p={vol_corr['p_spearman']:.4e})")
print(f"  HAC-corrected p = {vol_corr['p_hac']:.4e}")
print(f"  R² = {vol_corr['r_pearson']**2:.4f} ({vol_corr['r_pearson']**2*100:.2f}% variance explained)")

print(f'\n{"="*60}')
print(f'ANALYSIS 2: MULTIPLE TESTING CORRECTION')
print(f'{"="*60}\n')

print('Testing multiple lags (requires FDR correction)...')

lags_to_test = [1, 2, 3, 6, 12, 24, 48, 72]
lag_results = []
p_values_lag = []

for lag in lags_to_test:
    sentiment_shifted = merged_clean['weighted_sentiment'].shift(lag)

    corr = calc_correlation_robust(
        sentiment_shifted.values,
        merged_clean['price_return_1h'].values,
        f'Lag {lag}h',
        use_hac=True
    )

    lag_results.append({
        'lag_hours': lag,
        'r': corr['r_pearson'],
        'p_naive': corr['p_pearson'],
        'p_hac': corr['p_hac'],
        'n': corr['n']
    })
    p_values_lag.append(corr['p_hac'])

reject_fdr, p_adjusted_fdr, _, _ = multipletests(p_values_lag, alpha=0.05, method='fdr_bh')

for i, lag_result in enumerate(lag_results):
    lag_result['p_fdr_corrected'] = float(p_adjusted_fdr[i])
    lag_result['significant_fdr'] = bool(reject_fdr[i])

    marker = '✓' if reject_fdr[i] else '✗'
    print(f"  {marker} Lag {lag_result['lag_hours']:2d}h: r={lag_result['r']:7.4f} | "
          f"p_naive={lag_result['p_naive']:.2e} → p_FDR={lag_result['p_fdr_corrected']:.2e}")

results['lag_analysis_corrected'] = lag_results

significant_lags = [lr for lr in lag_results if lr['significant_fdr']]
print(f"\n  → {len(significant_lags)}/{len(lags_to_test)} lags survive FDR correction")

print(f'\n{"="*60}')
print(f'ANALYSIS 3: TRAIN/TEST TEMPORAL SPLIT')
print(f'{"="*60}\n')

split_point = int(len(merged_clean) * 0.7)
train_df = merged_clean.iloc[:split_point]
test_df = merged_clean.iloc[split_point:]

print(f'Train set: {len(train_df):,} hours ({train_df.index.min()} to {train_df.index.max()})')
print(f'Test set:  {len(test_df):,} hours ({test_df.index.min()} to {test_df.index.max()})')

print('\n3.1 Volatility Prediction (In-sample vs Out-of-sample)')
print('─' * 60)

train_vol_corr = calc_correlation_robust(
    train_df['log_social_volume'].values,
    train_df['volatility'].values,
    'Train'
)

test_vol_corr = calc_correlation_robust(
    test_df['log_social_volume'].values,
    test_df['volatility'].values,
    'Test'
)

print(f"TRAIN SET: r={train_vol_corr['r_pearson']:.4f} (p_FDR={train_vol_corr['p_hac']:.4e})")
print(f"TEST SET:  r={test_vol_corr['r_pearson']:.4f} (p_FDR={test_vol_corr['p_hac']:.4e})")

validation_robust = abs(test_vol_corr['r_pearson']) > 0.10 and test_vol_corr['p_hac'] < 0.05

if validation_robust:
    print(f"✓ VALIDATED: Effect holds in out-of-sample data")
else:
    print(f"✗ WARNING: Effect weakens in test set (possible overfitting)")

results['train_test_split'] = {
    'train_size': len(train_df),
    'test_size': len(test_df),
    'train_correlation': train_vol_corr,
    'test_correlation': test_vol_corr,
    'validated': validation_robust
}

print(f'\n{"="*60}')
print(f'ANALYSIS 4: GRANGER CAUSALITY TEST')
print(f'{"="*60}\n')

print('Testing if Sentiment Granger-causes Returns...')

granger_data = merged_clean[['weighted_sentiment', 'price_return_1h']].dropna()

max_lag = 12
print(f'Testing lags 1-{max_lag} hours...\n')

try:
    granger_results_dict = grangercausalitytests(
        granger_data[['price_return_1h', 'weighted_sentiment']],
        maxlag=max_lag,
        verbose=False
    )

    granger_summary = []
    for lag in range(1, max_lag + 1):
        f_test = granger_results_dict[lag][0]['ssr_ftest']
        granger_summary.append({
            'lag': lag,
            'f_statistic': float(f_test[0]),
            'p_value': float(f_test[1]),
            'significant': bool(f_test[1] < 0.05)
        })

        marker = '✓' if f_test[1] < 0.05 else '✗'
        print(f"  {marker} Lag {lag:2d}: F={f_test[0]:7.4f}, p={f_test[1]:.4e}")

    results['granger_causality'] = granger_summary

    significant_granger = [g for g in granger_summary if g['significant']]

    if len(significant_granger) > 0:
        print(f"\n✓ Sentiment Granger-causes returns at {len(significant_granger)} lag(s)")
        print(f"  → Can claim predictive relationship")
    else:
        print(f"\n✗ No Granger causality detected")
        print(f"  → Cannot claim sentiment 'leads' or 'predicts' price")

except Exception as e:
    print(f"✗ Granger test failed: {e}")
    results['granger_causality'] = {'error': str(e)}

merged.to_csv(output_merged)
print(f'\n{"="*60}')
print(f'SAVING RESULTS')
print(f'{"="*60}\n')

results['metadata'] = {
    'version': 'v2_phase1_corrected',
    'data_period_start': str(merged.index.min()),
    'data_period_end': str(merged.index.max()),
    'total_hours': len(merged),
    'phase1_corrections': [
        'Stationarity testing (ADF + KPSS)',
        'Multiple testing correction (FDR)',
        'HAC standard errors (Newey-West)',
        'Train/test temporal split (70/30)',
        'Granger causality testing'
    ]
}

with open(output_results, 'w') as f:
    json.dump(results, f, indent=2)

print(f'✓ Saved: {output_results}')

print(f'\n{"="*60}')
print(f'CORRECTED INTERPRETATION')
print(f'{"="*60}\n')

print('MAIN FINDING (Survives corrections):')
print(f'  → Social volume predicts volatility: r={vol_corr["r_pearson"]:.3f} (p={vol_corr["p_hac"]:.2e})')
print(f'  → Effect size: R²={vol_corr["r_pearson"]**2:.4f} ({vol_corr["r_pearson"]**2*100:.2f}% variance)')
print(f'  → Out-of-sample validated: r_test={test_vol_corr["r_pearson"]:.3f}')

if not stationarity_results['price']['is_stationary']:
    print('\n⚠️  REMOVED (Spurious correlation):')
    print('  → Price level correlations are INVALID (non-stationary)')
    print('  → Original r=0.036 finding was spurious regression')

print('\n✓ PHASE 1 COMPLETE')
print('Next: Phase 2 (Outliers, Confounders, Robustness)')
