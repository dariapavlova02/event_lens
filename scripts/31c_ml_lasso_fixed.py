#!/usr/bin/env python3
"""
Script 31c: ML Regression with LASSO (Fixed Temporal Validation)
================================================================
IMPORTANT FIX: Uses TimeSeriesSplit for INNER cross-validation
to avoid look-ahead bias in hyperparameter tuning.

Previous issue: LassoCV(cv=5) uses random K-fold, not temporal
Solution: LassoCV(cv=TimeSeriesSplit(n_splits=3))

Methodology:
- Outer loop: 5-fold Walk-Forward TimeSeriesSplit
- Inner loop: 3-fold TimeSeriesSplit for alpha selection
- Scaler: fit ONLY on train fold data
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
RESULTS_DIR = (Path(__file__).resolve().parents[1] / 'results')

print("="*70)
print("ML REGRESSION: LASSO (FIXED TEMPORAL VALIDATION)")
print("="*70)
print()
print("🔧 Improvements over 31b:")
print("   ✅ Inner CV uses TimeSeriesSplit (not random)")
print("   ✅ Nested CV for unbiased hyperparameter selection")
print("   ✅ Detailed fold-by-fold reporting")
print()

# Load Data
print("📊 Loading data...")
graph_df = pd.read_csv(DATA_DIR / 'graph_features_temporal.csv', index_col=0, parse_dates=True)
market_df = yf.download("BTC-USD", start=graph_df.index.min(), end=graph_df.index.max(), progress=False)
if isinstance(market_df.columns, pd.MultiIndex):
    market_df = market_df.xs('BTC-USD', level=1, axis=1)

graph_df.index = pd.to_datetime(graph_df.index)
market_df['Returns'] = market_df['Close'].pct_change()
market_df['Log_Returns'] = np.log(market_df['Close'] / market_df['Close'].shift(1))
market_df['Volatility'] = market_df['Log_Returns'].rolling(20).std() * np.sqrt(365)

# Target: 5-day forward volatility (shifted into future)
market_df['Target_Vol'] = market_df['Volatility'].shift(-5)

full_df = graph_df.join(market_df[['Returns', 'Volatility', 'Target_Vol']], how='inner').dropna()
print(f"   Samples: {len(full_df)}")
print(f"   Period: {full_df.index.min().date()} → {full_df.index.max().date()}")

# Feature sets
baseline_features = ['Volatility', 'Returns', 'graph_nodes', 'graph_edges']
graph_features = baseline_features + [
    'graph_density', 'graph_transitivity', 'graph_degree_gini',
    'graph_centrality_stability', 'graph_modularity'
]

# Outer CV: Walk-Forward
outer_tscv = TimeSeriesSplit(n_splits=5)

# Inner CV: Also TimeSeriesSplit (FIXED!)
inner_tscv = TimeSeriesSplit(n_splits=3)

def evaluate_with_nested_cv(features, name, save_details=False):
    """
    Nested Walk-Forward Cross-Validation:
    - Outer loop: 5 folds for generalization estimate
    - Inner loop: 3 folds for hyperparameter selection (alpha)
    """
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")

    X = full_df[features]
    y = full_df['Target_Vol']

    fold_results = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scaler: fit ONLY on train fold
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # LassoCV with TEMPORAL inner CV (FIXED!)
        model = LassoCV(
            cv=inner_tscv,  # ← TimeSeriesSplit instead of random K-fold
            alphas=np.logspace(-4, 1, 50),
            max_iter=10000,
            random_state=42
        )
        model.fit(X_train_s, y_train)

        # Predict
        y_pred = model.predict(X_test_s)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        fold_results.append({
            'fold': fold_idx + 1,
            'train_rows': len(X_train),
            'test_rows': len(X_test),
            'best_alpha': model.alpha_,
            'mse': mse,
            'r2': r2,
            'n_features': np.sum(model.coef_ != 0)
        })

        print(f"   Fold {fold_idx+1}: Train={len(X_train)}, Test={len(X_test)}")
        print(f"           Alpha={model.alpha_:.6f}, MSE={mse:.6f}, R²={r2:.4f}")

    # Summary
    mean_mse = np.mean([f['mse'] for f in fold_results])
    std_mse = np.std([f['mse'] for f in fold_results])
    mean_r2 = np.mean([f['r2'] for f in fold_results])
    std_r2 = np.std([f['r2'] for f in fold_results])

    print(f"\n   📊 Summary:")
    print(f"      MSE: {mean_mse:.6f} ± {std_mse:.6f}")
    print(f"      R²:  {mean_r2:.4f} ± {std_r2:.4f}")

    return {
        'name': name,
        'mean_mse': mean_mse,
        'std_mse': std_mse,
        'mean_r2': mean_r2,
        'std_r2': std_r2,
        'folds': fold_results
    }

# Run evaluation
print("\n" + "="*70)
print("RUNNING NESTED WALK-FORWARD CV (Baseline)")
print("="*70)
res_baseline = evaluate_with_nested_cv(baseline_features, "Baseline (Lasso)")

print("\n" + "="*70)
print("RUNNING NESTED WALK-FORWARD CV (Graph-Enhanced)")
print("="*70)
res_graph = evaluate_with_nested_cv(graph_features, "Graph-Enhanced (Lasso)")

# Comparison
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

mse_improvement = (res_baseline['mean_mse'] - res_graph['mean_mse']) / res_baseline['mean_mse']
r2_diff = res_graph['mean_r2'] - res_baseline['mean_r2']

print(f"\nBaseline MSE:     {res_baseline['mean_mse']:.6f} ± {res_baseline['std_mse']:.6f}")
print(f"Graph MSE:        {res_graph['mean_mse']:.6f} ± {res_graph['std_mse']:.6f}")
print(f"MSE Improvement:  {mse_improvement:.2%}")
print(f"R² Difference:    {r2_diff:+.4f}")

# Save results
results = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'target': 'volatility',
    'validation': 'nested_walk_forward_cv',
    'outer_folds': 5,
    'inner_folds': 3,
    'methodology_fixes': [
        'Inner CV uses TimeSeriesSplit (not random K-fold)',
        'Scaler fit only on train fold',
        'No look-ahead bias in hyperparameter selection'
    ],
    'baseline': res_baseline,
    'graph_enhanced': res_graph,
    'improvement': {
        'mse_reduction_pct': mse_improvement * 100,
        'r2_difference': r2_diff
    }
}

# Save to JSONL (append mode for tracking runs)
output_jsonl = RESULTS_DIR / 'nested_cv_runs.jsonl'

# Convert numpy types to Python native for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    return obj

with open(output_jsonl, 'a') as f:
    f.write(json.dumps(convert_numpy({
        'timestamp': results['timestamp'],
        'target': 'volatility',
        'folds': res_graph['folds'],
        'dataset_hash': str(hash(tuple(full_df.index[:10])))
    })) + '\n')

print(f"\n✅ Results appended to: {output_jsonl}")

print("\n" + "="*70)
print("TEMPORAL VALIDATION FIXED!")
print("="*70)
print("""
Previous issue (Script 31b):
   LassoCV(cv=5) → Uses random K-fold for inner CV

Fixed (Script 31c):
   LassoCV(cv=TimeSeriesSplit(n_splits=3)) → Temporal inner CV

Why it matters:
   - Random K-fold can cause data from "future" folds to influence
     hyperparameter selection for "past" folds
   - TimeSeriesSplit ensures all training data is before test data
   - This prevents look-ahead bias in alpha selection
""")
