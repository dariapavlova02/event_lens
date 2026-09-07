#!/usr/bin/env python3
"""Multi-asset Telegram sentiment pipeline for BTC/ETH/SOL."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
yf = None
yimport = False
try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None
    yimport = False
else:
    yimport = True

from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "multi_asset" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class AssetConfig:
    name: str
    ticker: str
    yfinance_symbol: str
    text_keywords: List[str]
    name_keywords: List[str] = field(default_factory=list)

    @property
    def tickers(self) -> List[str]:
        all_tickers = {self.ticker.upper()}
        return sorted(all_tickers)


ASSETS: Dict[str, AssetConfig] = {
    "BTC": AssetConfig(
        name="Bitcoin",
        ticker="BTC",
        yfinance_symbol="BTC-USD",
        text_keywords=["btc", "bitcoin"],
        name_keywords=["bitcoin"],
    ),
    "ETH": AssetConfig(
        name="Ethereum",
        ticker="ETH",
        yfinance_symbol="ETH-USD",
        text_keywords=["eth", "ethereum"],
        name_keywords=["ethereum", "ether"],
    ),
    "SOL": AssetConfig(
        name="Solana",
        ticker="SOL",
        yfinance_symbol="SOL-USD",
        text_keywords=["sol", "solana"],
        name_keywords=["solana", "sol"],
    ),
}

CATEGORY_LIST = [
    "REGULATION",
    "MARKET",
    "HACK",
    "OPINION",
    "COMMUNITY",
    "TECHNOLOGY",
    "UNKNOWN",
]
MARKET_COLS = {
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "price_return_1h",
    "price_return_3h",
    "price_return_6h",
    "price_return_12h",
    "price_return_24h",
    "volatility",
    "price_change_1h",
    "price_change_3h",
    "price_change_6h",
    "price_change_12h",
    "price_change_24h",
    "log_price",
}


def json_default(val):
    if isinstance(val, (np.floating, np.float32, np.float64)):
        return float(val)
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (pd.Timestamp,)):
        return val.isoformat()
    return str(val)


def load_all_messages() -> pd.DataFrame:
    path = DATA_DIR / "all_messages_tbsa.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def parse_targets(raw: str) -> List[dict]:
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def normalize_sentiment(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        mapping = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        return mapping.get(value.lower(), 0.0)
    return 0.0


def select_asset_target(targets: List[dict], config: AssetConfig) -> Optional[dict]:
    candidates = []
    normalized_names = config.name_keywords
    for target in targets:
        ticker = (target.get("ticker") or "").upper()
        name = (target.get("name") or "").lower()
        if ticker in config.tickers:
            candidates.append(target)
            continue
        if any(alias in name for alias in normalized_names):
            candidates.append(target)
    if not candidates:
        return None
    return max(candidates, key=lambda x: x.get("impact", 0))


def extract_asset_messages(source_df: pd.DataFrame, config: AssetConfig) -> pd.DataFrame:
    records = []
    for row in source_df.itertuples():
        if getattr(row, "tbsa_has_result", False) is False:
            continue
        targets = parse_targets(getattr(row, "tbsa_targets_json", ""))
        target = select_asset_target(targets, config)
        if not target:
            continue
        sentiment = normalize_sentiment(target.get("sentiment"))
        impact = target.get("impact", 1) or 0
        if impact == 0:
            impact = 1
        records.append(
            {
                "date": row.date,
                "text": getattr(row, "text", ""),
                "channel": getattr(row, "channel", ""),
                "sentiment": sentiment,
                "impact": float(impact),
                "category": target.get("category", "UNKNOWN") or "UNKNOWN",
                "is_fact": bool(getattr(row, "tbsa_is_fact", False)),
                "global_sentiment": getattr(row, "tbsa_global_sentiment", 0.0),
                "views": getattr(row, "views", 0) or 0,
                "forwards": getattr(row, "forwards", 0) or 0,
            }
        )
    return pd.DataFrame(records)


def weighted_sentiment(df_group: pd.DataFrame) -> float:
    total = df_group["impact"].sum()
    if total == 0:
        return 0.0
    return float((df_group["sentiment"] * df_group["impact"]).sum() / total)


def attention_score(df_group: pd.DataFrame) -> float:
    return float((df_group["views"] * (1 + df_group["forwards"])) .sum())


def category_weighted(df_group: pd.DataFrame, category: str) -> float:
    subset = df_group[df_group["category"] == category]
    if subset.empty:
        return 0.0
    total = subset["impact"].sum()
    if total == 0:
        return 0.0
    return float((subset["sentiment"] * subset["impact"]).sum() / total)


def category_volume(df_group: pd.DataFrame, category: str) -> float:
    return float(df_group[df_group["category"] == category]["impact"].sum())


def high_impact_sentiment(df_group: pd.DataFrame, threshold: float) -> float:
    subset = df_group[df_group["impact"] >= threshold]
    if subset.empty:
        return 0.0
    total = subset["impact"].sum()
    if total == 0:
        return 0.0
    return float((subset["sentiment"] * subset["impact"]).sum() / total)


def fact_opinion_sentiment(df_group: pd.DataFrame, fact: bool) -> float:
    subset = df_group[df_group["is_fact"] == fact]
    if subset.empty:
        return 0.0
    total = subset["impact"].sum()
    if total == 0:
        total = len(subset)
        if total == 0:
            return 0.0
        return float(subset["sentiment"].mean())
    return float((subset["sentiment"] * subset["impact"]).sum() / total)


def build_hourly_features(msg_df: pd.DataFrame) -> pd.DataFrame:
    if msg_df.empty:
        return pd.DataFrame()
    df = msg_df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    def resample_block(group):
        base = {
            "sentiment_baseline": group["sentiment"].mean() if len(group) else 0.0,
            "social_volume": group["impact"].sum(),
            "message_count": len(group),
            "sentiment_high_impact": high_impact_sentiment(group, 8),
            "sentiment_medium_impact": high_impact_sentiment(group, 5),
            "volume_high_impact": group[group["impact"] >= 8]["impact"].sum(),
            "volume_medium_impact": group[(group["impact"] >= 5) & (group["impact"] < 8)]["impact"].sum(),
            "volume_low_impact": group[group["impact"] < 5]["impact"].sum(),
            "count_high_impact": len(group[group["impact"] >= 8]),
            "sentiment_fact": fact_opinion_sentiment(group, True),
            "sentiment_opinion_type": fact_opinion_sentiment(group, False),
            "volume_fact": group[group["is_fact"]]["impact"].sum(),
            "volume_opinion_type": group[~group["is_fact"]]["impact"].sum(),
            "fact_ratio": float(group["is_fact"].sum() / len(group)) if len(group) else 0.0,
            "attention_weighted_sentiment": weighted_sentiment(group.assign(impact=group["impact"] * group["views"].clip(lower=1) * (1 + group["forwards"]))),
            "total_attention": attention_score(group),
            "total_views": group["views"].sum(),
            "total_forwards": group["forwards"].sum(),
            "sentiment_positive": float((group["sentiment"] > 0).sum()),
            "sentiment_negative": float((group["sentiment"] < 0).sum()),
            "sentiment_neutral": float((group["sentiment"] == 0).sum()),
            "sentiment_dispersion": float(group["sentiment"].std() if len(group) > 1 else 0.0),
            "impact_mean": float(group["impact"].mean()) if len(group) else 0.0,
            "impact_max": float(group["impact"].max()) if len(group) else 0.0,
            "global_sentiment_avg": group["global_sentiment"].mean() if len(group) else 0.0,
            "sentiment_raw_avg": group["sentiment"].mean() if len(group) else 0.0,
            "weighted_sentiment": weighted_sentiment(group),
        }
        for category in CATEGORY_LIST:
            base[f"sentiment_{category.lower()}"] = category_weighted(group, category)
            base[f"volume_{category.lower()}"] = category_volume(group, category)
        base["social_volume"] = float(base["social_volume"])
        return pd.Series(base)

    hourly = df.resample("1h").apply(resample_block)
    hourly = hourly[hourly["message_count"] > 0]

    feature_groups = [
        "sentiment_baseline",
        "social_volume",
        "sentiment_regulation",
        "sentiment_market",
        "sentiment_high_impact",
        "volume_high_impact",
    ]
    for col in feature_groups:
        if col not in hourly.columns:
            continue
        hourly[f"{col}_delta_1h"] = hourly[col].diff(1)
        hourly[f"{col}_delta_3h"] = hourly[col].diff(3)
        hourly[f"{col}_ma_6h"] = hourly[col].rolling(6, min_periods=1).mean()
        hourly[f"{col}_ma_24h"] = hourly[col].rolling(24, min_periods=1).mean()
        hourly[f"{col}_momentum"] = hourly[col] - hourly[f"{col}_ma_24h"]
    hourly["acceleration_sentiment"] = hourly["sentiment_baseline_delta_1h"].diff(1)
    hourly["acceleration_volume"] = hourly["social_volume_delta_1h"].diff(1)
    return hourly.dropna(how="all")


def download_market(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not yimport:
        raise RuntimeError("yfinance is not available in this environment")
    data = yf.download(
        symbol,
        start=start - timedelta(days=1),
        end=end + timedelta(days=1),
        interval="1h",
        progress=False,
        auto_adjust=True,
    )
    if data.empty:
        raise RuntimeError(f"Failed to download data for {symbol}")
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")
    return data


def attach_market_features(hourly: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        hourly,
        market[["Open", "High", "Low", "Close", "Volume"]],
        left_index=True,
        right_index=True,
        how="inner",
    )
    merged["price_return_1h"] = merged["Close"].pct_change()
    merged["price_return_3h"] = merged["Close"].pct_change(periods=3)
    merged["price_return_6h"] = merged["Close"].pct_change(periods=6)
    merged["price_return_12h"] = merged["Close"].pct_change(periods=12)
    merged["price_return_24h"] = merged["Close"].pct_change(periods=24)
    merged["volatility"] = (merged["High"] - merged["Low"]) / merged["Close"]
    merged["price_change_1h"] = merged["Close"].shift(-1) - merged["Close"]
    merged["price_change_3h"] = merged["Close"].shift(-3) - merged["Close"]
    merged["price_change_6h"] = merged["Close"].shift(-6) - merged["Close"]
    merged["price_change_12h"] = merged["Close"].shift(-12) - merged["Close"]
    merged["price_change_24h"] = merged["Close"].shift(-24) - merged["Close"]
    merged["log_price"] = np.log(merged["Close"])
    merged.dropna(inplace=True)
    return merged


def stationarity_test(series: pd.Series) -> Dict[str, float]:
    if len(series.dropna()) < 30:
        return {"is_stationary": False, "note": "insufficient_data"}
    try:
        adf = adfuller(series.dropna(), autolag="AIC")
        kpss_res = kpss(series.dropna(), regression="c", nlags="auto")
        return {
            "adf_statistic": float(adf[0]),
            "adf_pvalue": float(adf[1]),
            "kpss_statistic": float(kpss_res[0]),
            "kpss_pvalue": float(kpss_res[1]),
            "is_stationary": bool(adf[1] < 0.05 and kpss_res[1] > 0.05),
        }
    except Exception:
        return {"is_stationary": False, "note": "test_failed"}


def calc_corr(x: np.ndarray, y: np.ndarray, use_hac: bool = True) -> Dict[str, float]:
    mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    if mask.sum() < 30:
        return {"n": int(mask.sum()), "r_pearson": 0.0, "p_pearson": 1.0,
                "r_spearman": 0.0, "p_spearman": 1.0, "r_kendall": 0.0, "p_kendall": 1.0,
                "p_pearson_hac": 1.0, "r_squared": 0.0}
    x_clean = x[mask]
    y_clean = y[mask]
    r_pearson, p_pearson = pearsonr(x_clean, y_clean)
    r_spearman, p_spearman = spearmanr(x_clean, y_clean)
    r_kendall, p_kendall = kendalltau(x_clean, y_clean)
    p_hac = p_pearson
    if use_hac and len(x_clean) > 50:
        try:
            X = np.column_stack([np.ones(len(x_clean)), x_clean])
            model = OLS(y_clean, X).fit()
            cov = cov_hac(model, nlags=12)
            se_hac = np.sqrt(np.diag(cov))[1]
            t_stat = model.params[1] / se_hac
            p_hac = 2 * (1 - stats.t.cdf(abs(t_stat), len(x_clean) - 2))
        except Exception:
            pass
    return {
        "n": int(mask.sum()),
        "r_pearson": float(r_pearson),
        "p_pearson": float(p_pearson),
        "p_pearson_hac": float(p_hac),
        "r_spearman": float(r_spearman),
        "p_spearman": float(p_spearman),
        "r_kendall": float(r_kendall),
        "p_kendall": float(p_kendall),
        "r_squared": float(r_pearson ** 2),
    }


def test_granger(feature: pd.Series, target: pd.Series, max_lag: int = 12) -> Dict[str, object]:
    aligned = pd.concat([target, feature], axis=1).dropna()
    if len(aligned) < 100:
        return {"has_causality": False, "min_pvalue": 1.0, "significant_lags": []}
    try:
        gc = grangercausalitytests(aligned, maxlag=max_lag, verbose=False)
        pvalues = [gc[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
        significant = [lag for lag, p in enumerate(pvalues, start=1) if p < 0.05]
        return {
            "has_causality": bool(significant),
            "min_pvalue": float(min(pvalues)),
            "significant_lags": significant,
            "pvalues_by_lag": [float(p) for p in pvalues],
        }
    except Exception:
        return {"has_causality": False, "min_pvalue": 1.0, "significant_lags": []}


def run_univariate_analysis(asset: str, merged: pd.DataFrame, output_path: Path) -> dict:
    dataset = merged.copy()
    dataset["returns"] = dataset["price_return_1h"]
    dataset.dropna(inplace=True)
    train_size = int(0.8 * len(dataset))
    train = dataset.iloc[:train_size]
    test = dataset.iloc[train_size:]

    feature_cols = [col for col in dataset.columns if col not in MARKET_COLS and col not in {"returns"}]

    stationarity = {col: stationarity_test(dataset[col]) for col in feature_cols}

    targets = ["returns", "volatility"]
    correlations: Dict[str, Dict[str, dict]] = {t: {} for t in targets}
    for target in targets:
        for col in feature_cols:
            train_res = calc_corr(train[col].values, train[target].values, use_hac=True)
            test_res = calc_corr(test[col].values, test[target].values, use_hac=False)
            correlations[target][col] = {
                "train": train_res,
                "test": test_res,
                "is_stationary": stationarity.get(col, {}).get("is_stationary", False),
            }

    for target in targets:
        pvals = []
        cols = []
        for col, res in correlations[target].items():
            if res["train"]["n"] >= 30:
                pvals.append(res["train"]["p_pearson_hac"])
                cols.append(col)
        if not pvals:
            continue
        rejected, corrected, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        for flag, col, p in zip(rejected, cols, corrected):
            correlations[target][col]["fdr_rejected"] = bool(flag)
            correlations[target][col]["p_corrected"] = float(p)

    granger_results = {target: {} for target in targets}
    for target in targets:
        candidates = [col for col, res in correlations[target].items() if res.get("fdr_rejected") and abs(res["train"]["r_pearson"]) > 0.1]
        for col in candidates[:20]:
            granger_results[target][col] = test_granger(dataset[col], dataset[target])

    results = {
        "metadata": {
            "asset": asset,
            "period_start": dataset.index.min().isoformat() if not dataset.empty else None,
            "period_end": dataset.index.max().isoformat() if not dataset.empty else None,
            "total_rows": len(dataset),
            "train_rows": len(train),
            "test_rows": len(test),
        },
        "stationarity": stationarity,
        "correlations": correlations,
        "granger_causality": granger_results,
    }

    with output_path.open("w") as f:
        json.dump(results, f, indent=2, default=json_default)
    return results


def evaluate_model(model, X_train, y_train, X_test, y_test, scaler=None):
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled, X_test_scaled = X_train, X_test
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    return {
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "train_rmse": float(math.sqrt(mean_squared_error(y_train, y_train_pred))),
        "test_rmse": float(math.sqrt(mean_squared_error(y_test, y_test_pred))),
        "train_mae": float(mean_absolute_error(y_train, y_train_pred)),
        "test_mae": float(mean_absolute_error(y_test, y_test_pred)),
        "n_features_used": int((np.abs(getattr(model, "coef_", [])) > 0).sum()) if hasattr(model, "coef_") else len(X_train.columns),
    }


def cross_validate_model(model, X, y, splits=5):
    tscv = TimeSeriesSplit(n_splits=splits)
    scaler = StandardScaler()
    r2_scores, rmse_scores = [], []
    for train_idx, val_idx in tscv.split(X):
        X_tr = scaler.fit_transform(X.iloc[train_idx])
        X_val = scaler.transform(X.iloc[val_idx])
        model.fit(X_tr, y.iloc[train_idx])
        preds = model.predict(X_val)
        r2_scores.append(r2_score(y.iloc[val_idx], preds))
        rmse_scores.append(math.sqrt(mean_squared_error(y.iloc[val_idx], preds)))
    return {
        "mean_r2": float(np.mean(r2_scores)),
        "std_r2": float(np.std(r2_scores)),
        "mean_rmse": float(np.mean(rmse_scores)),
        "std_rmse": float(np.std(rmse_scores)),
    }


def get_top_features(model, feature_names, top_n=10):
    if not hasattr(model, "coef_"):
        return []
    coefs = np.array(model.coef_)
    idx = np.argsort(np.abs(coefs))[::-1][:top_n]
    return [
        {"feature": feature_names[i], "coefficient": float(coefs[i]), "abs_coefficient": float(abs(coefs[i]))}
        for i in idx if abs(coefs[i]) > 0
    ]


def run_multivariate_models(asset: str, merged: pd.DataFrame, univariate_results: dict, output_path: Path):
    dataset = merged.copy()
    dataset["returns"] = dataset["price_return_1h"]
    dataset.dropna(inplace=True)
    train_size = int(0.8 * len(dataset))
    train = dataset.iloc[:train_size]
    test = dataset.iloc[train_size:]

    models = {
        "lasso": Lasso(alpha=0.001, max_iter=5000),
        "ridge": Ridge(alpha=1.0),
        "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
    }

    results = {"baseline": {}}
    if "sentiment_baseline" in dataset.columns:
        baseline_X_train = train[["sentiment_baseline"]]
        baseline_X_test = test[["sentiment_baseline"]]
        for target in ["returns", "volatility"]:
            perf = evaluate_model(
                Ridge(alpha=0.0),
                baseline_X_train,
                train[target],
                baseline_X_test,
                test[target],
                scaler=StandardScaler(),
            )
            results["baseline"][target] = perf

    for target in ["returns", "volatility"]:
        corr = univariate_results["correlations"].get(target, {})
        significant_features = [col for col, res in corr.items() if res.get("fdr_rejected")]
        significant_features = [col for col in significant_features if col in dataset.columns and col not in MARKET_COLS]
        if not significant_features:
            continue
        X_train = train[significant_features]
        X_test = test[significant_features]
        y_train = train[target]
        y_test = test[target]
        results[target] = {}
        for name, model in models.items():
            perf = evaluate_model(model, X_train, y_train, X_test, y_test, scaler=StandardScaler())
            cv_scores = cross_validate_model(model, X_train, y_train)
            scaler_final = StandardScaler()
            model.fit(scaler_final.fit_transform(X_train), y_train)
            top_features = get_top_features(model, significant_features)
            results[target][name] = {
                "performance": perf,
                "cross_validation": cv_scores,
                "top_features": top_features,
            }
    with output_path.open("w") as f:
        json.dump(results, f, indent=2, default=json_default)
    return results


def process_asset(asset_code: str, config: AssetConfig, all_messages: pd.DataFrame):
    output_dir = OUTPUT_DIR / asset_code
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Processing {asset_code} ({config.name}) ===")
    asset_msgs = extract_asset_messages(all_messages, config)
    if asset_msgs.empty:
        print("  No messages found for asset")
        return
    asset_msgs.to_csv(output_dir / "messages.csv", index=False)
    hourly = build_hourly_features(asset_msgs)
    if hourly.empty:
        print("  No hourly features computed")
        return
    hourly.to_csv(output_dir / "hourly_features.csv")
    market = download_market(config.yfinance_symbol, hourly.index.min(), hourly.index.max())
    merged = attach_market_features(hourly, market)
    merged.to_csv(output_dir / "market_merged.csv")
    uni_results = run_univariate_analysis(asset_code, merged, output_dir / "univariate_analysis.json")
    run_multivariate_models(asset_code, merged, uni_results, output_dir / "ml_results.json")
    print(f"  Done. Rows: {len(merged)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-asset sentiment pipeline")
    parser.add_argument("--assets", nargs="*", default=["ETH", "SOL"], help="Asset tickers to process (BTC/ETH/SOL)")
    return parser.parse_args()


def main():
    args = parse_args()
    all_messages = load_all_messages()
    for asset in args.assets:
        asset = asset.upper()
        if asset not in ASSETS:
            print(f"Unknown asset {asset}, skipping")
            continue
        process_asset(asset, ASSETS[asset], all_messages)


if __name__ == "__main__":
    main()
