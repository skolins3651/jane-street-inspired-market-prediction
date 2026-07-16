from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "daily_prices_2016_2025.parquet"

BASE_SPLIT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset_with_features_and_splits_1d.parquet"
)

EXPANDED_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset_with_expanded_features_and_splits_1d.parquet"
)

ORIGINAL_FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "intraday_range",
    "overnight_gap",
    "log_volume",
    "log_dollar_volume",
    "relative_volume_20d",
]

EXPANDED_FEATURE_COLUMNS = [
    # Market-relative features
    "spy_return_1d",
    "spy_return_5d",
    "spy_return_20d",
    "return_1d_minus_spy",
    "return_5d_minus_spy",
    "return_20d_minus_spy",
    # Additional trend features
    "return_3d",
    "return_10d",
    "return_60d",
    "rolling_mean_return_5d",
    "rolling_mean_return_20d",
    "momentum_5d_minus_20d",
    "momentum_20d_minus_60d",
    # Volatility and range features
    "volatility_5d",
    "volatility_60d",
    "volatility_ratio_5d_20d",
    "volatility_ratio_20d_60d",
    "range_5d_mean",
    "range_20d_mean",
    # Volume and liquidity features
    "relative_volume_5d",
    "relative_volume_60d",
    "volume_zscore_20d",
    "dollar_volume_zscore_20d",
    "adv5_to_adv20",
    "adv20_to_adv60",
    # Cross-sectional rank features
    "return_1d_rank",
    "return_5d_rank",
    "return_20d_rank",
    "volatility_20d_rank",
    "relative_volume_20d_rank",
    "log_dollar_volume_rank",
    "overnight_gap_rank",
]


def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found: {path}")

    data = pd.read_parquet(path)
    validate_raw_data(data)
    return data


def load_base_split_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Base split dataset not found: {path}\n"
            "Run this first:\n"
            "python -m js_market_prediction.data.build_splits"
        )

    data = pd.read_parquet(path)
    validate_base_split_data(data)
    return data


def validate_raw_data(data: pd.DataFrame) -> None:
    required_columns = {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Raw data is missing columns: {sorted(missing_columns)}")

    if data.empty:
        raise ValueError("Raw data is empty.")

    if data[list(required_columns)].isna().any().any():
        raise ValueError("Raw data has missing values in required columns.")

    if "SPY" not in set(data["symbol"]):
        raise ValueError("Raw data must include SPY for market-relative features.")


def validate_base_split_data(data: pd.DataFrame) -> None:
    required_columns = {
        "date",
        "symbol",
        "split",
        "resp_1d",
        "target_1d",
        "weight_equal",
        "weight_liquidity",
        *ORIGINAL_FEATURE_COLUMNS,
    }

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"Base split dataset is missing columns: {sorted(missing_columns)}")

    if data.empty:
        raise ValueError("Base split dataset is empty.")

    if data[["date", "symbol"]].duplicated().any():
        raise ValueError("Base split dataset has duplicate date-symbol rows.")


def pct_change_by_symbol(data: pd.DataFrame, column: str, periods: int) -> pd.Series:
    return data.groupby("symbol")[column].transform(
        lambda series: series.pct_change(periods=periods, fill_method=None)
    )


def rolling_mean_by_symbol(data: pd.DataFrame, column: str, window: int) -> pd.Series:
    return data.groupby("symbol")[column].transform(
        lambda series: series.rolling(window=window, min_periods=window).mean()
    )


def rolling_std_by_symbol(data: pd.DataFrame, column: str, window: int) -> pd.Series:
    return data.groupby("symbol")[column].transform(
        lambda series: series.rolling(window=window, min_periods=window).std()
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def build_spy_feature_frame(raw_data: pd.DataFrame) -> pd.DataFrame:
    spy = raw_data[raw_data["symbol"] == "SPY"].copy()
    spy = spy.sort_values("date")

    spy["spy_return_1d"] = spy["close"].pct_change(periods=1, fill_method=None)
    spy["spy_return_5d"] = spy["close"].pct_change(periods=5, fill_method=None)
    spy["spy_return_20d"] = spy["close"].pct_change(periods=20, fill_method=None)

    return spy[
        [
            "date",
            "spy_return_1d",
            "spy_return_5d",
            "spy_return_20d",
        ]
    ]


def build_stock_feature_frame(raw_data: pd.DataFrame) -> pd.DataFrame:
    data = raw_data[raw_data["symbol"] != "SPY"].copy()
    data = data.sort_values(["symbol", "date"]).reset_index(drop=True)

    data["dollar_volume"] = data["close"] * data["volume"]

    data["return_1d"] = pct_change_by_symbol(data, "close", periods=1)
    data["return_3d"] = pct_change_by_symbol(data, "close", periods=3)
    data["return_5d"] = pct_change_by_symbol(data, "close", periods=5)
    data["return_10d"] = pct_change_by_symbol(data, "close", periods=10)
    data["return_20d"] = pct_change_by_symbol(data, "close", periods=20)
    data["return_60d"] = pct_change_by_symbol(data, "close", periods=60)

    data["rolling_mean_return_5d"] = rolling_mean_by_symbol(
        data,
        "return_1d",
        window=5,
    )
    data["rolling_mean_return_20d"] = rolling_mean_by_symbol(
        data,
        "return_1d",
        window=20,
    )

    data["momentum_5d_minus_20d"] = data["return_5d"] - data["return_20d"]
    data["momentum_20d_minus_60d"] = data["return_20d"] - data["return_60d"]

    data["volatility_5d"] = rolling_std_by_symbol(data, "return_1d", window=5)
    data["volatility_20d"] = rolling_std_by_symbol(data, "return_1d", window=20)
    data["volatility_60d"] = rolling_std_by_symbol(data, "return_1d", window=60)

    data["volatility_ratio_5d_20d"] = safe_divide(
        data["volatility_5d"],
        data["volatility_20d"],
    )
    data["volatility_ratio_20d_60d"] = safe_divide(
        data["volatility_20d"],
        data["volatility_60d"],
    )

    data["intraday_range"] = safe_divide(data["high"] - data["low"], data["close"])
    data["range_5d_mean"] = rolling_mean_by_symbol(
        data,
        "intraday_range",
        window=5,
    )
    data["range_20d_mean"] = rolling_mean_by_symbol(
        data,
        "intraday_range",
        window=20,
    )

    previous_close = data.groupby("symbol")["close"].shift(1)
    data["overnight_gap"] = safe_divide(data["open"], previous_close) - 1

    data["log_volume"] = np.log1p(data["volume"])
    data["log_dollar_volume"] = np.log1p(data["dollar_volume"])

    volume_mean_5d = rolling_mean_by_symbol(data, "volume", window=5)
    volume_mean_20d = rolling_mean_by_symbol(data, "volume", window=20)
    volume_mean_60d = rolling_mean_by_symbol(data, "volume", window=60)
    volume_std_20d = rolling_std_by_symbol(data, "volume", window=20)

    dollar_volume_mean_20d = rolling_mean_by_symbol(data, "dollar_volume", window=20)
    dollar_volume_std_20d = rolling_std_by_symbol(data, "dollar_volume", window=20)

    data["relative_volume_5d"] = safe_divide(data["volume"], volume_mean_5d)
    data["relative_volume_20d"] = safe_divide(data["volume"], volume_mean_20d)
    data["relative_volume_60d"] = safe_divide(data["volume"], volume_mean_60d)

    data["volume_zscore_20d"] = safe_divide(
        data["volume"] - volume_mean_20d,
        volume_std_20d,
    )
    data["dollar_volume_zscore_20d"] = safe_divide(
        data["dollar_volume"] - dollar_volume_mean_20d,
        dollar_volume_std_20d,
    )

    adv5 = rolling_mean_by_symbol(data, "dollar_volume", window=5)
    adv20 = rolling_mean_by_symbol(data, "dollar_volume", window=20)
    adv60 = rolling_mean_by_symbol(data, "dollar_volume", window=60)

    data["adv5_to_adv20"] = safe_divide(adv5, adv20)
    data["adv20_to_adv60"] = safe_divide(adv20, adv60)

    return data


def add_market_relative_features(
    stock_features: pd.DataFrame,
    spy_features: pd.DataFrame,
) -> pd.DataFrame:
    data = stock_features.merge(spy_features, on="date", how="left", validate="many_to_one")

    data["return_1d_minus_spy"] = data["return_1d"] - data["spy_return_1d"]
    data["return_5d_minus_spy"] = data["return_5d"] - data["spy_return_5d"]
    data["return_20d_minus_spy"] = data["return_20d"] - data["spy_return_20d"]

    return data


def add_cross_sectional_rank_features(data: pd.DataFrame) -> pd.DataFrame:
    rank_source_columns = [
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "relative_volume_20d",
        "log_dollar_volume",
        "overnight_gap",
    ]

    for column in rank_source_columns:
        data[f"{column}_rank"] = data.groupby("date")[column].rank(
            method="average",
            pct=True,
        )

    return data


def build_expanded_feature_frame(raw_data: pd.DataFrame) -> pd.DataFrame:
    spy_features = build_spy_feature_frame(raw_data)
    stock_features = build_stock_feature_frame(raw_data)

    features = add_market_relative_features(stock_features, spy_features)
    features = add_cross_sectional_rank_features(features)

    feature_columns = ["date", "symbol", *EXPANDED_FEATURE_COLUMNS]

    return features[feature_columns]


def merge_expanded_features(
    base_data: pd.DataFrame,
    expanded_features: pd.DataFrame,
) -> pd.DataFrame:
    merged = base_data.merge(
        expanded_features,
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )

    return merged


def clean_and_validate_expanded_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    data[EXPANDED_FEATURE_COLUMNS] = data[EXPANDED_FEATURE_COLUMNS].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    missing_before_drop = data[EXPANDED_FEATURE_COLUMNS].isna().sum()
    missing_before_drop = missing_before_drop[missing_before_drop > 0]

    print("\nMissing expanded-feature values before dropping rows:")
    if missing_before_drop.empty:
        print("None")
    else:
        print(missing_before_drop.to_string())

    row_count_before = len(data)
    data = data.dropna(subset=EXPANDED_FEATURE_COLUMNS).copy()
    row_count_after = len(data)

    print(f"\nRows before expanded-feature drop: {row_count_before:,}")
    print(f"Rows after expanded-feature drop:  {row_count_after:,}")
    print(f"Rows dropped:                       {row_count_before - row_count_after:,}")

    if data.empty:
        raise ValueError("Expanded dataset is empty after dropping missing features.")

    if data[EXPANDED_FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Expanded dataset still has missing expanded-feature values.")

    if not np.isfinite(data[EXPANDED_FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Expanded dataset contains non-finite expanded-feature values.")

    if data[["date", "symbol"]].duplicated().any():
        raise ValueError("Expanded dataset has duplicate date-symbol rows.")

    return data


def save_expanded_data(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(path, index=False)


def print_summary(data: pd.DataFrame) -> None:
    all_feature_columns = ORIGINAL_FEATURE_COLUMNS + EXPANDED_FEATURE_COLUMNS

    print("\nExpanded feature dataset summary:")
    print(f"Rows:                 {len(data):,}")
    print(f"Columns:              {len(data.columns):,}")
    print(f"Original features:    {len(ORIGINAL_FEATURE_COLUMNS):,}")
    print(f"Expanded features:    {len(EXPANDED_FEATURE_COLUMNS):,}")
    print(f"Total model features: {len(all_feature_columns):,}")
    print(f"First date:           {data['date'].min()}")
    print(f"Last date:            {data['date'].max()}")

    print("\nRows by split:")
    print(data["split"].value_counts().sort_index().to_string())

    print("\nDate range by split:")
    split_summary = data.groupby("split")["date"].agg(["min", "max", "nunique"])
    print(split_summary.to_string())


def main() -> None:
    raw_data = load_raw_data(RAW_DATA_PATH)
    base_split_data = load_base_split_data(BASE_SPLIT_DATA_PATH)

    expanded_features = build_expanded_feature_frame(raw_data)
    expanded_data = merge_expanded_features(base_split_data, expanded_features)
    expanded_data = clean_and_validate_expanded_data(expanded_data)

    save_expanded_data(expanded_data, EXPANDED_OUTPUT_PATH)
    print_summary(expanded_data)

    print(f"\nSaved expanded feature dataset to: {EXPANDED_OUTPUT_PATH}")
    print("\nExpanded feature build completed successfully.")


if __name__ == "__main__":
    main()