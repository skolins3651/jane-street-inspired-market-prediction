from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "daily_prices_2016_2025.parquet"
)

MODELING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset_1d.parquet"
)

FEATURE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset_with_features_1d.parquet"
)

FEATURE_COLUMNS = [
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


def load_raw_data(
    input_path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Load the raw daily market data."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {input_path}"
        )

    data = pd.read_parquet(input_path)

    return data


def load_modeling_data(
    input_path: Path = MODELING_DATA_PATH,
) -> pd.DataFrame:
    """Load the one-day modeling dataset from Phase 3.2."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Modeling data file not found: {input_path}"
        )

    data = pd.read_parquet(input_path)

    return data


def validate_input_data(
    raw_data: pd.DataFrame,
    modeling_data: pd.DataFrame,
) -> None:
    """Validate that the raw and modeling datasets are compatible."""
    raw_expected_columns = {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    modeling_expected_columns = {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "stock_return_1d_forward",
        "spy_return_1d_forward",
        "resp_1d",
        "target_1d",
        "weight_equal",
        "dollar_volume",
        "adv20",
        "weight_liquidity",
    }

    raw_missing_columns = (
        raw_expected_columns
        - set(raw_data.columns)
    )

    modeling_missing_columns = (
        modeling_expected_columns
        - set(modeling_data.columns)
    )

    if raw_missing_columns:
        raise ValueError(
            "Raw data is missing expected columns: "
            f"{sorted(raw_missing_columns)}"
        )

    if modeling_missing_columns:
        raise ValueError(
            "Modeling data is missing expected columns: "
            f"{sorted(modeling_missing_columns)}"
        )

    if raw_data.empty:
        raise ValueError("Raw data is empty.")

    if modeling_data.empty:
        raise ValueError("Modeling data is empty.")

    if "SPY" not in set(raw_data["symbol"]):
        raise ValueError("SPY is missing from the raw data.")

    if "SPY" in set(modeling_data["symbol"]):
        raise ValueError(
            "SPY should not appear in the modeling dataset."
        )

    raw_duplicate_count = raw_data.duplicated(
        subset=["date", "symbol"]
    ).sum()

    modeling_duplicate_count = modeling_data.duplicated(
        subset=["date", "symbol"]
    ).sum()

    if raw_duplicate_count > 0:
        raise ValueError(
            f"Raw data has {raw_duplicate_count:,} "
            "duplicate date-symbol rows."
        )

    if modeling_duplicate_count > 0:
        raise ValueError(
            f"Modeling data has {modeling_duplicate_count:,} "
            "duplicate date-symbol rows."
        )

    raw_stock_data = raw_data.loc[
        raw_data["symbol"] != "SPY"
    ].copy()

    raw_keys = pd.MultiIndex.from_frame(
        raw_stock_data[["date", "symbol"]]
    )

    modeling_keys = pd.MultiIndex.from_frame(
        modeling_data[["date", "symbol"]]
    )

    missing_modeling_keys = (
        modeling_keys.difference(raw_keys)
    )

    if len(missing_modeling_keys) > 0:
        raise ValueError(
            "At least one modeling date-symbol key is "
            "missing from the raw stock data."
        )


def build_stock_features(
    raw_data: pd.DataFrame,
) -> pd.DataFrame:
    """Build leakage-safe stock features using data through date t."""
    stock_data = raw_data.loc[
        raw_data["symbol"] != "SPY"
    ].copy()

    stock_data = stock_data.sort_values(
        ["symbol", "date"]
    ).reset_index(drop=True)

    grouped = stock_data.groupby("symbol")

    previous_close = grouped["close"].shift(1)

    stock_data["return_1d"] = (
        stock_data["close"]
        / previous_close
        - 1
    )

    stock_data["return_5d"] = (
        stock_data["close"]
        / grouped["close"].shift(5)
        - 1
    )

    stock_data["return_20d"] = (
        stock_data["close"]
        / grouped["close"].shift(20)
        - 1
    )

    stock_data["volatility_20d"] = (
        stock_data
        .groupby("symbol")["return_1d"]
        .transform(
            lambda series: series.rolling(
                window=20,
                min_periods=20,
            ).std()
        )
    )

    stock_data["intraday_range"] = (
        stock_data["high"]
        - stock_data["low"]
    ) / stock_data["close"]

    stock_data["overnight_gap"] = (
        stock_data["open"]
        / previous_close
        - 1
    )

    stock_data["feature_dollar_volume"] = (
        stock_data["close"]
        * stock_data["volume"]
    )

    stock_data["log_volume"] = np.log1p(
        stock_data["volume"]
    )

    stock_data["log_dollar_volume"] = np.log1p(
        stock_data["feature_dollar_volume"]
    )

    average_volume_20d = (
        stock_data
        .groupby("symbol")["volume"]
        .transform(
            lambda series: series.rolling(
                window=20,
                min_periods=20,
            ).mean()
        )
    )

    stock_data["relative_volume_20d"] = (
        stock_data["volume"]
        / average_volume_20d
    )

    feature_data = stock_data[
        ["date", "symbol", *FEATURE_COLUMNS]
    ].copy()

    return feature_data


def merge_features(
    modeling_data: pd.DataFrame,
    feature_data: pd.DataFrame,
) -> pd.DataFrame:
    """Merge stock features onto the one-day modeling dataset."""
    overlapping_columns = (
        set(modeling_data.columns)
        & set(FEATURE_COLUMNS)
    )

    if overlapping_columns:
        raise ValueError(
            "Modeling data already contains feature columns: "
            f"{sorted(overlapping_columns)}"
        )

    merged_data = modeling_data.merge(
        feature_data,
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )

    merged_data = merged_data.sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)

    return merged_data


def filter_feature_ready_rows(
    feature_dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Drop rows with incomplete feature values."""
    clean_data = feature_dataset.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    clean_data = clean_data.reset_index(drop=True)

    return clean_data


def validate_feature_dataset(clean_data: pd.DataFrame) -> None:
    """Validate the final feature-ready dataset."""
    expected_columns = {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "stock_return_1d_forward",
        "spy_return_1d_forward",
        "resp_1d",
        "target_1d",
        "weight_equal",
        "dollar_volume",
        "adv20",
        "weight_liquidity",
        *FEATURE_COLUMNS,
    }

    actual_columns = set(clean_data.columns)

    missing_columns = expected_columns - actual_columns
    extra_columns = actual_columns - expected_columns

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {sorted(missing_columns)}"
        )

    if extra_columns:
        raise ValueError(
            f"Unexpected columns: {sorted(extra_columns)}"
        )

    if clean_data.empty:
        raise ValueError("Feature dataset is empty.")

    if "SPY" in set(clean_data["symbol"]):
        raise ValueError("SPY should not appear in the feature dataset.")

    duplicate_rows = clean_data.duplicated(
        subset=["date", "symbol"]
    ).sum()

    if duplicate_rows > 0:
        raise ValueError(
            f"Found {duplicate_rows:,} duplicate date-symbol rows."
        )

    expected_symbols = 30
    unique_symbols = clean_data["symbol"].nunique()

    if unique_symbols != expected_symbols:
        raise ValueError(
            f"Expected {expected_symbols} symbols, "
            f"found {unique_symbols}."
        )

    rows_per_date = clean_data.groupby("date").size()

    if not (rows_per_date == expected_symbols).all():
        raise ValueError(
            "At least one date does not contain all "
            f"{expected_symbols} tradable symbols."
        )

    missing_values = clean_data.isna().sum()

    if missing_values.sum() > 0:
        raise ValueError(
            "Feature dataset contains missing values:\n"
            f"{missing_values[missing_values > 0]}"
        )

    valid_targets = set(clean_data["target_1d"].unique())

    if valid_targets != {0, 1}:
        raise ValueError(
            f"target_1d should contain only 0 and 1, "
            f"found {sorted(valid_targets)}."
        )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "stock_return_1d_forward",
        "spy_return_1d_forward",
        "resp_1d",
        "weight_equal",
        "dollar_volume",
        "adv20",
        "weight_liquidity",
        *FEATURE_COLUMNS,
    ]

    numeric_values = clean_data[numeric_columns].to_numpy()

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "At least one numeric column contains a non-finite value."
        )

    if not (clean_data["weight_equal"] == 1.0).all():
        raise ValueError("weight_equal should always equal 1.0.")

    if (clean_data["weight_liquidity"] <= 0).any():
        raise ValueError("weight_liquidity should always be positive.")

    liquidity_weight_mean_by_date = (
        clean_data
        .groupby("date")["weight_liquidity"]
        .mean()
    )

    if not np.allclose(
        liquidity_weight_mean_by_date,
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Liquidity weights do not average to 1.0 on every date."
        )


def save_feature_dataset(
    clean_data: pd.DataFrame,
    output_path: Path = FEATURE_DATA_PATH,
) -> None:
    """Save the feature-ready dataset to Parquet."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean_data.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    if not output_path.exists():
        raise FileNotFoundError(
            f"Feature data file was not created: {output_path}"
        )

    if output_path.stat().st_size == 0:
        raise ValueError(
            f"Feature data file is empty: {output_path}"
        )


def validate_saved_feature_dataset(
    expected_data: pd.DataFrame,
    output_path: Path = FEATURE_DATA_PATH,
) -> None:
    """Confirm the saved Parquet file matches the expected data."""
    saved_data = pd.read_parquet(
        output_path,
        engine="pyarrow",
    )

    pd.testing.assert_frame_equal(
        saved_data,
        expected_data,
        check_dtype=True,
        check_exact=True,
    )


def print_input_summary(
    raw_data: pd.DataFrame,
    modeling_data: pd.DataFrame,
) -> None:
    """Print a summary of the feature-building inputs."""
    raw_symbols = sorted(
        raw_data["symbol"].unique()
    )

    modeling_symbols = sorted(
        modeling_data["symbol"].unique()
    )

    print("Feature-building inputs loaded successfully.")

    print()
    print("Raw data:")
    print(f"Path: {RAW_DATA_PATH}")
    print(f"Rows: {len(raw_data):,}")
    print(f"Columns: {len(raw_data.columns)}")
    print(f"First date: {raw_data['date'].min().date()}")
    print(f"Last date: {raw_data['date'].max().date()}")
    print(f"Symbols: {len(raw_symbols)}")
    print(f"SPY present: {'SPY' in raw_symbols}")

    print()
    print("Modeling data:")
    print(f"Path: {MODELING_DATA_PATH}")
    print(f"Rows: {len(modeling_data):,}")
    print(f"Columns: {len(modeling_data.columns)}")
    print(f"First date: {modeling_data['date'].min().date()}")
    print(f"Last date: {modeling_data['date'].max().date()}")
    print(f"Symbols: {len(modeling_symbols)}")
    print(f"SPY present: {'SPY' in modeling_symbols}")


def print_feature_summary(
    feature_data: pd.DataFrame,
    feature_dataset: pd.DataFrame,
    clean_data: pd.DataFrame,
) -> None:
    """Print sanity checks for the engineered feature dataset."""
    print()
    print("Features built successfully.")
    print(f"Raw feature rows: {len(feature_data):,}")
    print(f"Rows after merge: {len(feature_dataset):,}")
    print(f"Feature-ready rows: {len(clean_data):,}")
    print(f"Columns: {len(clean_data.columns)}")

    print()
    print("Feature columns:")
    for column in FEATURE_COLUMNS:
        print(f"- {column}")

    print()
    print("Missing feature values before dropping incomplete rows:")
    print(feature_dataset[FEATURE_COLUMNS].isna().sum())

    print()
    print("Missing feature values after dropping incomplete rows:")
    print(clean_data[FEATURE_COLUMNS].isna().sum())

    print()
    print(f"First feature-ready date: {clean_data['date'].min().date()}")
    print(f"Last feature-ready date: {clean_data['date'].max().date()}")

    print()
    print("Target distribution after feature filtering:")
    print(
        clean_data["target_1d"]
        .value_counts(normalize=True)
        .sort_index()
    )

    print()
    print("Feature summary statistics:")
    print(
        clean_data[FEATURE_COLUMNS]
        .describe(
            percentiles=[
                0.01,
                0.05,
                0.50,
                0.95,
                0.99,
            ]
        )
        .T
    )

    print()
    print("First five feature-ready rows:")
    display_columns = [
        "date",
        "symbol",
        "resp_1d",
        "target_1d",
        "weight_equal",
        "weight_liquidity",
        *FEATURE_COLUMNS,
    ]
    print(clean_data[display_columns].head())


def main() -> None:
    """Build the first feature set for the one-day modeling problem."""
    raw_data = load_raw_data()
    modeling_data = load_modeling_data()

    validate_input_data(
        raw_data=raw_data,
        modeling_data=modeling_data,
    )

    print_input_summary(
        raw_data=raw_data,
        modeling_data=modeling_data,
    )

    feature_data = build_stock_features(raw_data)

    feature_dataset = merge_features(
        modeling_data=modeling_data,
        feature_data=feature_data,
    )

    clean_data = filter_feature_ready_rows(
        feature_dataset
    )

    validate_feature_dataset(clean_data)

    print_feature_summary(
        feature_data=feature_data,
        feature_dataset=feature_dataset,
        clean_data=clean_data,
    )

    save_feature_dataset(clean_data)
    validate_saved_feature_dataset(clean_data)

    print()
    print("Feature dataset saved successfully.")
    print(f"Output path: {FEATURE_DATA_PATH}")
    print(
        f"Output file size: "
        f"{FEATURE_DATA_PATH.stat().st_size / 1_000_000:.2f} MB"
    )


if __name__ == "__main__":
    main()