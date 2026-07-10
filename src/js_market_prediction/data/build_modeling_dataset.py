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

PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset_1d.parquet"
)


def load_raw_data(input_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw daily market data."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {input_path}"
        )

    data = pd.read_parquet(input_path)

    return data


def validate_raw_data(data: pd.DataFrame) -> None:
    """Check that the raw data has the expected basic structure."""
    expected_columns = {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    actual_columns = set(data.columns)

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

    if data.empty:
        raise ValueError("Raw data is empty.")

    duplicate_rows = data.duplicated(
        subset=["date", "symbol"]
    ).sum()

    if duplicate_rows > 0:
        raise ValueError(
            f"Found {duplicate_rows:,} duplicate date-symbol rows."
        )

    symbols = sorted(data["symbol"].unique())

    if "SPY" not in symbols:
        raise ValueError("SPY is missing from the raw data.")

    stock_symbols = [
        symbol
        for symbol in symbols
        if symbol != "SPY"
    ]

    if len(stock_symbols) != 30:
        raise ValueError(
            f"Expected 30 tradable stock symbols, "
            f"found {len(stock_symbols)}."
        )


def build_modeling_dataset(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Create one-day response, target, and weight columns."""
    data = raw_data.sort_values(
        ["symbol", "date"]
    ).reset_index(drop=True).copy()

    data["stock_return_1d_forward"] = (
        data.groupby("symbol")["close"]
        .shift(-1)
        / data["close"]
        - 1
    )

    spy_returns = (
        data.loc[
            data["symbol"] == "SPY",
            ["date", "stock_return_1d_forward"],
        ]
        .rename(
            columns={
                "stock_return_1d_forward": "spy_return_1d_forward"
            }
        )
    )

    modeling_data = data.merge(
        spy_returns,
        on="date",
        how="left",
    )

    modeling_data = modeling_data.loc[
        modeling_data["symbol"] != "SPY"
    ].copy()

    modeling_data["resp_1d"] = (
        modeling_data["stock_return_1d_forward"]
        - modeling_data["spy_return_1d_forward"]
    )

    modeling_data["target_1d"] = pd.Series(
        pd.NA,
        index=modeling_data.index,
        dtype="Int64",
    )

    valid_response = modeling_data["resp_1d"].notna()

    modeling_data.loc[
        valid_response,
        "target_1d",
    ] = (
        modeling_data.loc[
            valid_response,
            "resp_1d",
        ]
        > 0
    ).astype("int64")

    modeling_data["weight_equal"] = 1.0

    modeling_data["dollar_volume"] = (
        modeling_data["close"]
        * modeling_data["volume"]
    )

    modeling_data["adv20"] = (
        modeling_data
        .groupby("symbol")["dollar_volume"]
        .transform(
            lambda series: series.rolling(
                window=20,
                min_periods=20,
            ).mean()
        )
    )

    mean_adv20_by_date = (
        modeling_data
        .groupby("date")["adv20"]
        .transform("mean")
    )

    modeling_data["weight_liquidity"] = (
        modeling_data["adv20"]
        / mean_adv20_by_date
    )

    modeling_data = modeling_data.sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)

    return modeling_data


def filter_modeling_ready_rows(
    modeling_data: pd.DataFrame,
) -> pd.DataFrame:
    """Drop rows that cannot be used for the first baseline model."""
    required_columns = [
        "stock_return_1d_forward",
        "spy_return_1d_forward",
        "resp_1d",
        "target_1d",
        "adv20",
        "weight_liquidity",
    ]

    clean_data = modeling_data.dropna(
        subset=required_columns
    ).copy()

    clean_data["target_1d"] = clean_data[
        "target_1d"
    ].astype("int64")

    clean_data = clean_data.reset_index(drop=True)

    return clean_data


def validate_modeling_dataset(clean_data: pd.DataFrame) -> None:
    """Validate the final modeling-ready dataset."""
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
        raise ValueError("Modeling dataset is empty.")

    if "SPY" in set(clean_data["symbol"]):
        raise ValueError("SPY should not appear as a modeling row.")

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
            "Modeling dataset contains missing values:\n"
            f"{missing_values[missing_values > 0]}"
        )

    valid_targets = set(clean_data["target_1d"].unique())

    if valid_targets != {0, 1}:
        raise ValueError(
            f"target_1d should contain only 0 and 1, "
            f"found {sorted(valid_targets)}."
        )

    if not (clean_data["weight_equal"] == 1.0).all():
        raise ValueError("weight_equal should always equal 1.0.")

    if (clean_data["weight_liquidity"] <= 0).any():
        raise ValueError("weight_liquidity should always be positive.")

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
    ]

    numeric_values = clean_data[numeric_columns].to_numpy()

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "At least one numeric modeling column contains "
            "a non-finite value."
        )

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
            "Liquidity weights do not average to 1.0 "
            "on every date."
        )


def save_modeling_dataset(
    clean_data: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_PATH,
) -> None:
    """Save the modeling-ready dataset to Parquet."""
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
            f"Processed data file was not created: {output_path}"
        )

    if output_path.stat().st_size == 0:
        raise ValueError(
            f"Processed data file is empty: {output_path}"
        )


def validate_saved_modeling_dataset(
    expected_data: pd.DataFrame,
    output_path: Path = PROCESSED_DATA_PATH,
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


def print_raw_data_summary(data: pd.DataFrame) -> None:
    """Print a basic summary of the raw input dataset."""
    symbols = sorted(data["symbol"].unique())

    stock_symbols = [
        symbol
        for symbol in symbols
        if symbol != "SPY"
    ]

    print("Raw data loaded successfully.")
    print(f"Input path: {RAW_DATA_PATH}")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print(f"First date: {data['date'].min().date()}")
    print(f"Last date: {data['date'].max().date()}")
    print(f"Total symbols: {len(symbols)}")
    print(f"Tradable stock symbols: {len(stock_symbols)}")
    print(f"Benchmark symbol present: {'SPY' in symbols}")


def print_modeling_data_summary(
    modeling_data: pd.DataFrame,
    clean_data: pd.DataFrame,
) -> None:
    """Print checks for the target and weight construction."""
    modeling_columns = [
        "stock_return_1d_forward",
        "spy_return_1d_forward",
        "resp_1d",
        "target_1d",
        "weight_equal",
        "dollar_volume",
        "adv20",
        "weight_liquidity",
    ]

    print()
    print("Modeling dataset built successfully.")
    print(f"Rows after removing SPY: {len(modeling_data):,}")
    print(f"Modeling-ready rows: {len(clean_data):,}")
    print(f"Columns: {len(clean_data.columns)}")
    print(f"First modeling-ready date: {clean_data['date'].min().date()}")
    print(f"Last modeling-ready date: {clean_data['date'].max().date()}")

    print()
    print("Missing values before dropping incomplete rows:")
    print(modeling_data[modeling_columns].isna().sum())

    print()
    print("Missing values after dropping incomplete rows:")
    print(clean_data[modeling_columns].isna().sum())

    print()
    print("Target distribution:")
    print(
        clean_data["target_1d"]
        .value_counts(normalize=True)
        .sort_index()
    )

    liquidity_weight_mean_by_date = (
        clean_data
        .groupby("date")["weight_liquidity"]
        .mean()
    )

    print()
    print("Liquidity weight mean by date:")
    print(liquidity_weight_mean_by_date.describe())

    print()
    print("First five modeling-ready rows:")
    print(clean_data.head())


def main() -> None:
    """Build the initial one-day modeling dataset."""
    raw_data = load_raw_data()
    validate_raw_data(raw_data)
    print_raw_data_summary(raw_data)

    modeling_data = build_modeling_dataset(raw_data)
    clean_data = filter_modeling_ready_rows(modeling_data)

    validate_modeling_dataset(clean_data)

    print_modeling_data_summary(
        modeling_data=modeling_data,
        clean_data=clean_data,
    )

    save_modeling_dataset(clean_data)
    validate_saved_modeling_dataset(clean_data)

    print()
    print("Processed modeling dataset saved successfully.")
    print(f"Output path: {PROCESSED_DATA_PATH}")
    print(
        f"Output file size: "
        f"{PROCESSED_DATA_PATH.stat().st_size / 1_000_000:.2f} MB"
    )


if __name__ == "__main__":
    main()