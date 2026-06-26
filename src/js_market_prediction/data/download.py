import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "universe.json"

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "daily_prices_2016_2025.parquet"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "download_manifest.json"
)


def load_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Load the market-universe configuration from a JSON file."""
    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_config(config: dict[str, Any]) -> None:
    """Validate the fields needed for the initial market-data download."""
    required_keys = {
        "name",
        "source",
        "start_date",
        "end_date",
        "interval",
        "auto_adjust",
        "benchmark",
        "tickers",
    }

    missing_keys = required_keys - config.keys()

    if missing_keys:
        missing_list = ", ".join(sorted(missing_keys))
        raise ValueError(
            f"Configuration is missing required keys: {missing_list}"
        )

    tickers = config["tickers"]
    benchmark = config["benchmark"]

    if not isinstance(tickers, list):
        raise TypeError("'tickers' must be a list.")

    if len(tickers) != 30:
        raise ValueError(
            f"Expected 30 stock tickers, but found {len(tickers)}."
        )

    if len(set(tickers)) != len(tickers):
        raise ValueError("The stock ticker list contains duplicates.")

    if not isinstance(benchmark, str) or not benchmark:
        raise TypeError("'benchmark' must be a nonempty string.")

    if benchmark in tickers:
        raise ValueError(
            "The benchmark must be separate from the stock ticker list."
        )


def download_raw_data(config: dict[str, Any]) -> pd.DataFrame:
    """Download raw daily market data for the configured symbols."""
    tickers = config["tickers"]
    benchmark = config["benchmark"]
    all_symbols = [*tickers, benchmark]

    print(
        f"Downloading {len(all_symbols)} symbols "
        f"from {config['start_date']} to {config['end_date']}..."
    )

    data = yf.download(
        tickers=all_symbols,
        start=config["start_date"],
        end=config["end_date"],
        interval=config["interval"],
        auto_adjust=config["auto_adjust"],
        group_by="ticker",
        actions=False,
        progress=True,
        threads=True,
        multi_level_index=True,
    )

    if data is None or data.empty:
        raise RuntimeError("The market-data download returned no rows.")

    return data


def validate_raw_data(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Validate the structure and coverage of the raw yfinance download."""
    if data.empty:
        raise ValueError("The downloaded dataset is empty.")

    if not isinstance(data.columns, pd.MultiIndex):
        raise TypeError(
            "Expected downloaded data to have MultiIndex columns."
        )

    if data.columns.nlevels != 2:
        raise ValueError(
            f"Expected 2 column levels, but found {data.columns.nlevels}."
        )

    expected_symbols = set(
        [*config["tickers"], config["benchmark"]]
    )
    downloaded_symbols = set(
        data.columns.get_level_values("Ticker")
    )

    missing_symbols = expected_symbols - downloaded_symbols
    unexpected_symbols = downloaded_symbols - expected_symbols

    if missing_symbols:
        missing_list = ", ".join(sorted(missing_symbols))
        raise ValueError(
            f"Download is missing requested symbols: {missing_list}"
        )

    if unexpected_symbols:
        unexpected_list = ", ".join(sorted(unexpected_symbols))
        raise ValueError(
            f"Download contains unexpected symbols: {unexpected_list}"
        )

    expected_fields = {
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    }
    downloaded_fields = set(
        data.columns.get_level_values("Price")
    )

    missing_fields = expected_fields - downloaded_fields

    if missing_fields:
        missing_list = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"Download is missing expected price fields: {missing_list}"
        )

    if data.index.has_duplicates:
        duplicate_count = int(data.index.duplicated().sum())
        raise ValueError(
            f"Downloaded data contains {duplicate_count} duplicate dates."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "Downloaded dates are not sorted in increasing order."
        )

    completely_missing_symbols = []

    for symbol in sorted(expected_symbols):
        close_values = data[(symbol, "Close")]

        if close_values.notna().sum() == 0:
            completely_missing_symbols.append(symbol)

    if completely_missing_symbols:
        missing_list = ", ".join(completely_missing_symbols)
        raise ValueError(
            f"These symbols contain no valid closing prices: {missing_list}"
        )
    

def reshape_to_long(
    data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Convert wide yfinance data into one row per date and symbol."""
    all_symbols = [
        *config["tickers"],
        config["benchmark"],
    ]

    symbol_frames = []

    for symbol in all_symbols:
        symbol_data = data[symbol].copy()

        symbol_data.columns = [
            column.lower()
            for column in symbol_data.columns
        ]

        symbol_data = (
            symbol_data
            .reset_index()
            .rename(columns={"Date": "date"})
        )

        symbol_data.insert(
            loc=1,
            column="symbol",
            value=symbol,
        )

        symbol_frames.append(symbol_data)

    long_data = pd.concat(
        symbol_frames,
        ignore_index=True,
    )

    price_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    long_data = long_data[
        ["date", "symbol", *price_columns]
    ]

    long_data = (
        long_data
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    
    # aligns the date column to a consistent datetime format for validation and saving
    long_data["date"] = long_data["date"].astype(
        "datetime64[ms]"
    )

    return long_data


def validate_long_data(
    data: pd.DataFrame,
    raw_data: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Validate the canonical long-format market dataset."""
    expected_columns = [
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    if data.columns.tolist() != expected_columns:
        raise ValueError(
            "Long data has unexpected columns. "
            f"Expected {expected_columns}, "
            f"but found {data.columns.tolist()}."
        )

    if not pd.api.types.is_datetime64_any_dtype(data["date"]):
        raise TypeError("'date' must use a pandas datetime data type.")

    expected_symbols = set(
        [*config["tickers"], config["benchmark"]]
    )
    downloaded_symbols = set(data["symbol"].unique())

    missing_symbols = expected_symbols - downloaded_symbols
    unexpected_symbols = downloaded_symbols - expected_symbols

    if missing_symbols:
        missing_list = ", ".join(sorted(missing_symbols))
        raise ValueError(
            f"Long data is missing symbols: {missing_list}"
        )

    if unexpected_symbols:
        unexpected_list = ", ".join(sorted(unexpected_symbols))
        raise ValueError(
            f"Long data contains unexpected symbols: {unexpected_list}"
        )

    duplicate_mask = data.duplicated(
        subset=["date", "symbol"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            "Long data contains "
            f"{duplicate_count} rows with duplicate date-symbol keys."
        )

    expected_row_count = (
        len(raw_data.index) * len(expected_symbols)
    )

    if len(data) != expected_row_count:
        raise ValueError(
            f"Expected {expected_row_count} long-format rows, "
            f"but found {len(data)}."
        )

    expected_dates = pd.DatetimeIndex(
        raw_data.index
    ).sort_values()

    downloaded_dates = pd.DatetimeIndex(
        data["date"].unique()
    ).sort_values()

    if not expected_dates.equals(downloaded_dates):
        raise ValueError(
            "Long data does not contain the same dates as the raw data."
        )

    rows_per_symbol = data.groupby("symbol")["date"].nunique()

    incomplete_symbols = rows_per_symbol[
        rows_per_symbol != len(expected_dates)
    ]

    if not incomplete_symbols.empty:
        details = ", ".join(
            f"{symbol}: {count}"
            for symbol, count in incomplete_symbols.items()
        )
        raise ValueError(
            "Some symbols do not contain the expected number of dates: "
            f"{details}"
        )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    non_numeric_columns = [
        column
        for column in numeric_columns
        if not pd.api.types.is_numeric_dtype(data[column])
    ]

    if non_numeric_columns:
        column_list = ", ".join(non_numeric_columns)
        raise TypeError(
            f"Expected numeric market columns: {column_list}"
        )

    if data["volume"].dropna().lt(0).any():
        raise ValueError("Volume contains negative values.")

    price_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    if data[price_columns].lt(0).any().any():
        raise ValueError("Price columns contain negative values.")

    complete_price_rows = data.dropna(
        subset=price_columns
    )

    price_tolerance = 1e-10

    invalid_price_ranges = (
        (
            complete_price_rows["low"]
            - complete_price_rows["high"]
            > price_tolerance
        )
        | (
            complete_price_rows["open"]
            - complete_price_rows["high"]
            > price_tolerance
        )
        | (
            complete_price_rows["low"]
            - complete_price_rows["open"]
            > price_tolerance
        )
        | (
            complete_price_rows["close"]
            - complete_price_rows["high"]
            > price_tolerance
        )
        | (
            complete_price_rows["low"]
            - complete_price_rows["close"]
            > price_tolerance
        )
    )

    if invalid_price_ranges.any():
        invalid_rows = complete_price_rows.loc[
            invalid_price_ranges,
            [
                "date",
                "symbol",
                "open",
                "high",
                "low",
                "close",
            ],
        ]

        print()
        print("Rows with materially inconsistent OHLC ranges:")
        print(invalid_rows.to_string(index=False))

        raise ValueError(
            "Found "
            f"{len(invalid_rows)} rows with materially inconsistent "
            "OHLC ranges."
        )

    sorted_keys = (
        data[["date", "symbol"]]
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )

    actual_keys = (
        data[["date", "symbol"]]
        .reset_index(drop=True)
    )

    if not actual_keys.equals(sorted_keys):
        raise ValueError(
            "Long data is not sorted by date and symbol."
        )


def save_long_data(
    data: pd.DataFrame,
    output_path: Path = RAW_DATA_PATH,
) -> None:
    """Save the validated long-format dataset as a Parquet file."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    if not output_path.exists():
        raise RuntimeError(
            f"Parquet file was not created: {output_path}"
        )

    if output_path.stat().st_size == 0:
        raise RuntimeError(
            f"Parquet file is empty: {output_path}"
        )


def validate_saved_data(
    expected_data: pd.DataFrame,
    output_path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Read the saved Parquet file and compare it with the source table."""
    saved_data = pd.read_parquet(
        output_path,
        engine="pyarrow",
    )

    try:
        pd.testing.assert_frame_equal(
            saved_data,
            expected_data,
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise ValueError(
            "The saved Parquet file does not exactly match "
            "the validated source data."
        ) from error

    return saved_data


def build_manifest(
    config: dict[str, Any],
    data: pd.DataFrame,
    output_path: Path = RAW_DATA_PATH,
) -> dict[str, Any]:
    """Create metadata describing the downloaded market dataset."""
    symbols = [
        *config["tickers"],
        config["benchmark"],
    ]

    missing_values = {
        column: int(count)
        for column, count in data.isna().sum().items()
    }

    return {
        "dataset_name": config["name"],
        "description": config["description"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": config["source"],
        "source_library": "yfinance",
        "source_library_version": yf.__version__,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "requested_start_date": config["start_date"],
        "requested_end_date_exclusive": config["end_date"],
        "actual_first_date": data["date"].min().date().isoformat(),
        "actual_last_date": data["date"].max().date().isoformat(),
        "interval": config["interval"],
        "auto_adjust": config["auto_adjust"],
        "benchmark": config["benchmark"],
        "stock_tickers": config["tickers"],
        "all_symbols": symbols,
        "stock_count": len(config["tickers"]),
        "total_symbol_count": len(symbols),
        "row_count": len(data),
        "column_count": len(data.columns),
        "columns": data.columns.tolist(),
        "missing_values_by_column": missing_values,
        "duplicate_date_symbol_rows": int(
            data.duplicated(["date", "symbol"]).sum()
        ),
        "output_file": str(
            output_path.relative_to(PROJECT_ROOT)
        ),
        "output_file_size_bytes": output_path.stat().st_size,
    }


def save_manifest(
    manifest: dict[str, Any],
    output_path: Path = MANIFEST_PATH,
) -> None:
    """Save the dataset manifest as formatted JSON."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

        file.write("\n")


def validate_saved_manifest(
    expected_manifest: dict[str, Any],
    output_path: Path = MANIFEST_PATH,
) -> dict[str, Any]:
    """Read the saved manifest and compare it with its source."""
    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        saved_manifest = json.load(file)

    if saved_manifest != expected_manifest:
        raise ValueError(
            "The saved manifest does not match "
            "the generated manifest."
        )

    return saved_manifest


def main() -> None:
    """Load the configuration, download the data, and inspect the result."""
    config = load_config()
    validate_config(config)

    tickers = config["tickers"]
    benchmark = config["benchmark"]
    all_symbols = [*tickers, benchmark]

    print(f"Configuration: {config['name']}")
    print(f"Stock tickers: {len(tickers)}")
    print(f"Benchmark: {benchmark}")
    print(f"Total symbols to download: {len(all_symbols)}")
    print(f"Date range: {config['start_date']} to {config['end_date']}")
    print("Configuration validation passed.")

    raw_data = download_raw_data(config)
    validate_raw_data(raw_data, config)

    long_data = reshape_to_long(raw_data, config)
    validate_long_data(long_data, raw_data, config)
    
    save_long_data(long_data)
    saved_data = validate_saved_data(long_data)
    
    manifest = build_manifest(
        config,
        saved_data,
    )

    save_manifest(manifest)
    saved_manifest = validate_saved_manifest(manifest)

    print()
    print("Download completed.")
    print(f"Raw shape: {raw_data.shape}")
    print(f"First date: {raw_data.index.min()}")
    print(f"Last date: {raw_data.index.max()}")
    print(f"Column levels: {raw_data.columns.names}")
    print(f"Downloaded symbols: {raw_data.columns.get_level_values(0).nunique()}")
    print("Raw-data validation passed.")

    print()
    print("Reshaping completed.")
    print(f"Long shape: {long_data.shape}")
    print(f"Long columns: {long_data.columns.tolist()}")
    print()
    print(long_data.head(10).to_string(index=False))
    print("Long-data validation passed.")
    
    file_size_mb = RAW_DATA_PATH.stat().st_size / (1024 ** 2)

    print()
    print("Parquet save completed.")
    print(
        "Saved file: "
        f"{RAW_DATA_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(f"Saved shape: {saved_data.shape}")
    print(f"File size: {file_size_mb:.2f} MB")
    print("Parquet round-trip validation passed.")
    
    print()
    print("Manifest save completed.")
    print(
        "Saved manifest: "
        f"{MANIFEST_PATH.relative_to(PROJECT_ROOT)}"
    )
    print(
        "Manifest symbols: "
        f"{saved_manifest['total_symbol_count']}"
    )
    print(
        "Manifest rows: "
        f"{saved_manifest['row_count']}"
    )
    print("Manifest round-trip validation passed.")


if __name__ == "__main__":
    main()