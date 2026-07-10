from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset_with_features_1d.parquet"
)

SPLIT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling_dataset_with_features_and_splits_1d.parquet"
)

TRAIN_END_DATE = pd.Timestamp("2021-12-31")
VALIDATION_END_DATE = pd.Timestamp("2023-12-31")


def load_feature_data(
    input_path: Path = FEATURE_DATA_PATH,
) -> pd.DataFrame:
    """Load the feature-ready modeling dataset from Phase 3.3."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature data file not found: {input_path}"
        )

    data = pd.read_parquet(input_path)

    return data


def validate_feature_data(data: pd.DataFrame) -> None:
    """Validate the feature-ready dataset before assigning splits."""
    required_columns = {
        "date",
        "symbol",
        "resp_1d",
        "target_1d",
        "weight_equal",
        "weight_liquidity",
        "return_1d",
        "return_5d",
        "return_20d",
        "volatility_20d",
        "intraday_range",
        "overnight_gap",
        "log_volume",
        "log_dollar_volume",
        "relative_volume_20d",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Feature dataset is empty.")

    if "SPY" in set(data["symbol"]):
        raise ValueError("SPY should not appear in the feature dataset.")

    duplicate_rows = data.duplicated(
        subset=["date", "symbol"]
    ).sum()

    if duplicate_rows > 0:
        raise ValueError(
            f"Found {duplicate_rows:,} duplicate date-symbol rows."
        )

    if data.isna().sum().sum() > 0:
        raise ValueError("Feature dataset contains missing values.")


def assign_chronological_splits(data: pd.DataFrame) -> pd.DataFrame:
    """Assign chronological train, validation, and test labels."""
    split_data = data.sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True).copy()

    split_data["split"] = "test"

    split_data.loc[
        split_data["date"] <= TRAIN_END_DATE,
        "split",
    ] = "train"

    split_data.loc[
        (
            split_data["date"] > TRAIN_END_DATE
        )
        & (
            split_data["date"] <= VALIDATION_END_DATE
        ),
        "split",
    ] = "validation"

    return split_data


def validate_split_data(split_data: pd.DataFrame) -> None:
    """Validate the split-labeled dataset."""
    expected_splits = {
        "train",
        "validation",
        "test",
    }

    actual_splits = set(split_data["split"].unique())

    if actual_splits != expected_splits:
        raise ValueError(
            f"Expected splits {sorted(expected_splits)}, "
            f"found {sorted(actual_splits)}."
        )

    rows_per_date = split_data.groupby("date").size()

    if not (rows_per_date == 30).all():
        raise ValueError(
            "At least one date does not contain all 30 tradable symbols."
        )

    split_dates = (
        split_data
        .groupby("split")["date"]
        .agg(["min", "max", "nunique"])
        .sort_values("min")
    )

    train_max = split_dates.loc["train", "max"]
    validation_min = split_dates.loc["validation", "min"]
    validation_max = split_dates.loc["validation", "max"]
    test_min = split_dates.loc["test", "min"]

    if not train_max < validation_min:
        raise ValueError(
            "Train split overlaps with or follows validation split."
        )

    if not validation_max < test_min:
        raise ValueError(
            "Validation split overlaps with or follows test split."
        )

    if split_data.isna().sum().sum() > 0:
        raise ValueError("Split dataset contains missing values.")


def save_split_data(
    split_data: pd.DataFrame,
    output_path: Path = SPLIT_DATA_PATH,
) -> None:
    """Save the split-labeled dataset to Parquet."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    split_data.to_parquet(
        output_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )

    if not output_path.exists():
        raise FileNotFoundError(
            f"Split data file was not created: {output_path}"
        )

    if output_path.stat().st_size == 0:
        raise ValueError(
            f"Split data file is empty: {output_path}"
        )


def validate_saved_split_data(
    expected_data: pd.DataFrame,
    output_path: Path = SPLIT_DATA_PATH,
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


def print_split_summary(split_data: pd.DataFrame) -> None:
    """Print split date ranges and row counts."""
    print("Chronological splits assigned successfully.")
    print(f"Input path: {FEATURE_DATA_PATH}")
    print(f"Output path: {SPLIT_DATA_PATH}")

    print()
    print("Overall dataset:")
    print(f"Rows: {len(split_data):,}")
    print(f"Columns: {len(split_data.columns)}")
    print(f"First date: {split_data['date'].min().date()}")
    print(f"Last date: {split_data['date'].max().date()}")
    print(f"Symbols: {split_data['symbol'].nunique()}")

    print()
    print("Split summary:")
    split_summary = (
        split_data
        .groupby("split")
        .agg(
            first_date=("date", "min"),
            last_date=("date", "max"),
            trading_dates=("date", "nunique"),
            rows=("date", "size"),
            target_rate=("target_1d", "mean"),
        )
        .sort_values("first_date")
    )

    print(split_summary)

    print()
    print("Rows per split:")
    print(
        split_data["split"]
        .value_counts()
        .loc[["train", "validation", "test"]]
    )

    print()
    print("First five rows:")
    print(
        split_data[
            [
                "date",
                "symbol",
                "split",
                "resp_1d",
                "target_1d",
                "weight_equal",
                "weight_liquidity",
            ]
        ].head()
    )


def main() -> None:
    """Assign chronological train, validation, and test splits."""
    feature_data = load_feature_data()
    validate_feature_data(feature_data)

    split_data = assign_chronological_splits(feature_data)
    validate_split_data(split_data)

    print_split_summary(split_data)

    save_split_data(split_data)
    validate_saved_split_data(split_data)

    print()
    print("Split dataset saved successfully.")
    print(f"Output file size: {SPLIT_DATA_PATH.stat().st_size / 1_000_000:.2f} MB")


if __name__ == "__main__":
    main()