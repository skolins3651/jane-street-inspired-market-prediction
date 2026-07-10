import numpy as np
import pandas as pd

from js_market_prediction.evaluation.utility import calculate_adapted_utility


TOLERANCE = 1e-12


def assert_close(
    actual: float,
    expected: float,
    metric_name: str,
    tolerance: float = TOLERANCE,
) -> None:
    """Assert that two numeric values are close, allowing NaN checks."""
    if pd.isna(expected):
        if not pd.isna(actual):
            raise AssertionError(
                f"{metric_name}: expected NaN, got {actual}"
            )

        return

    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"{metric_name}: expected {expected}, got {actual}"
        )


def assert_results_close(
    actual: dict[str, float | int],
    expected: dict[str, float | int],
) -> None:
    """Assert all expected score outputs match."""
    for metric_name, expected_value in expected.items():
        actual_value = actual[metric_name]

        if isinstance(expected_value, int):
            if actual_value != expected_value:
                raise AssertionError(
                    f"{metric_name}: expected {expected_value}, "
                    f"got {actual_value}"
                )
        else:
            assert_close(
                actual=float(actual_value),
                expected=float(expected_value),
                metric_name=metric_name,
            )


def make_example_data(
    dates: list[str],
    responses: list[float],
    actions: list[int],
    weights: list[float],
) -> pd.DataFrame:
    """Create a small scoring example."""
    return pd.DataFrame(
        {
            "date": dates,
            "resp_1d": responses,
            "action": actions,
            "weight_equal": weights,
        }
    )


def validate_all_pass() -> None:
    """All-pass should produce zero utility and undefined mean response."""
    data = make_example_data(
        dates=[
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
        ],
        responses=[
            0.01,
            -0.02,
            0.03,
            0.01,
        ],
        actions=[
            0,
            0,
            0,
            0,
        ],
        weights=[
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    )

    actual = calculate_adapted_utility(
        data=data,
        action_column="action",
        weight_column="weight_equal",
    )

    expected = {
        "utility": 0.0,
        "total_profit": 0.0,
        "mean_daily_profit": 0.0,
        "daily_profit_std": 0.0,
        "c_stat": 0.0,
        "action_rate": 0.0,
        "mean_resp_taken": np.nan,
        "num_days": 2,
    }

    assert_results_close(
        actual=actual,
        expected=expected,
    )


def validate_all_positive_capped() -> None:
    """Consistent positive profits should be capped at multiplier 6."""
    data = make_example_data(
        dates=[
            "2024-01-02",
            "2024-01-03",
        ],
        responses=[
            0.01,
            0.02,
        ],
        actions=[
            1,
            1,
        ],
        weights=[
            1.0,
            1.0,
        ],
    )

    actual = calculate_adapted_utility(
        data=data,
        action_column="action",
        weight_column="weight_equal",
    )

    expected = {
        "utility": 0.18,
        "total_profit": 0.03,
        "mean_daily_profit": 0.015,
        "daily_profit_std": 0.005,
        "c_stat": 15.0,
        "action_rate": 1.0,
        "mean_resp_taken": 0.015,
        "num_days": 2,
    }

    assert_results_close(
        actual=actual,
        expected=expected,
    )


def validate_all_negative() -> None:
    """Negative total profit should produce zero clipped utility."""
    data = make_example_data(
        dates=[
            "2024-01-02",
            "2024-01-03",
        ],
        responses=[
            -0.01,
            -0.02,
        ],
        actions=[
            1,
            1,
        ],
        weights=[
            1.0,
            1.0,
        ],
    )

    actual = calculate_adapted_utility(
        data=data,
        action_column="action",
        weight_column="weight_equal",
    )

    expected = {
        "utility": 0.0,
        "total_profit": -0.03,
        "mean_daily_profit": -0.015,
        "daily_profit_std": 0.005,
        "c_stat": -15.0,
        "action_rate": 1.0,
        "mean_resp_taken": -0.015,
        "num_days": 2,
    }

    assert_results_close(
        actual=actual,
        expected=expected,
    )


def validate_volatile_positive_uncapped() -> None:
    """Volatile positive profit should have a lower uncapped multiplier."""
    data = make_example_data(
        dates=[
            "2024-01-02",
            "2024-01-03",
        ],
        responses=[
            0.10,
            -0.05,
        ],
        actions=[
            1,
            1,
        ],
        weights=[
            1.0,
            1.0,
        ],
    )

    actual = calculate_adapted_utility(
        data=data,
        action_column="action",
        weight_column="weight_equal",
    )

    expected = {
        "utility": 0.25,
        "total_profit": 0.05,
        "mean_daily_profit": 0.025,
        "daily_profit_std": 0.075,
        "c_stat": 5.0,
        "action_rate": 1.0,
        "mean_resp_taken": 0.025,
        "num_days": 2,
    }

    assert_results_close(
        actual=actual,
        expected=expected,
    )


def validate_weighted_selected_profit() -> None:
    """Only selected rows should contribute weighted response."""
    data = make_example_data(
        dates=[
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
        ],
        responses=[
            0.01,
            -0.02,
            0.03,
            0.01,
        ],
        actions=[
            1,
            0,
            1,
            1,
        ],
        weights=[
            2.0,
            10.0,
            1.0,
            3.0,
        ],
    )

    actual = calculate_adapted_utility(
        data=data,
        action_column="action",
        weight_column="weight_equal",
    )

    expected = {
        "utility": 0.48,
        "total_profit": 0.08,
        "mean_daily_profit": 0.04,
        "daily_profit_std": 0.02,
        "c_stat": 14.142135623730951,
        "action_rate": 0.75,
        "mean_resp_taken": 0.016666666666666666,
        "num_days": 2,
    }

    assert_results_close(
        actual=actual,
        expected=expected,
    )


def main() -> None:
    """Run all hand-checkable utility validation examples."""
    validation_examples = [
        ("all-pass", validate_all_pass),
        ("all-positive capped", validate_all_positive_capped),
        ("all-negative", validate_all_negative),
        (
            "volatile positive uncapped",
            validate_volatile_positive_uncapped,
        ),
        ("weighted selected profit", validate_weighted_selected_profit),
    ]

    for example_name, validation_function in validation_examples:
        validation_function()
        print(f"PASS: {example_name}")

    print()
    print("All utility validation examples passed.")


if __name__ == "__main__":
    main()