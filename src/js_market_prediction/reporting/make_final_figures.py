from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

BASELINE_UTILITY_PATH = PROCESSED_DIR / "baseline_utility_scores.csv"
THRESHOLD_TUNING_PATH = PROCESSED_DIR / "logistic_threshold_tuning_results.csv"
EXPANDED_FEATURE_UTILITY_PATH = (
    PROCESSED_DIR / "logistic_regression_expanded_features_utility_scores.csv"
)
GRADIENT_BOOSTING_UTILITY_PATH = (
    PROCESSED_DIR / "gradient_boosting_baseline_utility_scores.csv"
)
UTILITY_AWARE_SCORES_PATH = PROCESSED_DIR / "utility_aware_action_rule_scores.csv"
RANK_REGRESSION_SCORES_PATH = PROCESSED_DIR / "cross_sectional_rank_regression_scores.csv"
RANK_REGRESSION_CANDIDATES_PATH = (
    PROCESSED_DIR / "cross_sectional_rank_regression_validation_candidates.csv"
)

VALIDATION_SPLIT = "validation"
TEST_SPLIT = "test"

WEIGHT_EQUAL = "weight_equal"
WEIGHT_LIQUIDITY = "weight_liquidity"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required results file not found: {path}\n"
            "Run the relevant modeling and scoring scripts before making final figures."
        )

    return pd.read_csv(path)


def get_single_row(data: pd.DataFrame, **criteria: object) -> pd.Series:
    mask = pd.Series(True, index=data.index)

    for column, value in criteria.items():
        if column not in data.columns:
            raise ValueError(f"Column not found: {column}")

        mask &= data[column] == value

    rows = data[mask]

    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one row for criteria {criteria}, found {len(rows)}."
        )

    return rows.iloc[0]


def get_threshold_row(
    data: pd.DataFrame,
    threshold: float,
    weight_column: str,
) -> pd.Series:
    mask = (
        (data["split"] == VALIDATION_SPLIT)
        & (data["weight_column"] == weight_column)
        & np.isclose(data["threshold"], threshold)
    )

    rows = data[mask]

    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one threshold-tuning row for "
            f"threshold={threshold}, weight_column={weight_column}; found {len(rows)}."
        )

    return rows.iloc[0]


def build_main_comparison() -> pd.DataFrame:
    baseline = load_csv(BASELINE_UTILITY_PATH)
    threshold = load_csv(THRESHOLD_TUNING_PATH)
    expanded = load_csv(EXPANDED_FEATURE_UTILITY_PATH)
    gradient = load_csv(GRADIENT_BOOSTING_UTILITY_PATH)
    utility_aware = load_csv(UTILITY_AWARE_SCORES_PATH)
    rank_regression = load_csv(RANK_REGRESSION_SCORES_PATH)

    rows = []

    def add_candidate(
        label: str,
        short_label: str,
        equal_row: pd.Series,
        liquidity_row: pd.Series,
    ) -> None:
        rows.append(
            {
                "model_rule": label,
                "short_label": short_label,
                "equal_utility": equal_row["utility"],
                "liquidity_utility": liquidity_row["utility"],
                "action_rate": liquidity_row["action_rate"],
                "mean_resp_taken": liquidity_row["mean_resp_taken"],
            }
        )

    add_candidate(
        label="Original logistic, threshold 0.50",
        short_label="Original logistic",
        equal_row=get_single_row(
            baseline,
            split=VALIDATION_SPLIT,
            baseline="logistic_regression",
            weight_column=WEIGHT_EQUAL,
        ),
        liquidity_row=get_single_row(
            baseline,
            split=VALIDATION_SPLIT,
            baseline="logistic_regression",
            weight_column=WEIGHT_LIQUIDITY,
        ),
    )

    add_candidate(
        label="Tuned-threshold logistic, threshold 0.48",
        short_label="Tuned logistic",
        equal_row=get_threshold_row(
            threshold,
            threshold=0.48,
            weight_column=WEIGHT_EQUAL,
        ),
        liquidity_row=get_threshold_row(
            threshold,
            threshold=0.48,
            weight_column=WEIGHT_LIQUIDITY,
        ),
    )

    add_candidate(
        label="Expanded-feature logistic, threshold 0.50",
        short_label="Expanded logistic",
        equal_row=get_single_row(
            expanded,
            split=VALIDATION_SPLIT,
            model="logistic_regression_expanded_features",
            weight_column=WEIGHT_EQUAL,
        ),
        liquidity_row=get_single_row(
            expanded,
            split=VALIDATION_SPLIT,
            model="logistic_regression_expanded_features",
            weight_column=WEIGHT_LIQUIDITY,
        ),
    )

    add_candidate(
        label="Gradient boosting, threshold 0.50",
        short_label="Gradient boosting",
        equal_row=get_single_row(
            gradient,
            split=VALIDATION_SPLIT,
            model="gradient_boosting_baseline",
            weight_column=WEIGHT_EQUAL,
        ),
        liquidity_row=get_single_row(
            gradient,
            split=VALIDATION_SPLIT,
            model="gradient_boosting_baseline",
            weight_column=WEIGHT_LIQUIDITY,
        ),
    )

    add_candidate(
        label="Utility-aware top-20/day rule",
        short_label="Top-20 logistic",
        equal_row=get_single_row(
            utility_aware,
            split=VALIDATION_SPLIT,
            top_k=20,
            weight_column=WEIGHT_EQUAL,
        ),
        liquidity_row=get_single_row(
            utility_aware,
            split=VALIDATION_SPLIT,
            top_k=20,
            weight_column=WEIGHT_LIQUIDITY,
        ),
    )

    add_candidate(
        label="Cross-sectional rank regression, top-8/day",
        short_label="Rank regression",
        equal_row=get_single_row(
            rank_regression,
            split=VALIDATION_SPLIT,
            top_k=8,
            weight_column=WEIGHT_EQUAL,
        ),
        liquidity_row=get_single_row(
            rank_regression,
            split=VALIDATION_SPLIT,
            top_k=8,
            weight_column=WEIGHT_LIQUIDITY,
        ),
    )

    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURE_DIR / filename

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output_path}")


def annotate_bars_horizontal(ax: plt.Axes, values: pd.Series) -> None:
    for index, value in enumerate(values):
        ax.text(
            value,
            index,
            f" {value:.2f}",
            va="center",
            fontsize=9,
        )


def make_validation_liquidity_utility_chart(comparison: pd.DataFrame) -> None:
    plot_data = comparison.sort_values("liquidity_utility", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(plot_data["model_rule"], plot_data["liquidity_utility"])
    annotate_bars_horizontal(ax, plot_data["liquidity_utility"])

    ax.set_title("Validation Liquidity-Weighted Utility by Model")
    ax.set_xlabel("Validation liquidity-weighted utility")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)

    save_figure(fig, "validation_liquidity_utility_by_model.png")


def make_equal_vs_liquidity_chart(comparison: pd.DataFrame) -> None:
    plot_data = comparison.copy()

    x_positions = np.arange(len(plot_data))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(
        x_positions - width / 2,
        plot_data["equal_utility"],
        width=width,
        label="Equal-weight utility",
    )
    ax.bar(
        x_positions + width / 2,
        plot_data["liquidity_utility"],
        width=width,
        label="Liquidity-weighted utility",
    )

    ax.set_title("Validation Utility by Weighting Scheme")
    ax.set_ylabel("Validation utility")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_data["short_label"], rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, "validation_equal_vs_liquidity_utility.png")


def make_action_rate_vs_utility_chart(comparison: pd.DataFrame) -> None:
    plot_data = comparison.copy()

    fig, ax = plt.subplots(figsize=(9, 6))

    x_values = plot_data["action_rate"] * 100
    y_values = plot_data["liquidity_utility"]

    ax.scatter(x_values, y_values)

    label_offsets = {
    "Tuned logistic": (-62, 8),
    "Rank regression": (6, 6),
    "Top-20 logistic": (6, 6),
    "Original logistic": (6, 8),
    "Expanded logistic": (6, -10),
    "Gradient boosting": (6, 6),
}

    for _, row in plot_data.iterrows():
        x = row["action_rate"] * 100
        y = row["liquidity_utility"]

        offset = label_offsets.get(row["short_label"], (5, 5))

        ax.annotate(
            row["short_label"],
            xy=(x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlim(10, 92)

    ax.set_title("Action Rate vs. Validation Utility")
    ax.set_xlabel("Action rate (%)")
    ax.set_ylabel("Validation liquidity-weighted utility")
    ax.grid(alpha=0.25)

    save_figure(fig, "action_rate_vs_validation_utility.png")


def make_threshold_tuning_curve() -> None:
    data = load_csv(THRESHOLD_TUNING_PATH)
    validation = data[data["split"] == VALIDATION_SPLIT].copy()

    equal = validation[validation["weight_column"] == WEIGHT_EQUAL].sort_values(
        "threshold"
    )
    liquidity = validation[validation["weight_column"] == WEIGHT_LIQUIDITY].sort_values(
        "threshold"
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        equal["threshold"],
        equal["utility"],
        marker="o",
        label="Equal-weight utility",
    )
    ax.plot(
        liquidity["threshold"],
        liquidity["utility"],
        marker="o",
        label="Liquidity-weighted utility",
    )

    ax.set_title("Logistic Threshold Tuning Curve")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Validation utility")
    ax.legend()
    ax.grid(alpha=0.25)

    save_figure(fig, "logistic_threshold_tuning_curve.png")


def make_rank_regression_top_k_curve() -> None:
    data = load_csv(RANK_REGRESSION_CANDIDATES_PATH)
    validation = data[data["split"] == VALIDATION_SPLIT].copy()

    equal = validation[validation["weight_column"] == WEIGHT_EQUAL].sort_values("top_k")
    liquidity = validation[validation["weight_column"] == WEIGHT_LIQUIDITY].sort_values(
        "top_k"
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(
        equal["top_k"],
        equal["utility"],
        marker="o",
        label="Equal-weight utility",
    )
    ax.plot(
        liquidity["top_k"],
        liquidity["utility"],
        marker="o",
        label="Liquidity-weighted utility",
    )

    ax.set_title("Cross-Sectional Rank Regression Top-$k$ Curve")
    ax.set_xlabel("Stocks selected per day")
    ax.set_ylabel("Validation utility")
    ax.legend()
    ax.grid(alpha=0.25)

    save_figure(fig, "rank_regression_top_k_curve.png")


def make_final_validation_vs_test_chart() -> None:
    data = load_csv(RANK_REGRESSION_SCORES_PATH)

    selected = data[
        (data["top_k"] == 8)
        & (data["split"].isin([VALIDATION_SPLIT, TEST_SPLIT]))
    ].copy()

    split_order = [VALIDATION_SPLIT, TEST_SPLIT]
    x_positions = np.arange(len(split_order))
    width = 0.38

    equal_values = [
        get_single_row(
            selected,
            split=split,
            top_k=8,
            weight_column=WEIGHT_EQUAL,
        )["utility"]
        for split in split_order
    ]

    liquidity_values = [
        get_single_row(
            selected,
            split=split,
            top_k=8,
            weight_column=WEIGHT_LIQUIDITY,
        )["utility"]
        for split in split_order
    ]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        x_positions - width / 2,
        equal_values,
        width=width,
        label="Equal-weight utility",
    )
    ax.bar(
        x_positions + width / 2,
        liquidity_values,
        width=width,
        label="Liquidity-weighted utility",
    )

    ax.set_title("Final Model: Validation vs. Test Utility")
    ax.set_ylabel("Utility")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Validation", "Test"])
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    save_figure(fig, "final_model_validation_vs_test.png")


def main() -> None:
    comparison = build_main_comparison()

    make_validation_liquidity_utility_chart(comparison)
    make_equal_vs_liquidity_chart(comparison)
    make_action_rate_vs_utility_chart(comparison)
    make_threshold_tuning_curve()
    make_rank_regression_top_k_curve()
    make_final_validation_vs_test_chart()

    print("\nFinal figures created successfully.")


if __name__ == "__main__":
    main()