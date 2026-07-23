# Project Decisions and Limitations

This document records the project’s current architectural decisions, known limitations, and unresolved questions. It should be updated whenever a major design choice changes.

## Confirmed Decisions

### D001 — Use public market data

The original Jane Street competition dataset is unavailable, so the project uses publicly accessible historical market data rather than synthetic observations.

### D002 — Use `yfinance` for the initial dataset

Version 1 uses daily adjusted OHLCV data downloaded through `yfinance`. This source is suitable for an educational portfolio project, but it is not institutional-grade market data.

### D003 — Use a fixed research universe

The initial prediction universe contains 30 selected U.S. equities. SPY is included separately as a broad-market benchmark.

### D004 — Use a fixed historical period

The canonical initial dataset covers January 1, 2016 through December 31, 2025. The exclusive download end date is January 1, 2026.

### D005 — Store one row per date and symbol

The canonical raw table uses long format. Each row represents one security on one trading date and contains adjusted open, high, low, close, and volume observations.

### D006 — Store raw data as Parquet

The validated raw table is saved as Parquet so that numeric and datetime types are preserved efficiently. A JSON manifest records the dataset’s provenance, parameters, dimensions, package versions, and validation results.

### D007 — Preserve the competition framework, not the proprietary dataset

The project is intended to preserve the original challenge’s time-ordered prediction, execute-or-pass decisions, observation weighting, and utility-oriented evaluation. It does not claim to reproduce Jane Street’s proprietary trading opportunities or anonymous features.

### D008 — Use chronological evaluation

Future train, validation, and test splits will respect time order. Random shuffling will not be used for primary model evaluation.

### D009 — Define the initial modeling problem as next-day SPY-relative outperformance

The initial prediction problem uses one row per stock-date observation, excluding SPY as a tradable prediction row. The model observes features available by the close of trading day $t$ and predicts whether the stock will outperform SPY from close $t$ to close $t+1$.

The primary continuous response variable is next-day excess return relative to SPY:

```text
resp = next_day_stock_return - next_day_spy_return
```

The binary target is:

```text
target = 1 if resp > 0 else 0
```

The initial action space is binary. An action of `1` means taking the long stock opportunity, while an action of `0` means passing. Shorting, intraday prediction, longer primary horizons, and regression-based return prediction are intentionally deferred to possible future extensions.

### D010 — Create only the primary next-day target for the baseline modeling dataset

The initial modeling dataset will create one primary response horizon: next-day SPY-relative excess return.

The main target columns are:

```text
stock_return_1d_forward
spy_return_1d_forward
resp_1d
target_1d
```

`resp_1d` is defined as next-day stock return minus next-day SPY return. `target_1d` equals `1` when `resp_1d > 0` and `0` otherwise.

Secondary response horizons, such as 5-day or 10-day forward returns, are intentionally deferred. They may be useful for later model comparisons or extensions, but they are not needed for the initial baseline modeling problem.

The initial modeling dataset will also include two observation-weight columns:

```text
weight_equal = 1.0
```

and

```text
weight_liquidity = adv20 / cross_sectional_mean_adv20_on_date_t
```

where `adv20` is each stock's 20-day rolling average dollar volume, calculated using information available through day $t$.

Equal weights will be the default for the first baseline results. Liquidity weights are included so later evaluations can test whether weighting observations by economic significance changes the conclusions.

### D011 — Use liquidity-weighted adapted utility as the main final evaluation score

The project will evaluate model actions using an adapted Jane Street-style utility score.

For each trading date $t$, daily profit is defined as:

$$
p_t = \sum_{j=1}^{M} w_{j,t} r_{j,t} a_{j,t}
$$

where $j$ indexes stocks, $M$ is the number of tradable stocks, $r_{j,t}$ is the value of `resp_1d`, $a_{j,t}$ is the model action, and $w_{j,t}$ is the observation weight.

The consistency multiplier is:

$$
c = \frac{\sum_t p_t}{\sqrt{\sum_t p_t^2}} \cdot \sqrt{\frac{250}{N}}
$$

where $N$ is the number of unique trading dates in the evaluated split.

The final adapted utility score is:

$$
u = \min(\max(c, 0), 6) \cdot \sum_t p_t
$$

If $\sum_t p_t^2 = 0$, then the strategy produced zero daily profit on every evaluated date. In that case, the project defines $c = 0$ and $u = 0$.

The score will be calculated under both equal weights and liquidity weights. Equal-weight utility is used as an interpretability and sanity-check score. Liquidity-weighted utility is the main final evaluation score because it better reflects the goal of rewarding consistent above-market performance on more economically meaningful opportunities.

The main model-selection comparison is validation liquidity-weighted utility. Equal-weight utility and companion diagnostics, including `total_profit`, `mean_daily_profit`, `daily_profit_std`, `c_stat`, `action_rate`, and `mean_resp_taken`, should be reported as supporting evidence.

The test split should be reserved for final reporting after the model, feature set, or action rule has already been selected.


## Known Limitations

### Fixed-universe selection bias

The stock universe was selected using present-day knowledge and is not a point-in-time reconstruction of a historical index. The dataset may therefore be affected by survivorship and selection bias.

### Public-data limitations

Yahoo Finance data is appropriate for personal and educational research, but it may contain revisions, rounding differences, or occasional source-quality issues. The download pipeline validates the table but cannot guarantee institutional-grade accuracy.

### Daily rather than intraday observations

The original Jane Street observations represented proprietary trading opportunities. This project instead uses daily stock-level observations, so its market structure and prediction problem are materially different.

### No original competition features or outcomes

The dataset does not contain Jane Street’s anonymous features, original response variables, or proprietary observation weights. Features, targets, and weights must be designed from the public data.

### Adjusted historical prices

The OHLC fields are adjusted for corporate actions. These values are useful for return analysis but may differ from the prices that were displayed in the market on the original trading date.

### Simplified initial scope

The initial dataset does not yet model transaction costs, bid-ask spreads, slippage, market impact, short-selling constraints, or portfolio-capital limits.

## Open Design Decisions

The following questions remain unresolved and should be answered before or during later work:

1. Which initial engineered features will be included?
2. How will chronological training, validation, and test periods be chosen?
3. Which baseline strategies and baseline models will be compared?
4. How will the Jane Street utility score be adapted to stock-date observations?
5. How will transaction costs and other trading frictions eventually be incorporated?
6. Will later extensions explore additional response horizons, regression-based return prediction, probability thresholds, shorting, or longer-horizon targets?

Some earlier open questions were resolved in the initial modeling decisions (D009 and D010 especially), including prediction timing, the primary target horizon, the use of SPY-relative returns, the binary action definition, and the initial observation-weight definitions.
