# Modeling Problem Definition

This document defines the initial prediction problem for the Jane Street–inspired market prediction project.

## Prediction Question

For each stock-date observation, can information available by the close of trading day $t$ be used to predict whether the stock will outperform SPY from close $t$ to close $t+1$?

This adapts the original Jane Street competition framework to public daily equity data. The project preserves the execute-or-pass structure of the original challenge, but it does not claim to reproduce Jane Street's proprietary trading opportunities, anonymous features, response variables, or observation weights.

## Observation Unit

Each modeling row represents one stock on one trading date.

The 30 selected equities form the tradable prediction universe. SPY is included in the dataset as the market benchmark, but SPY is not treated as a tradable prediction row for the initial modeling problem.

## Prediction Timing

The prediction is assumed to be made after the market close on trading day $t$.

This means that features for date $t$ may use that day's open, high, low, close, and volume, along with historical information from prior dates. Features may not use information from date $t+1$ or later.

## Primary Response Variable

The primary continuous response variable is `resp_1d`, the next-day SPY-relative excess return.

```text
stock_return_1d_forward = close_stock[t+1] / close_stock[t] - 1

spy_return_1d_forward = close_SPY[t+1] / close_SPY[t] - 1

resp_1d = stock_return_1d_forward - spy_return_1d_forward
```

A positive `resp_1d` means that the stock outperformed SPY over the next close-to-close trading interval. A negative `resp_1d` means that the stock underperformed SPY over that interval.

The project uses `resp_1d` rather than a generic `resp` column name so the prediction horizon remains explicit. Additional response horizons may be added in later extensions, but the baseline modeling problem focuses only on next-day outperformance.

## Binary Target

The initial model target is `target_1d`, a binary classification label:

```text
target_1d = 1 if resp_1d > 0 else 0
```

A target value of `1` means the stock outperformed SPY over the next trading day. A target value of `0` means the stock did not outperform SPY.

If `resp_1d` is exactly zero, `target_1d` is defined as `0`, because the stock did not outperform the benchmark.

## Observation Weights

The initial modeling dataset includes two observation-weight columns.

```text
weight_equal = 1.0
```

`weight_equal` treats every valid stock-date observation equally. This is the clearest default for baseline modeling because it makes the first results easy to interpret.

The dataset also includes a liquidity-based weight:

```text
dollar_volume = close * volume

adv20 = 20-day rolling average of dollar_volume for each stock, using information available through day t

weight_liquidity = adv20 / cross_sectional_mean_adv20_on_date_t
```

`weight_liquidity` gives more importance to observations from more liquid stocks while keeping the average liquidity weight on each date close to `1.0`. This makes liquidity-weighted results comparable to equal-weight results without allowing the overall scale of weights to drift upward or downward over time.

The initial dataset does not intentionally create zero-weight observations. Zero weights may be considered later if a clear exclusion rule is introduced, but arbitrary zero weights would not have a well-defined meaning in this public-data adaptation.

## Action Definition

The initial action space is binary:

```text
action = 1 means take the long stock opportunity
action = 0 means pass
```

In Phase 3, the model does not short stocks. It only decides whether to take or pass on a long opportunity.

## Relationship Between Target and Action

The binary target describes what actually happened after the prediction time.

The action describes what the model chooses to do before the outcome is known.

In baseline models, the action may be created directly from a rule or from a predicted probability. Later phases may explore more sophisticated thresholding, probability calibration, or utility-based decision rules.

## Initial Modeling Scope

The initial model focuses on a simple, leakage-aware, time-ordered baseline modeling problem.

Included in the initial scope:

* next-day SPY-relative outperformance;
* binary long/pass decisions;
* features available by the close of trading day $t$;
* chronological train, validation, and test splits;
* simple non-ML and machine-learning baselines.

Excluded from the initial scope:

* shorting;
* intraday prediction;
* transaction costs;
* slippage;
* bid-ask spreads;
* market impact;
* portfolio capital constraints;
* longer-horizon primary targets;
* regression-based return prediction;
* utility-optimized threshold selection.

These exclusions are intentional. They keep the project focused on creating a clean baseline problem before more advanced modeling and evaluation choices are introduced in later phases.
