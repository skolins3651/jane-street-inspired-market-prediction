# Utility Score Definition

This document defines the trading-style evaluation framework for the Jane Street–inspired market prediction project.

The project uses a public stock-date dataset rather than the original proprietary Jane Street competition data. The evaluation framework therefore adapts the original competition's utility-score philosophy to the project's one-day SPY-relative outperformance problem.

## Evaluation Goal

The goal is not merely to classify stock-date observations correctly. The goal is to evaluate whether a model can choose long opportunities that generate positive, reasonably consistent next-day returns relative to SPY.

The utility score is designed to reward three things:

1. positive total realized response from selected opportunities;
2. consistency of realized daily performance;
3. selective action rather than taking every opportunity indiscriminately.

## Inputs

The utility score is calculated from a table containing at least the following columns:

| Column             | Meaning                                                              |
| ------------------ | -------------------------------------------------------------------- |
| `date`             | Trading date associated with the prediction row.                     |
| `resp_1d`          | Next-day SPY-relative excess return.                                 |
| `action`           | Model decision: `1` means take the long opportunity, `0` means pass. |
| `weight_equal`     | Equal observation weight.                                            |
| `weight_liquidity` | Liquidity-based observation weight.                                  |

The response variable is defined as a difference between a stock's next-day return and SPY's next-day return. The former is defined as follows:

```math
\text{stock\_return\_1d\_forward}_{j,t}
=
\frac{\text{close}_{j,t+1}}{\text{close}_{j,t}} - 1
```

where $j$ indexes a stock, $t$ is a trading date, and $\text{close}_{j,t}$ is the closing price of stock $j$ on date $t$. The latter is defined as:

```math
\text{spy\_return\_1d\_forward}_{t}
=
\frac{\text{close}_{\text{SPY},t+1}}{\text{close}_{\text{SPY},t}} - 1
```

using the same trading-date index $t$, but applied to SPY. Finally, we have:

```math
\text{resp\_1d}_{j,t}
=
\text{stock\_return\_1d\_forward}_{j,t}
-
\text{spy\_return\_1d\_forward}_{t}.
```

A positive `resp_1d` means that the stock outperformed SPY over the next close-to-close trading interval. A negative `resp_1d` means that the stock underperformed SPY over that interval.

## Daily Profit

For each trading date $t$, daily profit is defined as:

$$
p_t = \sum_{j=1}^{M} w_{j,t} \, r_{j,t} \, a_{j,t}
$$

where $M$ is the number of tradable stocks, $w_{j,t}$ is the observation weight of stock $j$ on date $t$, $r_{j,t} = \text{resp\_1d}_{j,t}$, and $a_{j,t}$ is the action for stock $j$ on date $t$. In general, the subscript $j,t$ will always represent $j$ indexing the stock observations available on date $t$ in this document.

This means:

* rows with `action = 0` contribute nothing to daily profit;
* rows with `action = 1` contribute their weighted next-day SPY-relative response;
* positive selected responses increase daily profit;
* negative selected responses decrease daily profit.

The daily aggregation is important because the score should reward strategies that perform consistently across time, not strategies that rely entirely on one unusually lucky day.

## Adapted Utility Formula

After calculating daily profits $p_t$, the utility score is computed using the Jane Street-style structure for a consistency multiplier:

$$
c = \frac{\sum_t p_t}{\sqrt{\sum_t p_t^2}} \cdot \sqrt{\frac{250}{N}}
$$

where $N$ is the number of unique trading dates being evaluated.

If $\sum_t p_t^2 = 0$, which is relevant in the always-pass strategy, then the strategy produced zero daily profit on every evaluated date. In that case, the project defines $c = 0$ and $u = 0$.

The final utility score is:

$$
u = \min(\max(c, 0), 6) \cdot \sum_t p_t
$$

The term $c$ behaves like a stability-adjusted return measure. It is higher when total daily profit is positive and when the profit is spread consistently across many dates.

The clipping term $\min(\max(c, 0), 6)$ has two effects:

* strategies with negative stability-adjusted performance receive no positive utility credit;
* extremely high stability-adjusted values are capped so that the score does not grow without bound from the multiplier alone.

This consistency multiplier helps standardize the score and, because of clipping, flattens strategies with negative stability-adjusted performance to zero utility. That behavior is useful for a contest metric, but it may be undesirable in other contexts, such as practical risk analysis or comparing two losing strategies. In those settings, it may be important to know whether one strategy has substantially worse raw profit, daily volatility, or downside behavior than another. The multiplier is included to preserve the spirit of the original Jane Street competition metric, not because it is a universal measure of financial risk or model quality.

## Equal-Weight and Liquidity-Weight Scores

The project calculates the adapted utility score under two weighting schemes.

### Equal-Weight Utility

Equal-weight utility uses:

$$
w^{\mathrm{equal}}_{j,t} = 1
$$

where every valid stock-date observation receives the same weight $w_{j,t} = 1$.

Equal-weight utility is the clearest sanity-check score. It treats every stock-date opportunity equally and makes the first scoring results easier to interpret.

### Liquidity-Weighted Utility

Liquidity-weighted utility is based on each stock's 20-day rolling average dollar volume relative to the cross-sectional average on the same date. The 20-day rolling average dollar volume is calculated as:

$$
A_{j,t} = \frac{1}{20} \sum_{s=t-19}^{t} \text{close}_{j,s} \cdot \text{volume}_{j,s}
$$

where $\text{close}_{j,s}$ is the closing price of stock $j$ on date $s$ and $\text{volume}_{j,s}$ is the trading volume of stock $j$ on date $s$. The product of these two values yields the dollar volume of a stock $j$ on date $s$; the value of $A_{j,t}$, then, represents the average of a certain stock's dollar volumes over the last 20 days.

We then define the cross-sectional mean dollar volume on a date $t$ as follows:

$$
\bar{A}_t = \frac{1}{M} \sum_{j=1}^{M} A_{j,t}
$$

where as before, $M$ is the number of tradable stocks on day $t$. Finally, liquidity weight is defined as:

$$
w^{\mathrm{liquidity}}_{j,t} = \frac{A_{j,t}}{\bar{A}_t}
$$

Liquidity weighting is used as a practical proxy for economic significance and trading capacity in the public-data setting. It does not perfectly measure institutional opportunity size, market impact, or true trade capacity, but it better reflects the idea that not every stock-date opportunity should count equally.

## Primary and Secondary Scores

The main final evaluation score is **liquidity-weighted adapted utility**.

That is to say, the utility $u$ is calculated using the liquidity weights $w^{\mathrm{liquidity}}_{j,t}$, which are used in the formula for the daily profit $p_t$.

This score best matches the spirit of the original Jane Street metric because it emphasizes consistent above-market performance on more economically meaningful opportunities.

Equal-weight utility remains important as a companion score. It is used to verify that results are not purely an artifact of the liquidity-weighting scheme.

The intended interpretation is:

* liquidity-weighted utility is the main project score;
* equal-weight utility is a sanity-check and interpretability score;
* both should be reported when comparing strategies or models.

## Companion Diagnostics

The utility score should not be interpreted alone.

The evaluation should also report:

| Diagnostic          | Purpose                                                 |
| ------------------- | ------------------------------------------------------- |
| `total_profit`      | Sum of daily profits across the evaluation period.      |
| `mean_daily_profit` | Average daily profit.                                   |
| `daily_profit_std`  | Volatility of daily profit.                             |
| `c_stat`            | Stability-adjusted score multiplier before clipping.    |
| `utility`           | Final adapted utility score.                            |
| `action_rate`       | Fraction of rows where the model takes the opportunity. |
| `mean_resp_taken`   | Average response among selected opportunities.          |
| `num_days`          | Number of trading dates evaluated.                      |

These diagnostics help distinguish genuinely useful strategies from pathological ones.

For example, a model might improve utility by trading only a tiny number of observations, relying on one unusually lucky date, or taking unstable risks. Companion diagnostics make those cases visible.

## Split Usage

Model and strategy comparisons should be made using the validation split.

The test split should be reserved for final reporting after a model, feature set, or action rule has already been selected. Test results should not be used repeatedly to tune features, thresholds, or model settings.

This protects the test split from becoming an accidental second validation set.

## What Counts as an Improvement

An improvement means:

1. higher validation liquidity-weighted utility than the relevant baseline;
2. no obviously pathological companion diagnostics;
3. interpretable behavior under equal-weight utility;
4. clear documentation of what changed.

Examples of pathological behavior include:

* extremely low action rate without a clear reason;
* utility driven by one or two outlier dates;
* strong liquidity-weighted improvement paired with nonsensical equal-weight behavior;
* higher utility caused by accidental leakage or split misuse;
* improved score with no reproducible explanation of what changed.

The main comparison should therefore be **validation liquidity-weighted utility** (liquidity-weighted utility on the validation split), with equal-weight utility and companion diagnostics used as supporting evidence.

## All-Pass Behavior

The always-pass strategy has `action = 0` for every row.

This produces zero daily profit on every date and therefore utility of zero.

The all-pass strategy is useful as a reference point, but a utility score of zero does not necessarily mean a strategy is harmless. A strategy with negative aggregate performance can also receive zero utility because the score clips negative stability-adjusted performance. For that reason, total profit and daily-profit diagnostics must always be reported alongside utility.

## Scope Limitations

The adapted utility score does not yet include:

* transaction costs;
* bid-ask spreads;
* slippage;
* market impact;
* borrow costs;
* short-selling constraints;
* portfolio capital limits;
* position sizing beyond observation weights.

These limitations are intentional for the current version of the project. The score evaluates selected long opportunities in a simplified public-data setting. Later extensions may make the evaluation more realistic.
