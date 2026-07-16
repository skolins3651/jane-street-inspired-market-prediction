# Equal-Weight and Liquidity-Weight Utility Comparison

This document compares the initial baseline strategies under equal-weight and liquidity-weight adapted utility.

## Purpose

The project reports both equal-weight and liquidity-weight utility because the two weighting schemes answer different questions.

Equal-weight utility asks whether a strategy performs broadly across stock-date observations.

Liquidity-weight utility asks whether a strategy performs better on more liquid, economically significant observations.

The main project score is liquidity-weighted utility, but equal-weight utility remains an important sanity check.

## Summary of Validation Results

| Baseline | Equal-Weight Utility | Liquidity-Weight Utility | Interpretation |
|---|---:|---:|---|
| Always-pass | 0.000000 | 0.000000 | Takes no opportunities. |
| Always-take | 0.843609 | 2.166882 | Positive under both weighting schemes. |
| 20-day momentum rule | 0.439000 | 0.002253 | Positive equal-weight utility, but almost no liquidity-weight utility. |
| Logistic regression | 0.000000 | 2.870843 | Best validation liquidity-weight utility, but equal-weight total profit is negative. |

## Summary of Test Results

| Baseline | Equal-Weight Utility | Liquidity-Weight Utility | Interpretation |
|---|---:|---:|---|
| Always-pass | 0.000000 | 0.000000 | Takes no opportunities. |
| Always-take | 0.004531 | 16.438757 | Strongly benefits from liquidity weighting on the test split. |
| 20-day momentum rule | 0.000000 | 7.565544 | Negative equal-weight total profit but positive liquidity-weight utility. |
| Logistic regression | 0.000000 | 2.476813 | Positive liquidity-weight utility but negative equal-weight total profit. |

## Interpretation

The comparison shows that several conclusions depend materially on the weighting scheme.

On the validation split, logistic regression has the highest liquidity-weighted utility among the initial baselines. However, its equal-weight utility is zero because its equal-weight total profit is negative. This suggests that the model's apparent improvement is concentrated in liquidity-weighted observations rather than broad stock-date performance.

The 20-day momentum rule behaves differently. It has positive equal-weight validation utility but almost no liquidity-weight validation utility, suggesting that its selected opportunities do not translate well into liquidity-weighted performance.

Always-take is a useful reference point because it performs positively under both validation weighting schemes, though it is not selective.

On the test split, always-take has the strongest liquidity-weighted utility. This result should be treated as final-reporting context rather than as a tuning signal, because the test split should not be used to choose features, thresholds, or models.

## Practical Conclusion

The main validation result is mixed.

Logistic regression is the best baseline by validation liquidity-weighted utility, which is the main project score. However, its equal-weight behavior is weak, and its test performance does not dominate the simpler baselines.

Therefore, future improvements should not merely optimize liquidity-weighted utility in isolation. They should also check equal-weight utility, total profit, daily-profit volatility, action rate, and mean response taken.

## Implications for Future Experiments

Future modeling experiments should be judged primarily by validation liquidity-weighted utility, but an improvement should be considered more convincing if it also shows:

* nonnegative or improved equal-weight utility;
* positive total profit under both weighting schemes;
* reasonable action rate;
* no dependence on one or two outlier dates;
* interpretable behavior compared with always-take and logistic regression.