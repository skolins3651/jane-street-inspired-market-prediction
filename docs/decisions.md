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

The following questions remain unresolved and should be answered before or during later phases:

1. At what precise point during each trading day is a prediction assumed to be made?
2. Which future-return horizon will be the primary prediction target?
3. Will the target use absolute returns, SPY-relative excess returns, or both?
4. Will additional return horizons be created to parallel the original `resp` variables?
5. How will observation weights be defined?
6. What rule will convert a model prediction into an execute-or-pass action?
7. How will the Jane Street utility score be adapted to stock-date observations?
8. Which initial engineered features will be included?
9. How will chronological training, validation, and test periods be chosen?
10. How will transaction costs and other trading frictions eventually be incorporated?

These questions are intentionally left open until the relevant modeling and evaluation phases. Recording them now prevents provisional assumptions from becoming invisible architectural decisions.
