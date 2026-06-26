# Initial Dataset Data Dictionary

## Dataset

**File:** `data/raw/daily_prices_2016_2025.parquet`

**Source:** Public daily market data downloaded through `yfinance`

**Requested period:** January 1, 2016 through December 31, 2025

**Universe:** 30 selected U.S. equities plus SPY as the market benchmark

**Table grain:** One row represents one symbol on one trading date

The initial raw dataset contains market observations only. It does not yet contain engineered features, prediction targets, observation weights, or model decisions.

## Columns

| Column   | Data type | Meaning                                                    | Units or format                          | Project role                                                                     |
| -------- | --------- | ---------------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------- |
| `date`   | Datetime  | Trading date associated with the market observation.       | `YYYY-MM-DD`                             | Identifies when the observation occurred and establishes chronological order.    |
| `symbol` | String    | Ticker symbol for the security represented by the row.     | Exchange ticker, such as `AAPL` or `SPY` | Identifies the stock or benchmark associated with the observation.               |
| `open`   | Numeric   | Adjusted opening price for the trading session.            | U.S. dollars per share or fund unit      | Raw input from which later features may be constructed.                          |
| `high`   | Numeric   | Adjusted highest price reached during the trading session. | U.S. dollars per share or fund unit      | Raw input used to describe the session’s price range and volatility.             |
| `low`    | Numeric   | Adjusted lowest price reached during the trading session.  | U.S. dollars per share or fund unit      | Raw input used to describe the session’s price range and volatility.             |
| `close`  | Numeric   | Adjusted closing price for the trading session.            | U.S. dollars per share or fund unit      | Primary price series for constructing returns, targets, and many later features. |
| `volume` | Numeric   | Reported trading volume during the session.                | Shares or fund units traded              | Raw measure of trading activity and liquidity.                                   |

## Price Adjustment

The download pipeline uses `auto_adjust=True`. As a result, `yfinance` automatically adjusts the open, high, low, and close fields rather than supplying a separate adjusted-close column.

Adjusted prices improve comparability across time when securities experience events such as stock splits or dividend distributions. The dataset therefore contains `close`, but not a separate `adj_close` column.

## Observation Key

The combination of a date and symbol uniquely identifies a row. For example, `AAPL 2016-01-04` is one row, and more specifically represents stock price data for `AAPL` on January 4, 2016.

A date appears once for every symbol in the universe, and each symbol appears across many trading dates. The same date-symbol combination must not appear more than once.

## Benchmark

`SPY` is included in the table because it will later be used to represent broad U.S. equity-market performance. It is analytically separate from the 30 stocks that form the initial prediction universe.

Potential later uses include:

* calculating market-relative or excess returns;
* constructing benchmark-relative features;
* distinguishing broad market movement from security-specific movement.

## Fields Added Later

Processed versions of the dataset are expected to add fields such as:

* lagged returns;
* momentum and volatility features;
* volume-based features;
* market-relative features;
* forward-return targets;
* observation weights;
* model predictions;
* execute-or-pass decisions.

Those fields are not part of the initial raw-data schema and will be documented separately when they are created.
