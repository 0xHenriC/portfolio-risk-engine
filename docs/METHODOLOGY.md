# Methodology

Notation: `q_i` quantity of line i, `P_i,t` close in quote currency, `X_c,t` EUR per unit of
currency c, `u_i` unit factor (held unit / quote unit), `V_i` today's EUR value of line i.
Returns are in percent unless stated.

## 1. Positions and cost (`journal.py`)

- A purchase lot costs `q x p x X` at the trade-date rate; the price includes fees, a sale
  price is net of fees.
- Cost basis is the **weighted average cost per holding cycle**. A cycle ends when the
  quantity returns to zero (tolerance 1e-6); the next purchase starts a new average.
- **Same session.** The journal has no time of day. A session's sales draw first on the
  quantity held before the session, at the prior average `A_prev`, then on the session's
  purchases at their mean price `A_day`. All the session's sales share one unit cost
  `(s_prev A_prev + s_day A_day) / (s_prev + s_day)`.
- Realised P&L of a sale = proceeds - quantity x unit cost, in EUR (currency included).
- The running quantity must equal the declared holding for every ticker, or the run stops.
  Input errors (side, sign, date, currency) are all collected and reported together.

## 2. Valuation and indices (`portfolio.py`)

- Line value `q_i x P_i,t x X_c,t x u_i`. A close older than 3 Monday-Friday sessions is not
  used; the line leaves the total and the total is flagged incomplete.
- **Chained indices.** Each day compares only positions held yesterday, at yesterday's
  quantities: `I_t = I_t-1 x sum(q_i,t-1 P_i,t) / sum(q_i,t-1 P_i,t-1)`. Purchases and sales
  therefore never register as performance. The total-return index adds gross dividends
  on their ex-date: `T_t = T_t-1 x (sum q P_t + sum q D_t) / sum q P_t-1`.

## 3. Performance by period (`performance.py`)

- Windows: 7 and 30 calendar days, year-to-date (from 31 December), 365 days, since the
  first purchase. The start is the last weekday session on or before the target date.
- Headline return = `T_end / T_start - 1`: time-weighted, gross dividends included, not
  annualised.
- Decomposition in EUR, line by line: market effect = gain on the start holding plus the
  gain on units bought in the window, valued at the end price or at their sale price;
  flow effect = purchases - sale proceeds. Units sold are matched first against the
  start holding, then against the window's purchases in date order.
  `value_end - value_start = market + flow` must hold within EUR 0.01.

## 4. Dividends and register (`dividends.py`, `register.py`)

- Dividend in EUR = amount x (EUR close / local close) on the ex-date or the last session
  before; dropped if no rate is known. Received = quantity held at the close of the
  session before the ex-date x amount: a purchase on the ex-date gives no right to the
  dividend, a sale on the ex-date keeps it (market rule).
- A trailing yield outside [0, 30 %] is treated as a data error and ignored.
- Register: local return `(sales + remaining x local close + local dividends) / local cost - 1`;
  EUR total return `(proceeds + remaining value + EUR dividends) / invested - 1`; currency
  effect `(1 + total) / (1 + local) - 1`, so the product identity holds exactly.

## 5. Risk (`risk.py`)

- Window: EUR closes of the held lines over at most 10 years to the valuation date,
  weekdays only.
- Daily portfolio return by full revaluation of today's positions:
  `r_t = sum_{i in A_t} V_i r_i,t / sum_{i in A_t} V_i`, where `A_t` is the set of lines
  with a return on day t; a day is kept only if `|A_t| >= max(2, n/2)`. Missing closes stay
  missing (no forward fill before returns).
- Volatility: `sd(r)` with ddof 1, annualised by `sqrt(252)`.
- **Historical VaR** at level c: the `1 - c` quantile of `r`, linear interpolation at
  position `(1 - c)(n - 1)` of the sorted sample. **Historical ES**: mean of returns at
  or below the VaR.
- **Normal VaR**: `mu - sigma z_c`; **normal ES**: `mu - sigma phi(z_c) / (1 - c)`, with
  `z_c = Phi^-1(c)`, sample mean and sd.
- **Rank method** (textbook, no interpolation): `k = round(n(1 - c))`, VaR = k-th worst
  return, ES = mean of the k - 1 worse returns.
- EUR amounts: percent x covered value.
- Consistency: ES must be at least as large a loss as the VaR at the same level, for every
  method; otherwise `RiskConsistencyError` is raised.
- **Maximum drawdown**: index `100 x prod(1 + r/100)`, drawdown `(I - max_{s<=t} I_s) / max`;
  reports peak, trough, recovery dates and the number of sessions between them.
- **Square root of time**: `VaR_h = VaR_1 x sqrt(h)`. Valid for i.i.d. returns with zero mean
  and a static book; wrong under volatility clustering, autocorrelation or fat tails.

## 6. Backtest (`backtest.py`)

- For each day t, the 99 % VaR is estimated on returns t-250 to t-1 (both methods).
  Exception: `r_t < VaR_t`.
- **Kupiec (proportion of failures)**, H0: exception probability = p = 1 %:
  `LR = -2 ln[(1-p)^(n-x) p^x] + 2 ln[(1-x/n)^(n-x) (x/n)^x]`, chi-square with 1 degree of freedom.
- **Christoffersen (independence)**, from the transition counts `n_ij` (state i yesterday,
  j today): `pi_0 = n01/(n00+n01)`, `pi_1 = n11/(n10+n11)`, `pi = (n01+n11)/N`;
  `LR = -2 [ln L(pi) - ln L(pi_0, pi_1)]`, chi-square(1). Conditional coverage = sum of both,
  chi-square(2).
- **Basel zones** for 250 observations at 99 %: green 0-4 exceptions, yellow 5-9, red 10+.
  Reported on the last 250 sessions and for the worst 250-session window.
- The P&L tested is today's book on past days (hypothetical, constant positions).

## 7. Stress tests (`stress.py`, `liquidity.py`, `scenarios.toml`)

- Hypothetical: per line, `new value = V x (1 + asset-class shock)(1 + zone shock)(1 + FX shock)`;
  cash pockets take the FX shock only. Output: loss in EUR, percent of wealth, per line.
- Historical: worst run of 5 consecutive sessions for today's book, over the whole sample
  (synthetic) or inside March 2020 and October 2008 (live).
- Liquidity: days to liquidate `q / (rate x ADV10)` at 20 % of volume, then 10 % and 5 %;
  portfolio score weighted by value over the lines with volume. Volume trend `ADV10 / ADV90`,
  drying up below 0.7.

## 8. Benchmark (`benchmark.py`)

- Brick weights = currency exposure of the securities, mapped to regional trackers
  (EUR, GBP, CHF and Nordics to Europe; JPY to Japan; HKD, SGD, KRW, TWD to Asia ex-Japan;
  USD to US; metal to Gold). Unmapped currencies are excluded and reported.
- Distributing trackers are turned into total return: `TR_t = TR_t-1 (P_t + D_t) / P_t-1`.
- Series aligned on a Monday-Friday calendar, gaps filled over at most 3 sessions.
  At least 60 observations.
- `beta = cov(r_P, r_B) / var(r_B)`, `alpha = (mean r_P - beta mean r_B) x 252` (simple
  annualisation), R squared, tracking error. Gross dividends in the book against net
  indices bias alpha upward; the bias is shown as a band of 15 % to 25 % of the dividend
  contribution.

## 9. Controls (`quality.py`)

- Price jump: a close within 5 % of 2, 3, 4, 5, 10, 20, 50, 100 times (or 1/x) the
  previous one looks like an unadjusted split. 3:2 is excluded, since a real crash of a
  third would match it.
- Freshness: lag in observed sessions; alert when an exchange peer has the missing
  session, information for a single session nobody on the exchange has.
- Session gaps: shortfall against the upper median of exchange peers above 15 %, for
  exchanges with at least five tickers.
- Quantities: a holding that moves by more than half, or disappears, between two runs is
  reported; a reported value more than 2 % away from the rebuilt value is reported.
