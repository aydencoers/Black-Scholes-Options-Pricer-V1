# Black-Scholes Options Pricing Calculator

A from-scratch Python implementation of the Black-Scholes-Merton model for
pricing European options, complete with all five major Greeks, an implied
volatility solver, correctness tests, and both a terminal demo and an
**interactive Streamlit web app** for exploring price, the Greeks, and
options strategy payoffs live.

Built as a portfolio project to demonstrate practical understanding of
options pricing theory and risk sensitivities — not just the ability to
call a library function.

**Live demo:** _add your Streamlit Community Cloud URL here after deploying
(see "Deploying the app" below)._

## What this project does

Given five inputs — the underlying price, strike price, time to expiration,
risk-free rate, and volatility (plus an optional continuous dividend
yield) — this project:

1. Prices a European call and put option.
2. Computes all five standard Greeks (Delta, Gamma, Theta, Vega, Rho) for
   both the call and put.
3. Solves backwards from a market price to the implied volatility, using
   Newton-Raphson with a bisection fallback.
4. Verifies its own correctness with sanity checks (put-call parity, and
   delta limits for deep in/out-of-the-money options).
5. Generates four charts showing how price and risk sensitivities move as
   the underlying price, volatility, and time change.
6. (Streamlit app) Lets you drag sliders for every input and watch the
   price, Greeks, and three interactive Plotly charts update live.
7. (Streamlit app) Lets you build a covered call, long straddle, or bull
   call spread and see the strategy's profit/loss diagram at expiration.
8. (Streamlit app) Pulls a **live option chain** for any ticker from Yahoo
   Finance, compares real market prices to the Black-Scholes model
   strike-by-strike, and plots the resulting **implied volatility smile**.

## The finance, briefly

**Black-Scholes-Merton model.** A closed-form formula for the fair value of
a European option (one that can only be exercised at expiration), derived
under the assumption that the underlying's price follows a log-normal
random walk with constant volatility, in a frictionless, no-arbitrage
market. It takes five inputs — spot price, strike, time to expiry,
risk-free rate, and volatility — and outputs a price. It's the baseline
every other options pricing model (binomial trees, local vol, stochastic
vol, jump-diffusion) is benchmarked against or extends.

**The Greeks.** The Greeks are the partial derivatives of the option price
with respect to each input — they answer "if the market moves a little,
how much does my position move, and in which direction?" **Delta** is the
hedge ratio (how much the option behaves like the underlying stock).
**Gamma** is how fast delta itself changes (how unstable the hedge is).
**Vega** is sensitivity to a change in implied volatility. **Theta** is
time decay — the value an option loses purely from time passing. **Rho**
is sensitivity to the risk-free rate. Together, they're the vocabulary
traders use to describe and manage the risk of an options position without
re-deriving the whole pricing formula every time the market moves.

**Implied volatility.** Black-Scholes normally takes volatility as an input
and produces a price. Implied volatility runs it backwards: given the
price the market is actually quoting, what volatility, fed into the
model, reproduces that price? It matters because the market trades
*prices*, not volatilities — implied vol is the standardized way to
compare how "expensive" options are across different strikes and
expirations, and it reflects the market's forward-looking expectation of
how much the underlying will move (the VIX is exactly this, computed on
S&P 500 options).

**Put-call parity.** A no-arbitrage identity — not a modeling assumption —
that must hold for European options regardless of pricing model:
`C - P = S*e^(-qT) - K*e^(-rT)`. It holds because a portfolio of
`long call + short put` has the exact same payoff, in every future state,
as `long stock - PV(strike)`. Two portfolios with identical future payoffs
must have the same price today, or there's a risk-free arbitrage. This
project uses it as a built-in correctness check on the call/put pricing
formulas.

**Model vs. market, and the volatility smile/skew.** Black-Scholes assumes
**one constant volatility** applies to every strike and every expiration
for a given underlying. The Model vs. Market tab tests that assumption
directly against reality: it pulls a real, live option chain, prices every
strike with the *same* single sigma, and lines the result up next to what
the market is actually paying. The two almost never match exactly — and
the pattern in the gap is informative, not just noise. For each strike,
the app also backs out the volatility the market's price *implies*
(reusing the same `implied_volatility()` solver as the Implied Volatility
tab) and plots implied vol against strike. If Black-Scholes' constant-sigma
assumption were literally true, that plot would be a flat line. In real
data it's curved — a pattern called the **volatility smile** (roughly
symmetric, higher implied vol at both tails — typical in FX options) or
the **volatility skew** (implied vol rises much more steeply on the
downside than the upside — the dominant pattern in equity options since
the 1987 crash). The reason: real option prices bake in **fat tails**
(extreme moves happen more often than a log-normal model predicts) and
**crash/jump risk** that smooth, constant-volatility diffusion doesn't
capture, plus heavy real-world demand for deep out-of-the-money puts as
portfolio insurance, which pushes those strikes' implied volatility up.
In short: the smile/skew is the market telling you constant volatility is
the wrong assumption, in a very specific, visualizable way — which is
exactly why more advanced models (local volatility, stochastic volatility
like Heston, jump-diffusion) exist to begin with.

**Strategy payoff diagrams.** A payoff diagram is a different question from
a Black-Scholes price: instead of "what is this worth *today*, before
expiration," it answers "what is this worth *at* expiration, as a function
of where the stock ends up." It's pure arithmetic on option payoffs
(`max(S-K, 0)` for a call, `max(K-S, 0)` for a put), with no volatility or
time-value assumptions involved. The app includes three classic
structures: a **covered call** (own the stock, sell a call against it —
caps upside, collects income, cushions a moderate decline), a **long
straddle** (buy a call and a put at the same strike — a pure bet that the
stock moves a lot, in either direction), and a **bull call spread** (buy a
call, sell a further out-of-the-money call — a cheaper, risk-defined
bullish bet with both profit and loss capped).

## Project structure

```
options-pricer/
├── black_scholes.py   # Core pricing engine: price, Greeks, implied vol solver (pure math, no plotting)
├── strategies.py      # Payoff-at-expiration math for covered call / straddle / spread (also pure math)
├── market_data.py     # yfinance wrapper: live spot price, expirations, option chains (pure data, no math)
├── visualize.py       # Matplotlib plotting functions for the terminal demo, built on black_scholes.py
├── demo.py            # Terminal entry point: prints results table + sanity checks, generates PNG charts
├── app.py             # Streamlit web app: sliders, live Greeks, Plotly charts, strategy payoffs, live market data
├── charts/            # Output directory for demo.py's generated PNG charts
├── requirements.txt
└── README.md
```

Note the layering: `app.py` and `visualize.py`/`demo.py` are two independent
presentation layers that both sit on top of the same `black_scholes.py`,
`strategies.py`, and `market_data.py` — neither the terminal version nor
the web app redefines any pricing, payoff, or data-fetching logic.

## How to run it

### Terminal demo

```bash
# From inside the options-pricer/ directory
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt

python demo.py
```

This prints a results table (price + all Greeks for a sample at-the-money
option), runs the sanity checks, and writes four PNG charts into `charts/`.

### Streamlit web app

```bash
# Same venv as above (it already has streamlit + plotly installed)
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`. It has four
tabs:

- **Pricer & Greeks** — sidebar sliders for S, K, T, r, sigma, and dividend
  yield, plus a call/put toggle. Price and all five Greeks are shown as
  live-updating metrics, with three interactive Plotly charts below
  (price vs. underlying, delta vs. underlying, and time decay).
- **Implied Volatility** — enter a market price and solve for the implied
  volatility, reusing the exact same solver from the terminal version.
- **Strategy Payoff** — pick a covered call, long straddle, or bull call
  spread, set the strikes/premiums, and see the profit/loss diagram at
  expiration, with max profit, max loss, and breakeven(s) called out.
- **Model vs. Market** — enter a ticker (e.g. `AAPL`, `SPY`) and pick a live
  expiration date to pull a real option chain from Yahoo Finance. Shows a
  strike-by-strike table of market price vs. Black-Scholes model price
  (using the sidebar's r, sigma, q) with the difference called out, a chart
  of model vs. market price by strike, and the implied volatility smile/skew
  backed out from the real market prices — with an explanation of why that
  curve isn't flat.

## Sample output

```
Results
------------------------------------------------------------
  Metric                            Call             Put
  ------------------------------------------------------
  Price                          10.4506          5.5735
  Delta                           0.6368         -0.3632
  Gamma                           0.0188          0.0188
  Vega (per 1% vol)               0.3752          0.3752
  Theta (per day)                -0.0176         -0.0045
  Rho (per 1% rate)               0.5323         -0.4189
------------------------------------------------------------

Sanity Checks
------------------------------------------------------------
  [PASS] Put-call parity gap: 0.00e+00 (should be ~0)
  [PASS] Deep ITM call delta: 1.000000 (should be ~1.0)
  [PASS] Deep OTM call delta: 0.000000 (should be ~0.0)
  [PASS] Implied vol round-trip: solved 0.3500, true 0.3500 (diff 5.55e-17)
------------------------------------------------------------
```

(Sample option: S=100, K=100, T=1 year, r=5%, sigma=20%, no dividend.)

## Charts generated

All saved to `charts/` as PNG files:

- **`price_vs_underlying.png`** — Call and put price across a range of
  underlying prices. Shows the characteristic convex (curved) shape of
  option value — the visual signature of positive gamma.
- **`delta_vs_underlying.png`** — Call and put delta across a range of
  underlying prices. Shows the hedge ratio sliding from 0 to 1 (calls) or
  -1 to 0 (puts) as the option moves from out-of-the-money to
  in-the-money.
- **`time_decay_call.png`** — Call value as time to expiration shrinks to
  zero, holding everything else fixed. Shows that time decay accelerates
  as expiration nears, rather than eroding linearly.
- **`vega_effect.png`** — Call and put price across a range of
  volatilities. Shows both option types gaining value as volatility rises,
  reflecting the asymmetric (capped downside, open-ended upside) payoff
  structure of options.

## Correctness checks, and what they prove

| Check | What it proves |
|---|---|
| Put-call parity gap ≈ 0 | The call and put pricing formulas are internally consistent with the no-arbitrage relationship between them — a bug in either formula would very likely break this. |
| Deep in-the-money call delta → 1 | Delta behaves correctly in the limit: an option that's virtually certain to be exercised should move almost 1-for-1 with the stock, just like owning the shares outright. |
| Deep out-of-the-money call delta → 0 | The mirror case: an option that's virtually certain to expire worthless should barely respond to small stock moves. |
| Implied volatility round-trip | Prices a call at a known volatility, feeds that price back into the implied vol solver, and confirms it recovers the same volatility — proving the solver is a correct numerical inverse of the pricing function. |

## Known model limitations (worth knowing for discussion)

- **Constant volatility assumption.** Real markets exhibit a "volatility
  smile/skew" — implied volatility differs across strikes and expirations
  for the same underlying, which directly contradicts Black-Scholes'
  assumption of one constant sigma. This is why more advanced models
  (local volatility, stochastic volatility like Heston, jump-diffusion)
  exist.
- **European exercise only.** This model does not price American options
  (which can be exercised any time before expiration) — those generally
  require numerical methods like binomial/trinomial trees or finite
  difference methods, especially when early exercise can be optimal (e.g.
  American calls on dividend-paying stocks).
- **No transaction costs or liquidity effects.** The model assumes a
  frictionless market; real hedging has costs that these formulas ignore.
- **Live market data is best-effort, not exchange-grade.** The Model vs.
  Market tab pulls quotes through `yfinance`, an unofficial wrapper around
  Yahoo Finance's public endpoints — prices can be delayed, and thinly
  traded strikes can have stale or wide bid/ask quotes (the implied
  volatility solver simply skips strikes where no valid solution exists,
  rather than showing a garbage number).

## Deploying the app (Streamlit Community Cloud)

Streamlit Community Cloud hosts the app for free, straight from a GitHub
repo, and gives you a shareable `*.streamlit.app` link — handy for putting
a live link on a resume or in an interview follow-up email.

1. Push this project to a GitHub repository (public, or private on a plan
   that supports it):
   ```bash
   git init
   git add .
   git commit -m "Black-Scholes options pricer with Streamlit app"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
   (`.gitignore` already excludes `venv/` and `__pycache__/`, so the repo
   stays small.)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
3. Click **"New app"**, select this repository and the `main` branch, and
   set the **main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud reads `requirements.txt` automatically
   and installs everything needed.
5. Once it's live, copy the URL (e.g.
   `https://<something>-options-pricer.streamlit.app`) and paste it into
   the "Live demo" line at the top of this README.

Any time you `git push` a change to `main`, the deployed app redeploys
automatically.

## Tech stack

- Python 3.9+
- NumPy — vectorized math
- SciPy — normal distribution functions (`scipy.stats.norm`)
- Matplotlib — static charts for the terminal demo
- Streamlit — interactive web app framework
- Plotly — interactive charts inside the Streamlit app
- pandas — tabular data for the option chain / model-vs-market comparison
- yfinance — live option chain and spot price data from Yahoo Finance
