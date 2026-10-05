"""
black_scholes.py
-----------------
Core pricing engine for European options under the Black-Scholes-Merton model.

This module contains ONLY math: option prices, the Greeks, and an implied
volatility solver. No plotting, no printing, no CLI — that separation is
intentional (a "core library" vs. "presentation layer" split), so the pricing
logic can be tested and reused independently of how it's displayed.

Model assumptions (good to know cold for an interview):
    - The underlying follows geometric Brownian motion (log-normal prices,
      constant volatility).
    - No arbitrage, frictionless markets (no transaction costs/taxes).
    - Constant risk-free rate and volatility over the option's life.
    - The option is European: it can only be exercised at expiration
      (as opposed to American options, which can be exercised any time).
    - The underlying can pay a continuous dividend yield q (0 if none).
"""

from __future__ import annotations

from dataclasses import dataclass
from scipy.stats import norm
import numpy as np


# ---------------------------------------------------------------------------
# Inputs container
# ---------------------------------------------------------------------------

@dataclass
class OptionInputs:
    """
    Bundles the five (or six) Black-Scholes inputs in one place.

    S: spot price of the underlying today.
    K: strike price (the price you can buy/sell at, if you exercise).
    T: time to expiration, in YEARS (e.g. 0.5 = 6 months).
    r: risk-free interest rate, as a decimal (e.g. 0.05 = 5%).
    sigma: volatility of the underlying, as a decimal annualized number
           (e.g. 0.20 = 20%). This is the market's *expected* future
           volatility, not a historical average.
    q: continuous dividend yield, as a decimal. Defaults to 0 (no dividend).
       Interview point: dividends make calls cheaper and puts more
       expensive, because holding the dividend-paying stock has a yield
       advantage over holding the call.
    """
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0


# ---------------------------------------------------------------------------
# Internal helper: the famous d1 / d2 terms
# ---------------------------------------------------------------------------

def _d1_d2(inputs: OptionInputs) -> tuple[float, float]:
    """
    Computes the intermediate d1 and d2 terms used throughout Black-Scholes.

    Plain-language meaning:
        d1 and d2 are standardized ("z-score style") measures of how far
        in-the-money the option is expected to be, adjusted for volatility
        and time. They don't have a clean one-sentence intuition on their
        own, but N(d2) turns out to be (risk-neutral) probability the option
        is exercised, and N(d1) is used to build Delta. You will almost
        certainly be asked "what is d1/d2" in an interview — the honest
        answer is: they fall out of solving the Black-Scholes PDE, and the
        useful takeaway is what N(d1) and N(d2) represent, not deriving them
        from scratch.

    Formulas:
        d1 = [ln(S/K) + (r - q + sigma^2 / 2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
    """
    S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q

    if T <= 0 or sigma <= 0:
        raise ValueError("T (time to expiry) and sigma (volatility) must both be > 0.")

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


# ---------------------------------------------------------------------------
# Pricing functions
# ---------------------------------------------------------------------------

def call_price(inputs: OptionInputs) -> float:
    """
    Price of a European CALL option (the right, not obligation, to BUY the
    underlying at strike K at expiration).

    Formula:
        C = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)

    Intuition: you're comparing the present value of "what you'd receive if
    you exercise" (the discounted stock, weighted by the probability-ish
    term N(d1)) against the present value of "what you'd pay" (the
    discounted strike, weighted by N(d2), the risk-neutral probability of
    finishing in-the-money).
    """
    S, K, T, r, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.q
    d1, d2 = _d1_d2(inputs)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(inputs: OptionInputs) -> float:
    """
    Price of a European PUT option (the right, not obligation, to SELL the
    underlying at strike K at expiration).

    Formula:
        P = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)

    This can be derived directly from the call price via put-call parity
    (see put_call_parity_check below) rather than treated as a separate
    model — that relationship is itself a common interview question.
    """
    S, K, T, r, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.q
    d1, d2 = _d1_d2(inputs)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def price(inputs: OptionInputs, option_type: str) -> float:
    """Dispatch helper: price('call'|'put'). Keeps call sites readable."""
    option_type = option_type.lower()
    if option_type == "call":
        return call_price(inputs)
    elif option_type == "put":
        return put_price(inputs)
    raise ValueError("option_type must be 'call' or 'put'")


# ---------------------------------------------------------------------------
# The Greeks
# ---------------------------------------------------------------------------
# The Greeks are partial derivatives of the option price with respect to each
# input. In practice, a trader uses them to answer: "if the market moves a
# little, how much does my position's value move, and in which direction?"
# They are the language of risk management on an options desk.

def delta(inputs: OptionInputs, option_type: str) -> float:
    """
    Delta: how much the option price changes for a $1 move in the underlying.

    Plain language: Delta is the "hedge ratio". If a call has delta 0.60,
    buying 1 call behaves (for small moves) like owning 60 shares of stock.
    A market maker who sold that call would buy 60 shares to be
    "delta-neutral" (hedged against small price moves). This hedging
    activity (delta hedging) is itself a major interview topic.

    Range: call delta is in [0, 1]; put delta is in [-1, 0].
    A deep in-the-money call's delta approaches 1 (it behaves just like the
    stock); a deep out-of-the-money call's delta approaches 0.
    """
    T, q = inputs.T, inputs.q
    d1, _ = _d1_d2(inputs)
    if option_type.lower() == "call":
        return float(np.exp(-q * T) * norm.cdf(d1))
    elif option_type.lower() == "put":
        return float(np.exp(-q * T) * (norm.cdf(d1) - 1))
    raise ValueError("option_type must be 'call' or 'put'")


def gamma(inputs: OptionInputs) -> float:
    """
    Gamma: how fast Delta itself changes for a $1 move in the underlying.
    (The second derivative of price with respect to the underlying.)

    Plain language: Gamma measures how "unstable" your hedge ratio is.
    High gamma (common near-the-money, close to expiration) means Delta
    can swing wildly with small price moves, so a delta-hedged position
    needs to be rebalanced often. Gamma is identical for calls and puts
    with the same strike/expiry (a direct consequence of put-call parity,
    since Delta_call - Delta_put = 1, a constant, so their slopes match).
    """
    S, T, sigma, q = inputs.S, inputs.T, inputs.sigma, inputs.q
    d1, _ = _d1_d2(inputs)
    return float(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def vega(inputs: OptionInputs) -> float:
    """
    Vega: how much the option price changes for a 1.00 (i.e. 100 percentage
    point) change in volatility. In practice people read it per 1% move,
    so we also report vega / 100 in the demo table.

    Plain language: Vega tells you how exposed the option is to changes in
    the market's *expectation* of future volatility (not realized moves).
    Vega is identical for calls and puts with the same strike/expiry (again
    a consequence of put-call parity). Long options = long vega (you want
    volatility to rise); short options = short vega.
    """
    S, T, q = inputs.S, inputs.T, inputs.q
    d1, _ = _d1_d2(inputs)
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def theta(inputs: OptionInputs, option_type: str) -> float:
    """
    Theta: how much the option price changes per YEAR as time passes, all
    else equal (we also report a per-day figure, dividing by 365, since
    that's how traders usually think about "time decay").

    Plain language: Theta is "time decay" — the option is a wasting asset,
    and (usually) loses value every day purely from the clock ticking,
    holding the stock price and volatility fixed. Long options = negative
    theta (you pay for time decay); short options = positive theta (you
    collect it). This is the key trade-off against Gamma/Vega: you are
    paying theta to be long gamma/vega, or collecting theta while being
    short gamma/vega. That trade-off is one of the most common options
    interview talking points.
    """
    S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q
    d1, d2 = _d1_d2(inputs)

    term1 = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type.lower() == "call":
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        term3 = q * S * np.exp(-q * T) * norm.cdf(d1)
        return float(term1 + term2 + term3)
    elif option_type.lower() == "put":
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
        term3 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
        return float(term1 + term2 + term3)
    raise ValueError("option_type must be 'call' or 'put'")


def rho(inputs: OptionInputs, option_type: str) -> float:
    """
    Rho: how much the option price changes for a 1.00 (100 percentage point)
    change in the risk-free interest rate. Like vega, usually read per 1%,
    so the demo also reports rho / 100.

    Plain language: Rho is usually the least-watched Greek day-to-day
    (rates move slowly compared to stock prices), but it matters a lot for
    long-dated options (LEAPS) and is directly relevant whenever interest
    rate expectations shift. Calls have positive rho (higher rates raise
    call value: deferring the strike payment is worth more when rates are
    higher); puts have negative rho.
    """
    K, T, r = inputs.K, inputs.T, inputs.r
    _, d2 = _d1_d2(inputs)
    if option_type.lower() == "call":
        return float(K * T * np.exp(-r * T) * norm.cdf(d2))
    elif option_type.lower() == "put":
        return float(-K * T * np.exp(-r * T) * norm.cdf(-d2))
    raise ValueError("option_type must be 'call' or 'put'")


def all_greeks(inputs: OptionInputs, option_type: str) -> dict:
    """Convenience bundle: returns every Greek for one option in a dict."""
    return {
        "delta": delta(inputs, option_type),
        "gamma": gamma(inputs),
        "vega": vega(inputs),
        "theta": theta(inputs, option_type),
        "rho": rho(inputs, option_type),
    }


# ---------------------------------------------------------------------------
# Implied volatility solver
# ---------------------------------------------------------------------------
# Black-Scholes normally takes volatility as an INPUT and gives you a price.
# Implied volatility runs that backwards: given the price the market is
# actually trading at, what volatility input, fed into Black-Scholes, would
# reproduce that price?
#
# Why it matters: the market quotes option PRICES, not volatilities. Traders
# convert prices to implied vol because it's a standardized, comparable way
# to talk about "how expensive" an option is across different strikes,
# expirations, and underlyings. It's also the market's forward-looking
# estimate of how much the underlying will move — which is why "IV" is
# quoted constantly in options trading (e.g. the VIX is implied vol on the
# S&P 500). A classic interview point: real markets show a "volatility
# smile/skew" — implied vol is NOT constant across strikes, which directly
# violates the Black-Scholes assumption of constant sigma. That gap between
# model and market is a huge chunk of what quant/derivatives interviews
# probe.

def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    q: float = 0.0,
    tol: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """
    Solves for the volatility (sigma) that makes the Black-Scholes price
    equal to the observed market price, using Newton-Raphson with a
    bisection fallback.

    How Newton-Raphson works here, in plain terms:
        1. Start with a guess for sigma (we use 0.20, a reasonable default).
        2. Compute the model price at that guess, and compare it to the
           real market price (the "error").
        3. Use vega (the derivative of price w.r.t. sigma) to estimate how
           much to adjust sigma to close that error, like walking downhill
           using the local slope.
        4. Repeat until the error is smaller than `tol`, or give up after
           `max_iterations`.

    Newton-Raphson converges very fast (few iterations) but can fail if
    vega is tiny (deep in/out-of-the-money, or very short-dated options) or
    the first guess is poor — the step can overshoot into a nonsensical
    (negative) sigma. As a safety net, if Newton-Raphson doesn't converge,
    we fall back to bisection: start with a wide bracket of plausible
    sigmas (e.g. 0.001 to 5.0, i.e. 0.1% to 500% annualized vol) and
    repeatedly halve the bracket, keeping whichever half still contains the
    root. Bisection is slower but guaranteed to converge as long as the
    market price is arbitrage-free (between the option's intrinsic value
    and its max theoretical value).
    """
    option_type = option_type.lower()

    def model_price(sigma: float) -> float:
        inp = OptionInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        return call_price(inp) if option_type == "call" else put_price(inp)

    # --- Attempt 1: Newton-Raphson ---
    sigma = 0.20  # a reasonable starting guess (20% annualized vol)
    for _ in range(max_iterations):
        inp = OptionInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        model = model_price(sigma)
        diff = model - market_price

        if abs(diff) < tol:
            return sigma

        v = vega(inp)  # d(price)/d(sigma)
        if v < 1e-10:
            break  # vega too small, Newton-Raphson step is unreliable -> fall back

        sigma = sigma - diff / v
        if sigma <= 0:
            break  # stepped into an invalid region -> fall back

    # --- Attempt 2: Bisection fallback (robust, slower) ---
    lo, hi = 1e-4, 5.0
    price_lo = model_price(lo) - market_price
    price_hi = model_price(hi) - market_price

    if price_lo * price_hi > 0:
        raise ValueError(
            "No implied volatility found in [0.01%, 500%]. "
            "The market price may violate no-arbitrage bounds for these inputs."
        )

    for _ in range(max_iterations):
        mid = (lo + hi) / 2
        price_mid = model_price(mid) - market_price

        if abs(price_mid) < tol:
            return mid

        if price_lo * price_mid < 0:
            hi = mid
            price_hi = price_mid
        else:
            lo = mid
            price_lo = price_mid

    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Sanity checks (used by tests / demo)
# ---------------------------------------------------------------------------

def put_call_parity_gap(inputs: OptionInputs) -> float:
    """
    Put-call parity is a no-arbitrage identity (NOT a model assumption —
    it holds for European options regardless of what pricing model you
    use, because it's derived purely from a replication argument):

        C - P = S * e^(-qT) - K * e^(-rT)

    Why it must hold: a portfolio of [long call + short put] has exactly
    the same payoff at expiration as [long stock - PV(strike)], for ANY
    stock price. Two portfolios with identical payoffs in every future
    state must have the same price today, or there's a free-money arbitrage
    (buy the cheap side, sell the expensive side, pocket the difference
    risk-free). This function returns the gap between the two sides; it
    should be ~0 (down to floating point error) for Black-Scholes prices,
    since our call and put formulas were both derived from the same model.
    """
    C = call_price(inputs)
    P = put_price(inputs)
    lhs = C - P
    rhs = inputs.S * np.exp(-inputs.q * inputs.T) - inputs.K * np.exp(-inputs.r * inputs.T)
    return lhs - rhs
