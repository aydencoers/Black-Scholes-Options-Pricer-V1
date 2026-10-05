"""
strategies.py
-------------
Payoff-at-expiration math for common options strategies.

Important distinction from black_scholes.py: this module does NOT use the
Black-Scholes model at all. These are intrinsic payoff diagrams — "if the
stock ends up at price X at expiration, what is this position worth?" —
which is pure arithmetic on option payoffs, independent of any pricing
model. Black-Scholes (and its Greeks) tell you what a position is worth
*today*, before expiration; a payoff diagram tells you what it's worth
*at* expiration, as a function of the final stock price.

Knowing this distinction cold is a good interview point: pricing models and
payoff diagrams answer different questions, and mixing them up is a common
beginner mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


def _call_payoff(S_at_expiry: np.ndarray, K: float) -> np.ndarray:
    """Intrinsic value of a call at expiration: max(S - K, 0)."""
    return np.maximum(S_at_expiry - K, 0.0)


def _put_payoff(S_at_expiry: np.ndarray, K: float) -> np.ndarray:
    """Intrinsic value of a put at expiration: max(K - S, 0)."""
    return np.maximum(K - S_at_expiry, 0.0)


@dataclass
class StrategyResult:
    """Container for a strategy's P&L curve plus a short label for the legend."""
    s_range: np.ndarray
    pnl: np.ndarray
    breakevens: list[float] = field(default_factory=list)
    max_profit: float | None = None   # None means "unlimited"
    max_loss: float | None = None     # None means "unlimited"


def covered_call(
    S0: float,
    K_call: float,
    call_premium: float,
    s_range: np.ndarray,
) -> StrategyResult:
    """
    Covered call: own 100 shares of stock (bought at S0) + sell 1 call at
    strike K_call, collecting `call_premium` today.

    Why traders use it: income generation on a stock you already plan to
    hold. You cap your upside at K_call (if the stock rallies past the
    strike, the call gets exercised against you and caps your gain), but
    you lower your breakeven by the premium collected, cushioning a
    moderate decline.

    P&L at expiration = (S_T - S0)            [stock P&L]
                       + call_premium          [premium collected]
                       - max(S_T - K_call, 0)  [obligation from the short call]
    """
    stock_pnl = s_range - S0
    short_call_pnl = call_premium - _call_payoff(s_range, K_call)
    pnl = stock_pnl + short_call_pnl

    max_profit = (K_call - S0) + call_premium  # capped: stock gain up to strike + premium
    max_loss = -(S0 - call_premium)  # worst case is S_T -> 0: lose the stock's value, net of premium collected
    breakeven = S0 - call_premium

    return StrategyResult(s_range=s_range, pnl=pnl, breakevens=[breakeven],
                           max_profit=max_profit, max_loss=max_loss)


def long_straddle(
    K: float,
    call_premium: float,
    put_premium: float,
    s_range: np.ndarray,
) -> StrategyResult:
    """
    Long straddle: buy 1 call AND 1 put at the SAME strike K, same
    expiration.

    Why traders use it: a pure bet on volatility / a big move, with no
    directional view — you profit if the stock moves far enough in EITHER
    direction to cover the combined premium paid. Max loss is capped at
    the total premium (if the stock sits exactly at K at expiration, both
    options expire worthless). This is the strategy to reach for when you
    expect a large move (e.g. around earnings) but don't know which way.

    P&L at expiration = max(S_T - K, 0) + max(K - S_T, 0) - (call_premium + put_premium)
    """
    total_premium = call_premium + put_premium
    pnl = _call_payoff(s_range, K) + _put_payoff(s_range, K) - total_premium

    max_loss = -total_premium
    max_profit = None  # unlimited on the upside (stock can rise indefinitely)
    breakevens = [K - total_premium, K + total_premium]

    return StrategyResult(s_range=s_range, pnl=pnl, breakevens=breakevens,
                           max_profit=max_profit, max_loss=max_loss)


def bull_call_spread(
    K_low: float,
    K_high: float,
    premium_low: float,
    premium_high: float,
    s_range: np.ndarray,
) -> StrategyResult:
    """
    Bull call spread (a "vertical spread"): buy 1 call at the lower strike
    K_low, sell 1 call at the higher strike K_high (same expiration).
    Requires K_high > K_low.

    Why traders use it: a cheaper, risk-defined way to bet on a moderate
    rally. Selling the higher-strike call reduces the cost of the position
    (vs. just buying a call outright) in exchange for capping the upside
    at K_high. Both max profit and max loss are known and limited up
    front — a common way to express a bullish view with controlled risk
    (and lower vega exposure than an outright long call).

    P&L at expiration = max(S_T - K_low, 0) - max(S_T - K_high, 0) - net_debit
    where net_debit = premium_low - premium_high (what you pay, net, to put the trade on).
    """
    if K_high <= K_low:
        raise ValueError("K_high must be greater than K_low for a bull call spread.")

    net_debit = premium_low - premium_high
    pnl = _call_payoff(s_range, K_low) - _call_payoff(s_range, K_high) - net_debit

    max_profit = (K_high - K_low) - net_debit
    max_loss = -net_debit
    breakeven = K_low + net_debit

    return StrategyResult(s_range=s_range, pnl=pnl, breakevens=[breakeven],
                           max_profit=max_profit, max_loss=max_loss)
