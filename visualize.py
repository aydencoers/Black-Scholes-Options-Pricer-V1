"""
visualize.py
------------
Plotting layer. Everything in here takes core pricing functions from
black_scholes.py and turns them into matplotlib charts. No new finance math
lives here — just sweeping one input across a range and plotting the result.

Kept deliberately separate from black_scholes.py: the pricing math shouldn't
know or care how (or whether) it gets charted.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from black_scholes import OptionInputs, call_price, put_price, delta, theta, vega


# A small shared style so every chart looks consistent and "professional"
# rather than matplotlib's raw defaults.
def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"

CALL_COLOR = "#1f77b4"   # blue
PUT_COLOR = "#d62728"    # red


def plot_price_vs_underlying(
    base: OptionInputs,
    s_range: tuple[float, float] = (0.5, 1.5),
    n_points: int = 200,
    save_path: str | None = None,
):
    """
    Chart 1: Option price vs. underlying price (S), for both call and put,
    holding K, T, r, sigma, q fixed at `base`'s values.

    Why this matters: this is the most basic "payoff-like" view of an
    option, except unlike a payoff diagram (which only shows value AT
    expiration), this shows the option's current theoretical value across
    a range of spot prices — so you can see the curve's convexity (the
    option price bends, it doesn't move 1-for-1 with the stock), which is
    the visual signature of Gamma.
    """
    s_values = np.linspace(base.S * s_range[0], base.S * s_range[1], n_points)
    call_prices = []
    put_prices = []
    for s in s_values:
        inp = OptionInputs(S=s, K=base.K, T=base.T, r=base.r, sigma=base.sigma, q=base.q)
        call_prices.append(call_price(inp))
        put_prices.append(put_price(inp))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_values, call_prices, label="Call price", color=CALL_COLOR, linewidth=2)
    ax.plot(s_values, put_prices, label="Put price", color=PUT_COLOR, linewidth=2)
    ax.axvline(base.K, color="gray", linestyle=":", linewidth=1, label=f"Strike (K={base.K:g})")
    _style_axes(
        ax,
        title="Option Price vs. Underlying Price",
        xlabel="Underlying price (S)",
        ylabel="Option price",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_delta_vs_underlying(
    base: OptionInputs,
    s_range: tuple[float, float] = (0.5, 1.5),
    n_points: int = 200,
    save_path: str | None = None,
):
    """
    Chart 2: Delta vs. underlying price, for both call and put.

    Why this matters: this is the clearest way to SEE the hedge ratio
    shift. Near expiration / near the strike, delta moves from ~0 to ~1
    (calls) very sharply (that sharp S-curve slope IS gamma). Far in- or
    out-of-the-money, delta flattens out near 1, 0, or -1 — the option
    starts behaving like (or like not holding) the stock at all.
    """
    s_values = np.linspace(base.S * s_range[0], base.S * s_range[1], n_points)
    call_deltas = []
    put_deltas = []
    for s in s_values:
        inp = OptionInputs(S=s, K=base.K, T=base.T, r=base.r, sigma=base.sigma, q=base.q)
        call_deltas.append(delta(inp, "call"))
        put_deltas.append(delta(inp, "put"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(s_values, call_deltas, label="Call delta", color=CALL_COLOR, linewidth=2)
    ax.plot(s_values, put_deltas, label="Put delta", color=PUT_COLOR, linewidth=2)
    ax.axvline(base.K, color="gray", linestyle=":", linewidth=1, label=f"Strike (K={base.K:g})")
    ax.axhline(0, color="black", linewidth=0.8)
    _style_axes(
        ax,
        title="Delta vs. Underlying Price (Hedge Ratio)",
        xlabel="Underlying price (S)",
        ylabel="Delta",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_time_decay(
    base: OptionInputs,
    option_type: str = "call",
    n_points: int = 200,
    save_path: str | None = None,
):
    """
    Chart 3: Option value vs. time remaining to expiration, holding S, K, r,
    sigma, q fixed at `base`'s values (this is "theta decay" made visible).

    Why this matters: this shows WHY options are called "wasting assets".
    Time decay is not linear — it accelerates as expiration approaches,
    especially for at-the-money options. That acceleration is the
    practical reason "theta burn" gets worse the closer you get to
    expiry, and it's why many strategies (e.g. selling options) are
    specifically designed to collect that accelerating decay.
    """
    # Sweep time-to-expiry from just-created (near base.T) down to just
    # before expiration. We walk forward in calendar time, so the x-axis
    # is "time remaining", decreasing left to right toward 0.
    t_values = np.linspace(base.T, 1e-4, n_points)
    values = []
    for t in t_values:
        inp = OptionInputs(S=base.S, K=base.K, T=t, r=base.r, sigma=base.sigma, q=base.q)
        price_fn = call_price if option_type.lower() == "call" else put_price
        values.append(price_fn(inp))

    fig, ax = plt.subplots(figsize=(8, 5))
    color = CALL_COLOR if option_type.lower() == "call" else PUT_COLOR
    ax.plot(t_values, values, color=color, linewidth=2, label=f"{option_type.capitalize()} value")
    ax.invert_xaxis()  # time counts DOWN toward expiration, left -> right
    _style_axes(
        ax,
        title=f"Time Decay: {option_type.capitalize()} Value as Expiration Approaches",
        xlabel="Time remaining to expiration (years)",
        ylabel="Option price",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_vega_effect(
    base: OptionInputs,
    sigma_range: tuple[float, float] = (0.05, 0.80),
    n_points: int = 200,
    save_path: str | None = None,
):
    """
    Chart 4: Option price vs. volatility, for both call and put, holding
    S, K, T, r, q fixed.

    Why this matters: both calls and puts get MORE valuable as volatility
    rises (this is the "long vega" picture) — more volatility means a
    wider range of possible future stock prices, and since option payoffs
    are asymmetric (capped downside at zero, unlimited/large upside), more
    spread in outcomes is pure upside for the option holder. This is a
    core reason option prices rose across the board during, e.g., market
    stress events, even if the stock itself didn't move much yet.
    """
    sigma_values = np.linspace(sigma_range[0], sigma_range[1], n_points)
    call_prices = []
    put_prices = []
    for sig in sigma_values:
        inp = OptionInputs(S=base.S, K=base.K, T=base.T, r=base.r, sigma=sig, q=base.q)
        call_prices.append(call_price(inp))
        put_prices.append(put_price(inp))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sigma_values * 100, call_prices, label="Call price", color=CALL_COLOR, linewidth=2)
    ax.plot(sigma_values * 100, put_prices, label="Put price", color=PUT_COLOR, linewidth=2)
    ax.axvline(base.sigma * 100, color="gray", linestyle=":", linewidth=1,
               label=f"Base vol ({base.sigma * 100:.0f}%)")
    _style_axes(
        ax,
        title="Option Price Sensitivity to Volatility (Vega Effect)",
        xlabel="Volatility, annualized (%)",
        ylabel="Option price",
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
