"""
app.py
------
Streamlit web app for the Black-Scholes options pricer.

This file contains ONLY UI/presentation code: it imports the pricing math
from black_scholes.py and the payoff math from strategies.py and never
redefines or duplicates any formula. That's deliberate — the same
separation of "core math" vs. "presentation" from the terminal version,
just with Streamlit + Plotly standing in for print statements + matplotlib.

Visual design: dark "terminal" theme (navy background, one green/red
accent pair, monospaced numerics) configured in .streamlit/config.toml
(native widget theme) plus the CSS block below (custom components: the
metric panel, section labels, header).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from black_scholes import (
    OptionInputs,
    call_price,
    put_price,
    all_greeks,
    implied_volatility,
    delta as delta_fn,
)
from strategies import covered_call, long_straddle, bull_call_spread
import market_data
from market_data import MarketDataError


# ---------------------------------------------------------------------------
# Palette — defined once, used everywhere (charts, CSS, metric coloring)
# ---------------------------------------------------------------------------
BG = "#0A0E1A"
PANEL_BG = "#141927"
BORDER = "rgba(255,255,255,0.08)"
GRID = "rgba(255,255,255,0.07)"
TEXT = "#E6E9EF"
SLATE = "#8B94A7"
GREEN = "#00C896"
RED = "#E74C3C"

FONT_SANS = "Inter, -apple-system, sans-serif"
FONT_MONO = "IBM Plex Mono, monospace"


# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Black-Scholes Options Pricer", layout="wide")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: {FONT_SANS} !important;
    }}

    /* strip default Streamlit chrome */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    header {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.75rem; padding-bottom: 2rem; max-width: 1180px; }}

    /* sidebar */
    [data-testid="stSidebar"] {{
        border-right: 1px solid {BORDER};
    }}

    /* numeric input fields render in mono, like a terminal */
    input, textarea {{
        font-family: {FONT_MONO} !important;
    }}
    [class*="ThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{
        font-family: {FONT_MONO} !important;
    }}

    /* ---- header ---- */
    .app-title {{
        font-family: {FONT_SANS};
        font-weight: 700;
        font-size: 1.65rem;
        color: {TEXT};
        letter-spacing: -0.01em;
        margin-bottom: 0.1rem;
    }}
    .app-subtitle {{
        font-family: {FONT_SANS};
        font-size: 0.92rem;
        color: {SLATE};
        margin-bottom: 1.4rem;
    }}

    /* ---- section + sidebar labels (no eyebrow caps, no dividers) ---- */
    .section-label {{
        font-family: {FONT_SANS};
        font-weight: 600;
        font-size: 0.95rem;
        color: {TEXT};
        margin: 1.75rem 0 0.6rem 0;
    }}
    .sidebar-title {{
        font-family: {FONT_SANS};
        font-weight: 600;
        font-size: 0.95rem;
        color: {TEXT};
        margin-bottom: 0.9rem;
    }}

    /* ---- metric panel: vertical label/value rows ---- */
    .metric-panel {{
        border: 1px solid {BORDER};
        border-radius: 3px;
        background: {PANEL_BG};
        overflow: hidden;
    }}
    .metric-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 1rem;
        border-bottom: 1px solid {BORDER};
    }}
    .metric-row:last-child {{ border-bottom: none; }}
    .metric-label {{
        font-family: {FONT_SANS};
        font-size: 0.85rem;
        color: {SLATE};
    }}
    .metric-value {{
        font-family: {FONT_MONO};
        font-size: 0.98rem;
        font-weight: 500;
    }}
    .metric-value.positive {{ color: {GREEN}; }}
    .metric-value.negative {{ color: {RED}; }}
    .metric-value.neutral {{ color: {TEXT}; }}

    /* ---- metric strip: horizontal cells (strategy summary) ---- */
    .metric-strip {{
        display: flex;
        border: 1px solid {BORDER};
        border-radius: 3px;
        background: {PANEL_BG};
        overflow: hidden;
    }}
    .strip-cell {{
        flex: 1;
        padding: 0.8rem 1.1rem;
        border-right: 1px solid {BORDER};
    }}
    .strip-cell:last-child {{ border-right: none; }}
    .strip-label {{
        font-family: {FONT_SANS};
        font-size: 0.8rem;
        color: {SLATE};
        margin-bottom: 0.3rem;
    }}
    .strip-value {{
        font-family: {FONT_MONO};
        font-size: 1.1rem;
        font-weight: 600;
    }}
    .strip-value.positive {{ color: {GREEN}; }}
    .strip-value.negative {{ color: {RED}; }}
    .strip-value.neutral {{ color: {TEXT}; }}

    /* ---- implied vol result box ---- */
    .iv-result {{
        border-left: 3px solid {GREEN};
        background: {PANEL_BG};
        padding: 0.75rem 1.1rem;
        font-family: {FONT_SANS};
        color: {TEXT};
        border-radius: 2px;
        margin-top: 0.75rem;
        font-size: 0.92rem;
    }}
    .iv-result.error {{ border-left-color: {RED}; }}
    .iv-value {{
        font-family: {FONT_MONO};
        font-weight: 600;
        color: {GREEN};
    }}

    .position-line {{
        font-family: {FONT_SANS};
        color: {SLATE};
        font-size: 0.85rem;
        margin-bottom: 0.9rem;
    }}
    .position-line .mono {{ font-family: {FONT_MONO}; color: {TEXT}; }}

    [data-testid="stCaptionContainer"] {{
        color: {SLATE} !important;
        font-family: {FONT_SANS} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Small rendering helpers (HTML components replacing st.metric / st.subheader)
# ---------------------------------------------------------------------------

def section_label(text: str) -> None:
    """A quiet section heading — no icon, no eyebrow caps, no horizontal rule."""
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


def sign_class(value: float) -> str:
    if value > 1e-12:
        return "positive"
    if value < -1e-12:
        return "negative"
    return "neutral"


def render_metric_panel(rows: list[tuple[str, str, str]]) -> None:
    """rows: (label, formatted value, sign class 'positive'|'negative'|'neutral')."""
    cells = "".join(
        f'<div class="metric-row">'
        f'<span class="metric-label">{label}</span>'
        f'<span class="metric-value {sign}">{value}</span>'
        f"</div>"
        for label, value, sign in rows
    )
    st.markdown(f'<div class="metric-panel">{cells}</div>', unsafe_allow_html=True)


def render_metric_strip(cells: list[tuple[str, str, str]]) -> None:
    """cells: (label, formatted value, sign class) laid out horizontally."""
    html = "".join(
        f'<div class="strip-cell">'
        f'<div class="strip-label">{label}</div>'
        f'<div class="strip-value {sign}">{value}</div>'
        f"</div>"
        for label, value, sign in cells
    )
    st.markdown(f'<div class="metric-strip">{html}</div>', unsafe_allow_html=True)


def style_chart(fig: go.Figure, height: int = 400) -> go.Figure:
    """Applies the shared dark/terminal theme to every chart in the app."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT_SANS, color=TEXT, size=12),
        height=height,
        margin=dict(t=36, b=40, l=10, r=10),
        legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor=PANEL_BG, font_family=FONT_MONO, font_color=TEXT),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, tickfont=dict(family=FONT_MONO))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=BORDER, tickfont=dict(family=FONT_MONO))
    return fig


# ---------------------------------------------------------------------------
# Cached market-data lookups
# ---------------------------------------------------------------------------
# Streamlit reruns this entire script top-to-bottom on every interaction —
# including moving a sidebar slider that has nothing to do with the market
# tab. Without caching, dragging the volatility slider would re-hit Yahoo
# Finance over the network every single time. st.cache_data memoizes each
# function's return value by its arguments, so the network call only fires
# again when the ticker/expiration/option type actually changes (or after
# the 5-minute ttl expires, since live quotes do go stale).

@st.cache_data(ttl=300, show_spinner="Looking up ticker...")
def cached_expirations(ticker: str) -> list[str]:
    return market_data.get_expirations(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def cached_spot(ticker: str) -> float:
    return market_data.get_spot_price(ticker)


@st.cache_data(ttl=300, show_spinner="Fetching option chain...")
def cached_chain(ticker: str, expiration: str, option_type: str) -> pd.DataFrame:
    return market_data.get_option_chain(ticker, expiration, option_type)


def error_box(message: str) -> None:
    st.markdown(f'<div class="iv-result error">{message}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: inputs
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-title">Parameters</div>', unsafe_allow_html=True)

option_type = st.sidebar.radio("Option type", options=["Call", "Put"], horizontal=True)

S = st.sidebar.slider("Underlying price (S)", min_value=1.0, max_value=500.0, value=100.0, step=1.0)
K = st.sidebar.slider("Strike price (K)", min_value=1.0, max_value=500.0, value=100.0, step=1.0)
T = st.sidebar.slider("Time to expiration (T, years)", min_value=0.01, max_value=3.0, value=1.0, step=0.01)
r_pct = st.sidebar.slider("Risk-free rate (r, %)", min_value=0.0, max_value=15.0, value=5.0, step=0.1)
sigma_pct = st.sidebar.slider("Volatility (sigma, %)", min_value=1.0, max_value=150.0, value=20.0, step=1.0)
q_pct = st.sidebar.slider("Dividend yield (q, %)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

r = r_pct / 100
sigma = sigma_pct / 100
q = q_pct / 100

inputs = OptionInputs(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
opt_key = option_type.lower()  # "call" or "put"

st.sidebar.caption(
    "Every number, metric, and chart recomputes live from the same "
    "Black-Scholes functions used in the terminal version (black_scholes.py)."
)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-title">Black-Scholes Options Pricer</div>'
    '<div class="app-subtitle">European option pricing, the Greeks, implied volatility, '
    'and strategy payoff analysis.</div>',
    unsafe_allow_html=True,
)

tab_pricer, tab_iv, tab_strategy, tab_market = st.tabs(
    ["Pricer & Greeks", "Implied Volatility", "Strategy Payoff", "Model vs. Market"]
)


# ---------------------------------------------------------------------------
# Tab 1: Pricer & Greeks
# ---------------------------------------------------------------------------
with tab_pricer:
    price_val = call_price(inputs) if opt_key == "call" else put_price(inputs)
    greeks = all_greeks(inputs, opt_key)

    section_label(f"{option_type} price & Greeks")

    vega_per_pct = greeks["vega"] / 100
    theta_per_day = greeks["theta"] / 365
    rho_per_pct = greeks["rho"] / 100

    render_metric_panel([
        ("Price", f"${price_val:,.2f}", "positive"),
        ("Delta", f"{greeks['delta']:+.4f}", sign_class(greeks["delta"])),
        ("Gamma", f"{greeks['gamma']:+.4f}", sign_class(greeks["gamma"])),
        ("Vega (per 1% vol)", f"{vega_per_pct:+.4f}", sign_class(vega_per_pct)),
        ("Theta (per day)", f"{theta_per_day:+.4f}", sign_class(theta_per_day)),
        ("Rho (per 1% rate)", f"{rho_per_pct:+.4f}", sign_class(rho_per_pct)),
    ])

    s_grid = np.linspace(max(0.5, S * 0.5), S * 1.5, 150)

    # --- Chart 1: Price vs underlying ---
    section_label("Price vs. underlying price")
    call_curve = [call_price(OptionInputs(S=s, K=K, T=T, r=r, sigma=sigma, q=q)) for s in s_grid]
    put_curve = [put_price(OptionInputs(S=s, K=K, T=T, r=r, sigma=sigma, q=q)) for s in s_grid]

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=s_grid, y=call_curve, mode="lines", name="Call",
                                    line=dict(color=GREEN, width=2.5)))
    fig_price.add_trace(go.Scatter(x=s_grid, y=put_curve, mode="lines", name="Put",
                                    line=dict(color=RED, width=2.5)))
    fig_price.add_vline(x=K, line_dash="dot", line_color=SLATE, line_width=1,
                         annotation_text=f"K={K:g}", annotation_position="top",
                         annotation_font=dict(color=SLATE, size=11))
    fig_price.add_vline(x=S, line_dash="dash", line_color=SLATE, line_width=1,
                         annotation_text="S", annotation_position="bottom",
                         annotation_font=dict(color=SLATE, size=11))
    fig_price.update_layout(xaxis_title="Underlying price", yaxis_title="Option price")
    st.plotly_chart(style_chart(fig_price), use_container_width=True)
    st.caption("The curve bends rather than moving 1-for-1 with the stock — that curvature is Gamma.")

    # --- Chart 2: Delta vs underlying ---
    section_label("Delta vs. underlying price (hedge ratio)")
    call_delta_curve = [delta_fn(OptionInputs(S=s, K=K, T=T, r=r, sigma=sigma, q=q), "call") for s in s_grid]
    put_delta_curve = [delta_fn(OptionInputs(S=s, K=K, T=T, r=r, sigma=sigma, q=q), "put") for s in s_grid]

    fig_delta = go.Figure()
    fig_delta.add_trace(go.Scatter(x=s_grid, y=call_delta_curve, mode="lines", name="Call delta",
                                    line=dict(color=GREEN, width=2.5)))
    fig_delta.add_trace(go.Scatter(x=s_grid, y=put_delta_curve, mode="lines", name="Put delta",
                                    line=dict(color=RED, width=2.5)))
    fig_delta.add_vline(x=K, line_dash="dot", line_color=SLATE, line_width=1)
    fig_delta.add_hline(y=0, line_color=SLATE, line_width=1)
    fig_delta.update_layout(xaxis_title="Underlying price", yaxis_title="Delta", height=360)
    st.plotly_chart(style_chart(fig_delta, height=360), use_container_width=True)
    st.caption("As the stock moves in-the-money, delta slides toward ±1 — the option starts behaving like the stock.")

    # --- Chart 3: Time decay ---
    section_label(f"Time decay: {option_type.lower()} value as expiration approaches")
    t_grid = np.linspace(T, 1e-4, 150)
    price_fn = call_price if opt_key == "call" else put_price
    decay_curve = [price_fn(OptionInputs(S=S, K=K, T=t, r=r, sigma=sigma, q=q)) for t in t_grid]

    fig_theta = go.Figure()
    color = GREEN if opt_key == "call" else RED
    fig_theta.add_trace(go.Scatter(x=t_grid, y=decay_curve, mode="lines", name=option_type,
                                    line=dict(color=color, width=2.5)))
    fig_theta.update_layout(
        xaxis_title="Time remaining to expiration (years)",
        yaxis_title="Option price",
        xaxis=dict(autorange="reversed"),
        height=360, showlegend=False,
    )
    st.plotly_chart(style_chart(fig_theta, height=360), use_container_width=True)
    st.caption("Decay accelerates as expiration nears — it is not linear.")


# ---------------------------------------------------------------------------
# Tab 2: Implied Volatility
# ---------------------------------------------------------------------------
with tab_iv:
    section_label("Implied volatility solver")
    st.write(
        "Enter a market price for the option (given the S, K, T, r, q set in the "
        "sidebar) and solve backwards for the volatility the market is implying. "
        "Reuses the exact same `implied_volatility()` function (Newton-Raphson with "
        "a bisection fallback) from the terminal version."
    )

    default_market_price = round(call_price(inputs) if opt_key == "call" else put_price(inputs), 4)
    market_price = st.number_input(
        f"Market price of the {option_type.lower()}",
        min_value=0.0, value=float(default_market_price), step=0.01,
    )

    if st.button("Solve for implied volatility"):
        try:
            iv = implied_volatility(
                market_price=market_price, S=S, K=K, T=T, r=r,
                option_type=opt_key, q=q,
            )
            note = ""
            if abs(iv - sigma) < 1e-4:
                note = ("<br><span style='font-size:0.82rem;'>Matches the sigma slider almost exactly — "
                        "expected, since the default market price above was generated from the model at that sigma.</span>")
            st.markdown(
                f'<div class="iv-result">Implied volatility: '
                f'<span class="iv-value">{iv * 100:.2f}%</span>{note}</div>',
                unsafe_allow_html=True,
            )
        except ValueError as e:
            st.markdown(f'<div class="iv-result error">{e}</div>', unsafe_allow_html=True)

    section_label("Why this matters")
    st.write(
        "The market quotes option *prices*, not volatilities. Implied vol is the "
        "standardized way to compare how expensive options are across strikes and "
        "expirations, and it reflects the market's forward-looking estimate of how "
        "much the underlying will move (the VIX is exactly this, computed on S&P 500 "
        "options). **Interview flag:** real markets show a 'volatility smile/skew' — "
        "implied vol differs by strike, which contradicts Black-Scholes' constant-sigma "
        "assumption."
    )


# ---------------------------------------------------------------------------
# Tab 3: Strategy Payoff
# ---------------------------------------------------------------------------
with tab_strategy:
    section_label("Options strategy payoff at expiration")
    st.write(
        "Unlike the first tab (option *value before expiration*, from Black-Scholes), "
        "this chart shows the strategy's **P&L at expiration** as a function of the "
        "final stock price — pure payoff arithmetic, independent of any pricing model."
    )

    strategy = st.selectbox("Strategy", ["Covered Call", "Long Straddle", "Bull Call Spread"])

    s_lo, s_hi = st.slider(
        "Stock price range to plot", min_value=1.0, max_value=500.0,
        value=(max(1.0, S * 0.5), S * 1.5), step=1.0,
    )
    s_range = np.linspace(s_lo, s_hi, 300)

    if strategy == "Covered Call":
        col1, col2, col3 = st.columns(3)
        S0 = col1.number_input("Stock purchase price (S0)", value=float(S), min_value=0.01)
        K_call = col2.number_input("Call strike sold", value=float(K), min_value=0.01)
        premium = col3.number_input(
            "Call premium collected",
            value=round(call_price(OptionInputs(S=S0, K=K_call, T=T, r=r, sigma=sigma, q=q)), 2),
            min_value=0.0,
        )
        result = covered_call(S0=S0, K_call=K_call, call_premium=premium, s_range=s_range)
        legs_desc = f"Long 100 shares @ ${S0:.2f}, short 1 call @ K=${K_call:.2f} for ${premium:.2f} premium"

    elif strategy == "Long Straddle":
        col1, col2, col3 = st.columns(3)
        K_strad = col1.number_input("Strike (same for call & put)", value=float(K), min_value=0.01)
        call_prem = col2.number_input(
            "Call premium paid",
            value=round(call_price(OptionInputs(S=S, K=K_strad, T=T, r=r, sigma=sigma, q=q)), 2),
            min_value=0.0,
        )
        put_prem = col3.number_input(
            "Put premium paid",
            value=round(put_price(OptionInputs(S=S, K=K_strad, T=T, r=r, sigma=sigma, q=q)), 2),
            min_value=0.0,
        )
        result = long_straddle(K=K_strad, call_premium=call_prem, put_premium=put_prem, s_range=s_range)
        legs_desc = f"Long 1 call + long 1 put, both @ K=${K_strad:.2f}"

    else:  # Bull Call Spread
        col1, col2 = st.columns(2)
        K_low = col1.number_input("Lower strike (buy)", value=float(K), min_value=0.01)
        K_high = col2.number_input("Higher strike (sell)", value=float(K) * 1.1, min_value=0.01)
        col3, col4 = st.columns(2)
        prem_low = col3.number_input(
            "Premium paid (lower strike call)",
            value=round(call_price(OptionInputs(S=S, K=K_low, T=T, r=r, sigma=sigma, q=q)), 2),
            min_value=0.0,
        )
        prem_high = col4.number_input(
            "Premium received (higher strike call)",
            value=round(call_price(OptionInputs(S=S, K=K_high, T=T, r=r, sigma=sigma, q=q)), 2),
            min_value=0.0,
        )
        if K_high <= K_low:
            st.markdown('<div class="iv-result error">Higher strike must be greater than the lower strike.</div>',
                        unsafe_allow_html=True)
            st.stop()
        result = bull_call_spread(K_low=K_low, K_high=K_high, premium_low=prem_low,
                                   premium_high=prem_high, s_range=s_range)
        legs_desc = f"Long 1 call @ K=${K_low:.2f}, short 1 call @ K=${K_high:.2f}"

    st.markdown(f'<div class="position-line">Position: <span class="mono">{legs_desc}</span></div>',
                unsafe_allow_html=True)

    render_metric_strip([
        ("Max profit", "Unlimited" if result.max_profit is None else f"${result.max_profit:,.2f}",
         "neutral" if result.max_profit is None else sign_class(result.max_profit)),
        ("Max loss", "Unlimited" if result.max_loss is None else f"${result.max_loss:,.2f}",
         "neutral" if result.max_loss is None else sign_class(result.max_loss)),
        ("Breakeven" + ("s" if len(result.breakevens) > 1 else ""),
         ", ".join(f"${b:,.2f}" for b in result.breakevens), "neutral"),
    ])

    pnl = result.pnl
    pnl_pos = np.where(pnl >= 0, pnl, np.nan)
    pnl_neg = np.where(pnl <= 0, pnl, np.nan)

    fig_payoff = go.Figure()
    fig_payoff.add_trace(go.Scatter(
        x=result.s_range, y=pnl_pos, mode="lines", name="Profit", showlegend=False,
        line=dict(color=GREEN, width=2.5), fill="tozeroy", fillcolor="rgba(0,200,150,0.15)",
        connectgaps=False,
    ))
    fig_payoff.add_trace(go.Scatter(
        x=result.s_range, y=pnl_neg, mode="lines", name="Loss", showlegend=False,
        line=dict(color=RED, width=2.5), fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
        connectgaps=False,
    ))
    fig_payoff.add_hline(y=0, line_color=SLATE, line_width=1)
    for b in result.breakevens:
        fig_payoff.add_vline(x=b, line_dash="dot", line_color=SLATE, line_width=1)
    fig_payoff.update_layout(xaxis_title="Stock price at expiration", yaxis_title="Profit / Loss ($)")
    st.plotly_chart(style_chart(fig_payoff), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Model vs. Market
# ---------------------------------------------------------------------------
with tab_market:
    section_label("Live option chain vs. Black-Scholes")
    st.write(
        "Pull a real option chain from Yahoo Finance and compare it to your Black-Scholes "
        "model. This tab uses the **live market spot price** (not the S slider) together "
        "with the risk-free rate, volatility, and dividend yield set in the sidebar as the "
        "model's assumptions."
    )

    col_ticker, col_exp = st.columns([1, 2])
    ticker_input = col_ticker.text_input("Ticker", value="AAPL").strip().upper()

    expirations: list[str] = []
    if ticker_input:
        try:
            expirations = cached_expirations(ticker_input)
        except MarketDataError as e:
            error_box(str(e))

    expiration = col_exp.selectbox("Expiration date", expirations) if expirations else None

    if ticker_input and expiration:
        try:
            spot = cached_spot(ticker_input)
            chain_df = cached_chain(ticker_input, expiration, opt_key)
            T_market = market_data.years_to_expiration(expiration)
        except MarketDataError as e:
            error_box(str(e))
        else:
            days_remaining = T_market * 365
            st.caption(
                f"Spot price ${spot:,.2f} · {days_remaining:.0f} days to expiration "
                f"(T={T_market:.4f} yrs) · {len(chain_df)} strikes · "
                f"model assumes r={r_pct:.2f}%, sigma={sigma_pct:.2f}%, q={q_pct:.2f}% from the sidebar"
            )

            price_fn = call_price if opt_key == "call" else put_price
            model_prices = [
                price_fn(OptionInputs(S=spot, K=k, T=T_market, r=r, sigma=sigma, q=q))
                for k in chain_df["strike"]
            ]
            compare_df = pd.DataFrame({
                "Strike": chain_df["strike"],
                "Market Price": chain_df["market_price"],
                "Model Price": model_prices,
            })
            compare_df["Diff (Market - Model)"] = compare_df["Market Price"] - compare_df["Model Price"]

            section_label("Model vs. market price, by strike")

            def _diff_color(val: float) -> str:
                color = GREEN if val > 0 else (RED if val < 0 else TEXT)
                return f"color: {color}; font-family: {FONT_MONO};"

            styled = (
                compare_df.style
                .format({
                    "Strike": "{:.2f}",
                    "Market Price": "${:.2f}",
                    "Model Price": "${:.2f}",
                    "Diff (Market - Model)": "{:+.2f}",
                })
                .map(_diff_color, subset=["Diff (Market - Model)"])
                .set_properties(**{"font-family": FONT_MONO})
            )
            st.dataframe(styled, use_container_width=True, hide_index=True, height=360)

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(
                x=compare_df["Strike"], y=compare_df["Model Price"], mode="lines",
                name="Model", line=dict(color=GREEN, width=2.5),
            ))
            fig_cmp.add_trace(go.Scatter(
                x=compare_df["Strike"], y=compare_df["Market Price"], mode="markers",
                name="Market", marker=dict(color=TEXT, size=6, symbol="circle-open", line=dict(width=1.5)),
            ))
            fig_cmp.add_vline(x=spot, line_dash="dash", line_color=SLATE, line_width=1,
                               annotation_text="S", annotation_position="top",
                               annotation_font=dict(color=SLATE, size=11))
            fig_cmp.update_layout(xaxis_title="Strike price", yaxis_title="Option price")
            st.plotly_chart(style_chart(fig_cmp), use_container_width=True)
            st.caption(
                "Where the white market dots sit away from the green model line, Black-Scholes "
                "at this single sigma is over- or under-pricing that strike relative to the market."
            )

            # ---- Implied volatility smile ----
            section_label("Implied volatility smile / skew")
            st.write(
                "For each strike, back out the volatility that makes the Black-Scholes price equal "
                "the market price — reusing the exact same `implied_volatility()` solver from the "
                "Implied Volatility tab, just run once per strike."
            )

            ivs = []
            failed = 0
            for k, mkt_price in zip(compare_df["Strike"], compare_df["Market Price"]):
                try:
                    iv = implied_volatility(
                        market_price=mkt_price, S=spot, K=k, T=T_market, r=r,
                        option_type=opt_key, q=q,
                    )
                    ivs.append(iv)
                except ValueError:
                    ivs.append(np.nan)
                    failed += 1

            smile_df = pd.DataFrame({"Strike": compare_df["Strike"], "Implied Vol": ivs}).dropna()

            if smile_df.empty:
                error_box(
                    "Could not solve implied volatility for any strike in this chain "
                    "(quotes may be too stale, thin, or arbitrage-violating)."
                )
            else:
                if failed:
                    st.caption(
                        f"{failed} of {len(compare_df)} strikes skipped — no valid implied volatility "
                        "found (usually illiquid, deep strikes with stale or crossed quotes)."
                    )

                fig_smile = go.Figure()
                fig_smile.add_trace(go.Scatter(
                    x=smile_df["Strike"], y=smile_df["Implied Vol"] * 100, mode="lines+markers",
                    name="Implied vol", line=dict(color=GREEN, width=2.5), marker=dict(size=5),
                ))
                fig_smile.add_hline(y=sigma_pct, line_dash="dot", line_color=SLATE, line_width=1,
                                     annotation_text=f"Assumed sigma = {sigma_pct:.0f}%",
                                     annotation_font=dict(color=SLATE, size=11))
                fig_smile.add_vline(x=spot, line_dash="dash", line_color=SLATE, line_width=1,
                                     annotation_text="S", annotation_position="top",
                                     annotation_font=dict(color=SLATE, size=11))
                fig_smile.update_layout(xaxis_title="Strike price", yaxis_title="Implied volatility (%)")
                st.plotly_chart(style_chart(fig_smile), use_container_width=True)

            section_label("Why isn't this curve flat?")
            st.write(
                "Black-Scholes assumes **one constant volatility** applies to every strike and "
                "expiration for a given underlying. If that were true, the chart above would be a "
                "flat horizontal line — every strike would imply the exact same sigma. In practice "
                "it almost never is.\n\n"
                "Real option prices build in more than just 'how much will the stock wiggle, on "
                "average.' They also price in **fat tails** (large moves happen more often than a "
                "log-normal, constant-volatility model predicts) and **crash/jump risk**, which isn't "
                "well described by smooth diffusion at all. Demand for deep out-of-the-money puts as "
                "portfolio insurance (and for calls as lottery-ticket upside bets) pushes those "
                "strikes' prices — and therefore their *implied* volatility — higher than an "
                "at-the-money option's implied volatility.\n\n"
                "The shape has a name depending on the market: a roughly symmetric upward curve at "
                "both tails is a **volatility smile** (common in FX options); a skew that rises much "
                "more steeply on the downside — so out-of-the-money puts carry much higher implied "
                "vol than out-of-the-money calls — is a **volatility skew**, and it's been the "
                "dominant pattern in equity index options since the 1987 crash taught the market to "
                "price crash risk explicitly. **Interview flag:** being able to say *'the smile/skew "
                "is the market telling you constant volatility is wrong, and it reflects fat-tail and "
                "crash risk that Black-Scholes doesn't model'* is exactly the kind of answer that "
                "shows real understanding, not just formula recall."
            )
    elif ticker_input and not expirations:
        st.caption("Enter a valid, optionable ticker to load its expiration dates.")
