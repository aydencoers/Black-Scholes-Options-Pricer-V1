"""
market_data.py
---------------
Thin wrapper around yfinance for fetching live option chain data from
Yahoo Finance.

Deliberately contains NO pricing math — only fetching and light reshaping
of market data (same separation-of-concerns principle as the rest of this
project: black_scholes.py is the model, strategies.py is payoff math, this
module is market data, and app.py is presentation that wires them together).
Comparing this market data to the Black-Scholes model happens in app.py,
using the exact same black_scholes.py functions as every other tab.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


class MarketDataError(Exception):
    """
    Raised whenever a ticker, expiration, or option chain can't be fetched
    or doesn't contain usable data. Caught in app.py and shown as a clean
    message instead of letting the app crash with a raw traceback.
    """


def get_spot_price(ticker: str) -> float:
    """Fetches the most recent traded price for the underlying."""
    try:
        tk = yf.Ticker(ticker)
        price = tk.fast_info.get("lastPrice")
        if price is None:
            # Fallback: fast_info can occasionally miss this field; pull the
            # last daily close instead.
            hist = tk.history(period="1d")
            if hist.empty:
                raise MarketDataError(f"No recent price data found for '{ticker}'.")
            price = float(hist["Close"].iloc[-1])
        return float(price)
    except MarketDataError:
        raise
    except Exception as e:
        raise MarketDataError(f"Could not fetch a spot price for '{ticker}': {e}") from e


def get_expirations(ticker: str) -> list[str]:
    """Returns the list of available option expiration dates ('YYYY-MM-DD' strings)."""
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            raise MarketDataError(f"'{ticker}' has no listed options chain.")
        return list(expirations)
    except MarketDataError:
        raise
    except Exception as e:
        raise MarketDataError(f"Could not look up '{ticker}': {e}") from e


def get_option_chain(ticker: str, expiration: str, option_type: str) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per strike: strike, bid, ask, lastPrice,
    and a computed market_price (mid of bid/ask when both are quoted and
    positive, falling back to lastPrice otherwise — the mid is generally a
    better estimate of "fair" market price than the last trade, which can be
    stale on thinly traded strikes).
    """
    try:
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiration)
        raw = chain.calls if option_type == "call" else chain.puts

        if raw is None or raw.empty:
            raise MarketDataError(f"No {option_type} quotes available for '{ticker}' on {expiration}.")

        df = raw[["strike", "bid", "ask", "lastPrice"]].copy()
        df["bid"] = df["bid"].fillna(0)
        df["ask"] = df["ask"].fillna(0)

        has_two_sided_quote = (df["bid"] > 0) & (df["ask"] > 0)
        df["market_price"] = df["lastPrice"]
        df.loc[has_two_sided_quote, "market_price"] = (
            (df.loc[has_two_sided_quote, "bid"] + df.loc[has_two_sided_quote, "ask"]) / 2
        )

        df = df[df["market_price"] > 0].sort_values("strike").reset_index(drop=True)
        if df.empty:
            raise MarketDataError(
                f"'{ticker}' {option_type}s on {expiration} have no strikes with a usable price."
            )
        return df
    except MarketDataError:
        raise
    except Exception as e:
        raise MarketDataError(f"Could not fetch the option chain for '{ticker}': {e}") from e


def years_to_expiration(expiration: str) -> float:
    """
    Converts a 'YYYY-MM-DD' expiration string into time-to-expiry in years.
    Floored at 1 day so an option expiring "today" still gives Black-Scholes
    a valid, strictly positive T (the model divides by sqrt(T), so T=0 would
    raise an error).
    """
    expiry_date = datetime.strptime(expiration, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    days_remaining = (expiry_date - now).total_seconds() / 86400
    return max(days_remaining, 1.0) / 365
