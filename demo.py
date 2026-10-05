"""
demo.py
-------
Entry point / demo script. This is the only file meant to be run directly.

It:
    1. Prices a sample option (both call and put) and prints a readable
       table of the price + all five Greeks.
    2. Runs a few sanity checks that prove the math is correct (put-call
       parity, deep-ITM delta -> 1, implied vol round-trip) and prints the
       results.
    3. Generates all four required charts and saves them to charts/.

Run it with:  python demo.py
(after activating the virtual environment — see README.md)
"""

from __future__ import annotations

from black_scholes import (
    OptionInputs,
    call_price,
    put_price,
    all_greeks,
    implied_volatility,
    put_call_parity_gap,
)
from visualize import (
    plot_price_vs_underlying,
    plot_delta_vs_underlying,
    plot_time_decay,
    plot_vega_effect,
)

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Sample option + results table
# ---------------------------------------------------------------------------

def print_results_table(inputs: OptionInputs) -> None:
    """Prints price + all Greeks for both a call and a put, side by side."""
    call_val = call_price(inputs)
    put_val = put_price(inputs)
    call_greeks = all_greeks(inputs, "call")
    put_greeks = all_greeks(inputs, "put")

    print("\nInputs")
    print("-" * 60)
    print(f"  Underlying price (S):     {inputs.S:>10.2f}")
    print(f"  Strike price (K):         {inputs.K:>10.2f}")
    print(f"  Time to expiration (T):   {inputs.T:>10.2f} years")
    print(f"  Risk-free rate (r):       {inputs.r * 100:>10.2f} %")
    print(f"  Volatility (sigma):       {inputs.sigma * 100:>10.2f} %")
    print(f"  Dividend yield (q):       {inputs.q * 100:>10.2f} %")

    print("\nResults")
    print("-" * 60)
    print(f"  {'Metric':<22}{'Call':>16}{'Put':>16}")
    print(f"  {'-' * 22}{'-' * 16:>16}{'-' * 16:>16}")
    print(f"  {'Price':<22}{call_val:>16.4f}{put_val:>16.4f}")
    print(f"  {'Delta':<22}{call_greeks['delta']:>16.4f}{put_greeks['delta']:>16.4f}")
    print(f"  {'Gamma':<22}{call_greeks['gamma']:>16.4f}{put_greeks['gamma']:>16.4f}")
    print(f"  {'Vega (per 1% vol)':<22}{call_greeks['vega'] / 100:>16.4f}"
          f"{put_greeks['vega'] / 100:>16.4f}")
    print(f"  {'Theta (per day)':<22}{call_greeks['theta'] / 365:>16.4f}"
          f"{put_greeks['theta'] / 365:>16.4f}")
    print(f"  {'Rho (per 1% rate)':<22}{call_greeks['rho'] / 100:>16.4f}"
          f"{put_greeks['rho'] / 100:>16.4f}")
    print("-" * 60)


# ---------------------------------------------------------------------------
# 2. Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(inputs: OptionInputs) -> None:
    """
    Runs three checks that prove the implementation is mathematically
    sound, and prints what each one demonstrates.
    """
    print("\nSanity Checks")
    print("-" * 60)

    # --- Check 1: Put-call parity ---
    # Proves: the call and put formulas are internally consistent with each
    # other and with the no-arbitrage relationship C - P = S*e^-qT - K*e^-rT.
    # A bug in either formula would very likely break this.
    gap = put_call_parity_gap(inputs)
    status = "PASS" if abs(gap) < 1e-6 else "FAIL"
    print(f"  [{status}] Put-call parity gap: {gap:.2e} (should be ~0)")

    # --- Check 2: Deep ITM call delta -> 1 ---
    # Proves: the delta formula behaves correctly in the limit. A call that
    # is extremely in-the-money is (almost) certain to be exercised, so it
    # should move almost 1-for-1 with the stock, just like owning the
    # shares outright.
    deep_itm = OptionInputs(S=inputs.S * 5, K=inputs.K, T=inputs.T, r=inputs.r,
                             sigma=inputs.sigma, q=inputs.q)
    from black_scholes import delta
    d = delta(deep_itm, "call")
    status = "PASS" if d > 0.999 else "FAIL"
    print(f"  [{status}] Deep ITM call delta: {d:.6f} (should be ~1.0)")

    # --- Check 3: Deep OTM call delta -> 0 ---
    # Mirror image of check 2: an option that's almost certainly going to
    # expire worthless should barely move at all as the stock wiggles.
    deep_otm = OptionInputs(S=inputs.S * 0.2, K=inputs.K, T=inputs.T, r=inputs.r,
                             sigma=inputs.sigma, q=inputs.q)
    d_otm = delta(deep_otm, "call")
    status = "PASS" if d_otm < 0.001 else "FAIL"
    print(f"  [{status}] Deep OTM call delta: {d_otm:.6f} (should be ~0.0)")

    # --- Check 4: Implied vol round-trip ---
    # Proves: the implied vol solver correctly inverts the pricing function.
    # We price a call at a known sigma, feed that price back into the
    # solver, and confirm we recover the same sigma.
    true_sigma = 0.35
    synthetic_inputs = OptionInputs(S=inputs.S, K=inputs.K, T=inputs.T, r=inputs.r,
                                     sigma=true_sigma, q=inputs.q)
    synthetic_price = call_price(synthetic_inputs)
    recovered_sigma = implied_volatility(
        market_price=synthetic_price, S=inputs.S, K=inputs.K, T=inputs.T,
        r=inputs.r, option_type="call", q=inputs.q,
    )
    diff = abs(recovered_sigma - true_sigma)
    status = "PASS" if diff < 1e-4 else "FAIL"
    print(f"  [{status}] Implied vol round-trip: solved {recovered_sigma:.4f}, "
          f"true {true_sigma:.4f} (diff {diff:.2e})")

    print("-" * 60)


# ---------------------------------------------------------------------------
# 3. Charts
# ---------------------------------------------------------------------------

def generate_charts(inputs: OptionInputs, out_dir: str = "charts") -> None:
    """Generates and saves all four required charts as PNG files."""
    import os
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nGenerating charts into ./{out_dir}/ ...")
    plot_price_vs_underlying(inputs, save_path=f"{out_dir}/price_vs_underlying.png")
    plot_delta_vs_underlying(inputs, save_path=f"{out_dir}/delta_vs_underlying.png")
    plot_time_decay(inputs, option_type="call", save_path=f"{out_dir}/time_decay_call.png")
    plot_vega_effect(inputs, save_path=f"{out_dir}/vega_effect.png")
    print("Done. Files written:")
    print(f"  - {out_dir}/price_vs_underlying.png")
    print(f"  - {out_dir}/delta_vs_underlying.png")
    print(f"  - {out_dir}/time_decay_call.png")
    print(f"  - {out_dir}/vega_effect.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # A realistic, round-number sample option: at-the-money, 1 year to
    # expiry, 5% risk-free rate, 20% volatility, no dividend.
    sample = OptionInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20, q=0.0)

    print("=" * 60)
    print("Black-Scholes Options Pricing Calculator")
    print("=" * 60)

    print_results_table(sample)
    run_sanity_checks(sample)
    generate_charts(sample)

    print("\nAll done. See the charts/ directory for the generated plots.")
