"""
Every configuration ranked by full-period drawdown, with how long that drawdown
took alongside it.

The left panel is the maximum drawdown of the eight-year test path. The right
panel is the length of the episode that produced it, in trading days from the
peak to the trough, which says whether a given drawdown was a crash or a slow
bleed — two portfolios can share a drawdown and differ entirely in how long they
spent in it.

GMV is excluded. PTO-MDD carries no loss weight, and where its drawdown is
identical across drawdown limits the bars are collapsed into one labelled
accordingly, so the ranking is not padded with repeats of the same number.

Usage
-----
    python plot_ranked_mdd.py
    python plot_ranked_mdd.py --data 10 --horizon 126
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks import attach_date_idx
from carryforward import parse_lb, parse_n1
from make_overall_plots import (DELTA_LIST, LAM_LIST, N1_LIST, REBAL,
                                build_stores, load_universe, make_folds)
from performance import build_equity_curve

PLOT_DIR = "./plots/model_plot"
DROP     = {"GMV"}                      # benchmarks left out of the ranking
C_MODEL  = "#E8703A"                    # DFL-MDD
C_OTHER  = "#9AA0A6"
C_DUR    = "#7FB2D4"


def mdd_and_duration(res):
    """Full-period drawdown and the trading days from its peak to its trough."""
    eq   = build_equity_curve(res)
    peak = np.maximum.accumulate(eq)
    dd   = (peak - eq) / (peak + 1e-10)
    trough = int(np.argmax(dd))
    start  = int(np.argmax(eq[:trough + 1])) if trough > 0 else 0
    return float(dd[trough]) * 100.0, trough - start


def collect(n_stocks, horizon, lb, stores, folds):
    """(label, mdd, duration, is_dfl_mdd) for every series at this lookback."""
    dfl_store_cf, dfl_mvo_idx, pto_mdd_idx, mvo_idx, bench_store = stores
    rows = []

    for lam in LAM_LIST:
        key = (DELTA_LIST[0], lam)
        for res, l in dfl_store_cf.get(key, []):
            if parse_lb(l) != lb:
                continue
            m, d = mdd_and_duration(res)
            rows.append((f"DFL-MDD (λ={lam}, d̄={parse_n1(l)})", m, d, True))
        for res, l in dfl_mvo_idx.get(key, []):
            if parse_lb(l) != lb:
                continue
            m, d = mdd_and_duration(res)
            rows.append((f"DFL-MVO (λ={lam})", m, d, False))

    # PTO-MDD has no loss weight; collapse limits that give the same drawdown
    pto = {}
    for res, l in pto_mdd_idx:
        if parse_lb(l) != lb:
            continue
        m, d = mdd_and_duration(res)
        pto.setdefault((round(m, 2), d), []).append(parse_n1(l))
    for (m, d), lims in pto.items():
        tag = ("all limits" if len(lims) == len(N1_LIST)
               else ", ".join(f"d̄={x}" for x in sorted(lims)))
        rows.append((f"PTO-MDD ({tag})", m, d, False))

    for res, l in mvo_idx:
        if parse_lb(l) == lb:
            m, d = mdd_and_duration(res)
            rows.append(("PTO-MVO", m, d, False))

    for blbl, bres in bench_store.items():
        name = blbl.split(" (")[0]
        if name in DROP:
            continue
        if not blbl.startswith("EW") and f"LB={lb}" not in blbl:
            continue
        m, d = mdd_and_duration(bres)
        rows.append((name, m, d, False))

    return sorted(rows, key=lambda r: r[1])


def draw(rows, n_stocks, horizon, lb, out_dir, with_duration=True):
    labels = [r[0] for r in rows]
    mdd    = np.array([r[1] for r in rows])
    dur    = np.array([r[2] for r in rows])
    is_dfl = [r[3] for r in rows]
    y      = np.arange(len(rows))

    if with_duration:
        fig, (axm, axd) = plt.subplots(
            1, 2, figsize=(12.4, 0.34 * len(rows) + 1.8),
            sharey=True, gridspec_kw={"width_ratios": [2.15, 1]})
    else:
        fig, axm = plt.subplots(figsize=(9.0, 0.34 * len(rows) + 1.8))
        axd = None

    colors = [C_MODEL if f else C_OTHER for f in is_dfl]
    axm.barh(y, mdd, color=colors, height=0.72)
    for i, v in enumerate(mdd):
        axm.text(v + mdd.max() * 0.012, i, f"{v:.2f}", va="center",
                 fontsize=8, fontweight="bold" if is_dfl[i] else "normal",
                 color="#B4551F" if is_dfl[i] else "#444")
    axm.set_xlim(0, mdd.max() * 1.16)
    axm.set_xlabel("MDD (%)", fontsize=10)
    axm.set_yticks(y)
    axm.set_yticklabels(labels, fontsize=8.5)
    axm.invert_yaxis()

    if axd is not None:
        axd.barh(y, dur, color=C_DUR, height=0.72)
        for i, v in enumerate(dur):
            axd.text(v + dur.max() * 0.015, i, f"{v:,}", va="center",
                     fontsize=8, color="#2F5F80")
        axd.set_xlim(0, dur.max() * 1.18)
        axd.set_xlabel("Days from peak to trough", fontsize=10)

    for ax in ([axm] if axd is None else [axm, axd]):
        ax.grid(axis="x", alpha=0.22, lw=0.7)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)

    fig.suptitle(("Configurations ranked by maximum drawdown, and how long it took"
                  if axd is not None else
                  "Configurations ranked by maximum drawdown")
                 + f"  ({n_stocks} Industries, Lookback = {lb})",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])

    os.makedirs(out_dir, exist_ok=True)
    stem = "ranked_mdd_duration" if axd is not None else "ranked_mdd"
    path = os.path.join(
        out_dir, f"{stem}_{n_stocks}_inds_h{horizon}_LB{lb}.png")
    fig.savefig(path, bbox_inches="tight", dpi=450)
    plt.close(fig)
    print(f"  saved: {path}  ({len(rows)} series)")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", type=int, choices=[10, 30])
    ap.add_argument("--horizon", type=int, default=126)
    ap.add_argument("--lb", action="append", type=int)
    args = ap.parse_args()

    for n_stocks in (args.data or [10, 30]):
        full_np, full_dates = load_universe(n_stocks)
        folds  = make_folds(full_np, full_dates, args.horizon)
        stores = build_stores(n_stocks, args.horizon, full_np, folds)
        for lb in (args.lb or [252, 504]):
            rows = collect(n_stocks, args.horizon, lb, stores, folds)
            if rows:
                draw(rows, n_stocks, args.horizon, lb, PLOT_DIR, True)
                draw(rows, n_stocks, args.horizon, lb, PLOT_DIR, False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
