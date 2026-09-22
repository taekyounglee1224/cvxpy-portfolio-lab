"""
Figure for Reviewer #2, comment 13.

One figure per universe, four panels telling the whole argument:

  (a) mean per-window drawdown vs the budget n1, with 95% CI    -> monotone
  (b) paired per-window difference, n1=0.4 minus n1=0.1         -> a real positive shift
  (c) violation rate and realized drawdown vs the budget        -> constraint is slack
  (d) aggregate eight-year maximum drawdown vs n1               -> no ordering

Usage:
    python plot_n1_monotonicity.py [--outdir plots/n1_monotonicity]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from analyze_n1_monotonicity import (
    load_long, to_blocks, page_trend_test, N1_LIST, LAM_LIST, LB_LIST,
)

H_COL = {126: "#0B6E8F", 252: "#B23A48"}
N1_COL = dict(zip(N1_LIST, ["#0B6E8F", "#E07A3F", "#3D8B5A", "#B23A48"]))
RESULT_DIR = "results"


def agg_mdd(n_stocks, horizon):
    """Mean eight-year MDD of DFL-MDD by n1 (TC=0, averaged over lam and LB)."""
    tag = "" if (n_stocks == 10 and horizon == 126) else f"h{horizon}_"
    f = f"{RESULT_DIR}/{n_stocks}_inds_{tag}tc_full_cf.csv"
    d = pd.read_csv(f, encoding="utf-8-sig")
    d = d[(d.tc_bps == 0) & (d.group == "DFL-MDD")].copy()
    d["n1"] = d.label.str.extract(r"n1=([\d.]+)").astype(float)
    return d.groupby("n1")["MDD(%)"].agg(["mean", "sem"])


def panel_trend(ax, blocks):
    """Paired increment relative to the tightest budget.

    The design is repeated measures, so the relevant uncertainty is the paired
    standard error of the within-window difference, not the spread of drawdown
    across dates. Plotting the increment makes the error bars match the test.
    """
    for H, X in blocks.items():
        base = X[N1_LIST[0]].values
        m, lo, hi = [], [], []
        for n1 in N1_LIST:
            d = (X[n1].values - base) * 100
            se = d.std(ddof=1) / np.sqrt(len(d))
            m.append(d.mean()); lo.append(1.96 * se); hi.append(1.96 * se)
        ax.errorbar(N1_LIST, m, yerr=[lo, hi], marker="o", ms=6, lw=1.8,
                    capsize=4, color=H_COL[H], label=f"H={H}")
    ax.axhline(0, color="#444", lw=1.0, ls="--")
    for i, (H, X) in enumerate(blocks.items()):
        _, _, p = page_trend_test(X.values)
        ax.text(0.97, 0.12 - 0.07 * i, f"H={H}   Page's L  p = {p:.1e}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.5, color=H_COL[H])
    ax.set_xlabel("drawdown budget  $n_1$")
    ax.set_ylabel("increase in per-window drawdown vs $n_1$=0.1  (%p)")
    ax.set_title("(a)  Loosening the budget raises realized drawdown",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")


def panel_paired(ax, blocks):
    all_d = np.concatenate([(X[N1_LIST[-1]].values - X[N1_LIST[0]].values) * 100
                           for X in blocks.values()])
    lim = np.percentile(np.abs(all_d), 99)
    bins = np.linspace(-lim, lim, 46)
    for H, X in blocks.items():
        d = (X[N1_LIST[-1]].values - X[N1_LIST[0]].values) * 100
        ax.hist(np.clip(d, -lim, lim), bins=bins, alpha=0.5,
                color=H_COL[H], label=f"H={H}")
        ax.axvline(d.mean(), color=H_COL[H], lw=1.8, ls="-")
    ax.set_xlim(-lim, lim)
    ax.axvline(0, color="#444", lw=1.0, ls="--")
    txt = []
    for H, X in blocks.items():
        lo, hi = X[N1_LIST[0]].values, X[N1_LIST[-1]].values
        t, p2 = stats.ttest_rel(hi, lo)
        p = p2 / 2 if t > 0 else 1 - p2 / 2
        txt.append(f"H={H}:  mean {100*(hi-lo).mean():+.3f}%p,  p = {p:.1e}")
    ax.text(0.03, 0.97, "\n".join(txt), transform=ax.transAxes, va="top",
            fontsize=8.5, color="#333")
    ax.set_xlabel("$M_{real}(n_1{=}0.4) - M_{real}(n_1{=}0.1)$  (%p)")
    ax.set_ylabel("windows")
    ax.set_title("(b)  Paired difference on the same decision dates",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")


def panel_slack(ax, longs):
    d = longs[126]
    ax2 = ax.twinx()
    mean = [d[d.n1 == n].M_real.mean() * 100 for n in N1_LIST]
    p90 = [np.percentile(d[d.n1 == n].M_real, 90) * 100 for n in N1_LIST]
    viol = [(d[d.n1 == n].M_real > n).mean() * 100 for n in N1_LIST]

    ax.plot([n * 100 for n in N1_LIST], [n * 100 for n in N1_LIST], color="#9AA5B1",
            ls="--", lw=1.4, label="budget  $n_1$")
    ax.plot([n * 100 for n in N1_LIST], p90, marker="s", ms=5, lw=1.6,
            color="#E07A3F", label="realized p90")
    ax.plot([n * 100 for n in N1_LIST], mean, marker="o", ms=6, lw=1.8,
            color="#0B6E8F", label="realized mean")
    ax.fill_between([n * 100 for n in N1_LIST], mean, [n * 100 for n in N1_LIST],
                    color="#9AA5B1", alpha=0.10)

    ax2.bar([n * 100 for n in N1_LIST], viol, width=3.2, color="#B23A48",
            alpha=0.28, zorder=0)
    ax2.set_ylabel("violation rate (%)", color="#B23A48", fontsize=9.5)
    ax2.tick_params(axis="y", labelcolor="#B23A48", labelsize=8.5)
    ax2.set_ylim(0, max(viol) * 3.2 + 1)

    ax.set_xlabel("drawdown budget  $n_1$ (%)")
    ax.set_ylabel("drawdown (%)")
    ax.set_title("(c)  The budget is slack: realized risk sits far below it  (H=126)",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")


def panel_agg(ax, n_stocks):
    for H in (126, 252):
        g = agg_mdd(n_stocks, H)
        ax.errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], marker="o", ms=6,
                    lw=1.8, capsize=4, color=H_COL[H], label=f"H={H}")
    ax.set_xlabel("drawdown budget  $n_1$")
    ax.set_ylabel("aggregate 8-year MDD (%)")
    ax.set_title("(d)  The aggregate statistic shows no ordering",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=8.5, frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="plots/n1_monotonicity")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for n_stocks in (10, 30):
        longs = {H: load_long(n_stocks, H) for H in (126, 252)}
        blocks = {H: to_blocks(longs[H], False) for H in (126, 252)}

        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        panel_trend(axes[0, 0], blocks)
        panel_paired(axes[0, 1], blocks)
        panel_slack(axes[1, 0], longs)
        panel_agg(axes[1, 1], n_stocks)
        for ax in axes.ravel():
            ax.grid(alpha=0.22, lw=0.7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[1, 0].spines["right"].set_visible(True)

        fig.suptitle(f"Does a tighter drawdown budget reduce realized risk?  "
                     f"({n_stocks} industries)", fontsize=13.5, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out = f"{args.outdir}/n1_monotonicity_{n_stocks}_inds.png"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
