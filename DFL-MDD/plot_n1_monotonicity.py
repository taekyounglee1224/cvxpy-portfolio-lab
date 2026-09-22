"""
Figure for the drawdown-budget monotonicity test (Reviewer #2, comment 13).

One figure per universe, four panels:

  (a) mean per-window drawdown by budget n1, with within-subject 95% CI
  (b) distribution of per-window drawdown, one colour per budget
  (c) pairwise paired t-statistics as a heatmap (lower: H=126, upper: H=252)
  (d) realized drawdown against the budget, with the violation rate

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
from matplotlib.colors import TwoSlopeNorm
from scipy import stats

from analyze_n1_monotonicity import load_long, to_blocks, N1_LIST

H_COL = {126: "#0B6E8F", 252: "#B23A48"}
N1_COL = dict(zip(N1_LIST, ["#0B6E8F", "#E07A3F", "#3D8B5A", "#B23A48"]))


# ----------------------------------------------------------------------------
def within_subject_ci(A, conf=1.96):
    """Cousineau-Morey within-subject CI half-widths.

    The design is repeated measures: the same decision date is evaluated under
    every budget. Removing each block's own mean strips the between-date
    variation that is common to all budgets, so the error bars reflect the
    paired comparison the tests actually make. The Morey factor k/(k-1)
    corrects the bias introduced by the normalisation.
    """
    n, k = A.shape
    Y = A - A.mean(axis=1, keepdims=True) + A.mean()
    se = Y.std(axis=0, ddof=1) / np.sqrt(n)
    return conf * se * np.sqrt(k / (k - 1))


# ----------------------------------------------------------------------------
def panel_levels(ax, blocks):
    for H, X in blocks.items():
        A = X.values * 100
        ax.errorbar(N1_LIST, A.mean(axis=0), yerr=within_subject_ci(A),
                    marker="o", ms=6, lw=1.8, capsize=4,
                    color=H_COL[H], label=f"H={H}")
    ax.set_xticks(N1_LIST)
    ax.set_xlabel("drawdown budget  $n_1$")
    ax.set_ylabel("mean per-window drawdown (%)")
    ax.set_title("(a)  Realized drawdown rises with the budget",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=9, frameon=False, title="within-subject 95% CI",
              title_fontsize=8)


def panel_hist(ax, long126):
    v = long126.M_real.values * 100
    lim = np.percentile(v, 99)
    bins = np.linspace(0, lim, 50)
    for n1 in N1_LIST:
        d = long126[long126.n1 == n1].M_real.values * 100
        ax.hist(d[d <= lim], bins=bins, histtype="step", lw=1.9,
                color=N1_COL[n1], label=f"$n_1$ = {n1}")
    ax.set_xlim(0, lim)
    ax.set_xlabel("per-window drawdown (%)")
    ax.set_ylabel("windows")
    ax.set_title("(b)  Distribution by budget  (H=126)",
                 fontsize=10.5, loc="left")
    ax.legend(fontsize=9, frameon=False)


def panel_heatmap(ax, blocks, fig):
    """Pairwise paired t-statistics; lower triangle H=126, upper H=252."""
    k = len(N1_LIST)
    M = np.full((k, k), np.nan)
    ann = np.empty((k, k), dtype=object)

    for H, X in blocks.items():
        A = X.values
        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                lower = i > j                      # lower triangle
                if (H == 126) != lower:
                    continue
                a, b = min(i, j), max(i, j)        # b is the looser budget
                t, p2 = stats.ttest_rel(A[:, b], A[:, a])
                p = p2 / 2 if t > 0 else 1 - p2 / 2
                mark = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
                M[i, j] = t
                ann[i, j] = f"{t:+.1f}\n{mark}" if mark else f"{t:+.1f}"

    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r",
                   norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
    for i in range(k):
        for j in range(k):
            if i == j:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                           facecolor="#EDEFF2", edgecolor="none"))
            elif ann[i, j]:
                ax.text(j, i, ann[i, j], ha="center", va="center", fontsize=8.5,
                        color="#111" if abs(M[i, j]) < vmax * .6 else "white")
    ax.set_xticks(range(k), [str(v) for v in N1_LIST])
    ax.set_yticks(range(k), [str(v) for v in N1_LIST])
    ax.set_xlabel("drawdown budget  $n_1$")
    ax.set_ylabel("drawdown budget  $n_1$")
    ax.set_title("(c)  Paired t-statistic, looser vs tighter budget\n"
                 "      lower left: H=126     upper right: H=252",
                 fontsize=10.5, loc="left")
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("t  (positive = looser budget has larger drawdown)", fontsize=8)
    cb.ax.tick_params(labelsize=8)


def panel_slack(ax, long126):
    x = [n * 100 for n in N1_LIST]
    mean = [long126[long126.n1 == n].M_real.mean() * 100 for n in N1_LIST]
    p90 = [np.percentile(long126[long126.n1 == n].M_real, 90) * 100 for n in N1_LIST]
    viol = [(long126[long126.n1 == n].M_real > n).mean() * 100 for n in N1_LIST]

    ax2 = ax.twinx()
    ax2.bar(x, viol, width=3.4, color="#B23A48", alpha=0.22, zorder=0,
            label="violation rate")
    ax2.set_ylabel("violation rate (%)", color="#B23A48", fontsize=9.5)
    ax2.tick_params(axis="y", labelcolor="#B23A48", labelsize=8.5)
    ax2.set_ylim(0, max(viol) * 3.4 + 1)

    ax.plot(x, x, color="#9AA5B1", ls="--", lw=1.5, label="budget  $n_1$")
    ax.fill_between(x, mean, x, color="#9AA5B1", alpha=0.10)
    ax.plot(x, p90, marker="s", ms=5, lw=1.7, color="#E07A3F",
            label="realized p90")
    ax.plot(x, mean, marker="o", ms=6, lw=1.9, color="#0B6E8F",
            label="realized mean")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, frameon=False, loc="upper left")
    ax.set_xticks(x)
    ax.set_xlabel("drawdown budget  $n_1$ (%)")
    ax.set_ylabel("drawdown (%)")
    ax.set_title("(d)  The budget is slack: realized risk sits far below it  (H=126)",
                 fontsize=10.5, loc="left")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="plots/n1_monotonicity")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for n_stocks in (10, 30):
        longs = {H: load_long(n_stocks, H) for H in (126, 252)}
        blocks = {H: to_blocks(longs[H], False) for H in (126, 252)}

        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
        panel_levels(axes[0, 0], blocks)
        panel_hist(axes[0, 1], longs[126])
        panel_heatmap(axes[1, 0], blocks, fig)
        panel_slack(axes[1, 1], longs[126])

        for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
            ax.grid(alpha=0.22, lw=0.7)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        axes[1, 1].spines["right"].set_visible(True)

        fig.suptitle(f"Monotonicity test on drawdown constraints  "
                     f"({n_stocks} industries)",
                     fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out = f"{args.outdir}/n1_monotonicity_{n_stocks}_inds.png"
        fig.savefig(out, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"  saved: {out}")


if __name__ == "__main__":
    main()
