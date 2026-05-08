"""
plot_mdd.py
───────────
Per-Window MDD 분포 시각화 유틸리티.

사용법
------
    import importlib
    import plot_mdd
    importlib.reload(plot_mdd)
    from plot_mdd import plot_mdd_distribution
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde

__all__ = ["plot_mdd_distribution"]


def plot_mdd_distribution(all_results, title_prefix="DFL-MDD"):
    """
    Parameters
    ----------
    all_results : list of (results, label) tuples
                  results는 backtest_* 반환값 (list of dicts with 'M_real')
    title_prefix : str
    """
    n_configs = len(all_results)
    fig = plt.figure(figsize=(16, 5 * n_configs))
    gs  = gridspec.GridSpec(n_configs, 3, figure=fig, hspace=0.5, wspace=0.35)

    for row, (results, label) in enumerate(all_results):
        mdds = np.array([r["M_real"] for r in results]) * 100  # % 단위

        windows    = np.arange(1, len(mdds) + 1)
        mean_mdd   = mdds.mean()
        median_mdd = np.median(mdds)
        p95_mdd    = np.percentile(mdds, 95)

        # ── 1. 시계열 바 차트 ──────────────────────────────
        ax1 = fig.add_subplot(gs[row, 0])
        colors = ["#d62728" if v > p95_mdd else "#1f77b4" for v in mdds]
        ax1.bar(windows, mdds, color=colors, alpha=0.8, width=0.7)
        ax1.axhline(mean_mdd,   color="red",    linestyle="--", lw=1.5,
                    label=f"Mean={mean_mdd:.2f}%")
        ax1.axhline(median_mdd, color="orange", linestyle=":",  lw=1.5,
                    label=f"Median={median_mdd:.2f}%")
        ax1.set_xlabel("Rebalancing Window")
        ax1.set_ylabel("MDD (%)")
        ax1.set_title(f"{label}\nPer-Window MDD (time series)")
        ax1.legend(fontsize=8)

        # ── 2. 히스토그램 + KDE ────────────────────────────
        ax2 = fig.add_subplot(gs[row, 1])
        n_bins = min(20, max(5, len(mdds) // 3))
        ax2.hist(mdds, bins=n_bins, color="#1f77b4", alpha=0.7,
                 edgecolor="white", density=True, label="Histogram")

        if len(mdds) >= 4:
            kde = gaussian_kde(mdds, bw_method="scott")
            xs  = np.linspace(mdds.min() * 0.8, mdds.max() * 1.1, 300)
            ax2.plot(xs, kde(xs), color="navy", lw=2, label="KDE")

        ax2.axvline(mean_mdd,   color="red",    linestyle="--", lw=1.5,
                    label=f"Mean={mean_mdd:.2f}%")
        ax2.axvline(median_mdd, color="orange", linestyle=":",  lw=1.5,
                    label=f"Median={median_mdd:.2f}%")
        ax2.axvline(p95_mdd,    color="purple", linestyle="-.", lw=1.5,
                    label=f"P95={p95_mdd:.2f}%")
        ax2.set_xlabel("MDD (%)")
        ax2.set_ylabel("Density")
        ax2.set_title(f"{label}\nMDD Distribution")
        ax2.legend(fontsize=8)

        # ── 3. Box + Strip plot ────────────────────────────
        ax3 = fig.add_subplot(gs[row, 2])
        ax3.boxplot(mdds, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#aec7e8", alpha=0.7),
                    medianprops=dict(color="orange", lw=2),
                    whiskerprops=dict(lw=1.5),
                    flierprops=dict(marker="o", markersize=4,
                                    markerfacecolor="#d62728", alpha=0.7))

        jitter = np.random.uniform(-0.1, 0.1, size=len(mdds))
        ax3.scatter(1 + jitter, mdds, alpha=0.5, s=20, color="#1f77b4", zorder=3)

        stats_text = (
            f"n={len(mdds)}\n"
            f"Mean  : {mean_mdd:.2f}%\n"
            f"Median: {median_mdd:.2f}%\n"
            f"Std   : {mdds.std():.2f}%\n"
            f"Min   : {mdds.min():.2f}%\n"
            f"Max   : {mdds.max():.2f}%\n"
            f"P95   : {p95_mdd:.2f}%"
        )
        ax3.text(1.35, mdds.max(), stats_text, fontsize=8,
                 va="top", ha="left",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                           edgecolor="gray", alpha=0.9))
        ax3.set_xticks([])
        ax3.set_ylabel("MDD (%)")
        ax3.set_title(f"{label}\nBoxplot")

    fig.suptitle(f"{title_prefix} — Per-Window MDD Distribution",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()
    return fig
