"""
plot_mdd.py
───────────
Monthly MDD 분포 시각화 유틸리티.

사용법
------
    import importlib
    import plot_mdd
    importlib.reload(plot_mdd)
    from plot_mdd import plot_mdd_distribution
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

__all__ = ["plot_mdd_distribution"]


def _short_label(label):
    """'DFL-MDD (LB=252, n1=0.1)' → 'LB=252, n1=0.1' (Lookback·n1만)"""
    m = re.search(r"\(([^)]*)\)", label)
    return m.group(1).strip() if m else label


def plot_mdd_distribution(all_results, title_prefix="DFL-MDD", ncols=4):
    """
    Parameters
    ----------
    all_results  : list of (results, label) tuples
                   results는 backtest_* 반환값 (list of dicts with 'M_real')
    title_prefix : str  (suptitle 앞에 붙는 접두어)
    ncols        : int  (열 개수, 기본 4 → 8개 config면 4×2)
    """
    n_configs = len(all_results)
    nrows     = -(-n_configs // ncols)   # 올림 나눗셈

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for idx, (results, label) in enumerate(all_results):
        ax   = axes[idx]
        mdds = np.array([r["M_real"] for r in results]) * 100  # % 단위

        mean_mdd   = mdds.mean()
        median_mdd = np.median(mdds)
        p95_mdd    = np.percentile(mdds, 95)

        # 히스토그램
        n_bins = min(20, max(5, len(mdds) // 3))
        ax.hist(mdds, bins=n_bins, color="#7EA6D9", alpha=0.7,
                edgecolor="white", density=True, label="Histogram")

        # KDE
        if len(mdds) >= 4:
            kde = gaussian_kde(mdds, bw_method="scott")
            xs  = np.linspace(mdds.min() * 0.8, mdds.max() * 1.1, 300)
            ax.plot(xs, kde(xs), color="navy", lw=2, label="KDE")

        # 통계선
        ax.axvline(mean_mdd,   color="red",    linestyle="--", lw=1.4,
                   label=f"Mean={mean_mdd:.2f}%")
        ax.axvline(median_mdd, color="green",  linestyle="--", lw=1.4,
                   label=f"Median={median_mdd:.2f}%")
        ax.axvline(p95_mdd,    color="purple", linestyle="--", lw=1.4,
                   label=f"95%={p95_mdd:.2f}%")

        # subplot 제목: Lookback·n1만
        ax.set_title(f"{_short_label(label)}\nMDD Distribution",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("MDD [%]")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7.5)
        ax.grid(True, alpha=0.2)

    # 남는 빈 축 제거
    for j in range(n_configs, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Monthly MDD Distribution",
                 fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    return fig
