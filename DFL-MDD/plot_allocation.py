"""
plot_allocation.py
──────────────────
포트폴리오 자산 배분 시계열 시각화 유틸리티.

사용법
------
    import importlib
    import plot_allocation
    importlib.reload(plot_allocation)
    from plot_allocation import plot_allocation
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

__all__ = ["plot_allocation"]

ASSET_COLORS = [
    "#4E79A7",  # steel blue
    "#F28E2B",  # orange
    "#59A14F",  # green
    "#E15759",  # red
    "#76B7B2",  # teal
    "#EDC948",  # yellow
    "#B07AA1",  # purple
    "#FF9DA7",  # pink
    "#9C755F",  # brown
    "#BAB0AC",  # gray
]


def _build_allocation_df(results, stock_names, folds, full_dates, REBAL, HORIZON):
    """results (flat list) → 날짜 인덱스 × 자산 weight DataFrame"""
    dates   = []
    weights = []
    idx = 0
    for fold_info in folds:
        t = fold_info["test_start_idx"]
        while (t + HORIZON <= fold_info["test_end_idx"]) and (idx < len(results)):
            dates.append(full_dates[t])
            weights.append(results[idx]["weights"])
            idx += 1
            t += REBAL
    return pd.DataFrame(weights,
                        index=pd.DatetimeIndex(dates),
                        columns=stock_names)


def plot_allocation(dfl_results_store, all_results_mvo,
                    DELTA_LIST, LAM_LIST, LOOKBACK_LIST,
                    stock_names, folds, full_dates,
                    REBAL, HORIZON, N_STOCKS, PLOT_DIR):
    """
    Parameters
    ----------
    dfl_results_store : dict  {(delta, lam): all_results_dfl_mdd}
    all_results_mvo   : list of (results, label)
    DELTA_LIST, LAM_LIST, LOOKBACK_LIST : hyperparameter lists
    stock_names       : list of str
    folds             : fold 정의 리스트
    full_dates        : pd.DatetimeIndex
    REBAL, HORIZON    : int
    N_STOCKS          : int  (파일명용)
    PLOT_DIR          : str  (저장 경로)
    """
    asset_colors = ASSET_COLORS[:len(stock_names)]
    os.makedirs(PLOT_DIR, exist_ok=True)

    for delta_val in DELTA_LIST:
        for lam_val in LAM_LIST:
            if (delta_val, lam_val) not in dfl_results_store:
                print(f"  스킵: delta={delta_val}, lam={lam_val} (결과 없음)")
                continue

            all_results_dfl_mdd = dfl_results_store[(delta_val, lam_val)]

            for lb in LOOKBACK_LIST:
                dfl_items = [(res, lbl) for res, lbl in all_results_dfl_mdd
                             if f"LB={lb}" in lbl]
                mvo_items = [(res, lbl) for res, lbl in all_results_mvo
                             if f"LB={lb}" in lbl]
                all_items = dfl_items + mvo_items

                fig, axes = plt.subplots(len(all_items), 1,
                                         figsize=(9, 3.5 * len(all_items)),
                                         sharex=True)
                if len(all_items) == 1:
                    axes = [axes]

                for ax, (res, lbl) in zip(axes, all_items):
                    df = _build_allocation_df(res, stock_names, folds,
                                              full_dates, REBAL, HORIZON)
                    ax.stackplot(df.index, (df * 100).T,
                                 labels=stock_names,
                                 colors=asset_colors,
                                 alpha=0.92)
                    ax.set_title(lbl, fontsize=10, fontweight="bold")
                    ax.set_ylabel("Allocations")
                    ax.yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
                    ax.set_ylim(0, 100)
                    ax.set_xlim(df.index[0], df.index[-1])  # ← 양 옆 여백 제거
                    ax.axhline(100, color="gray", linewidth=0.8, linestyle="--")
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                    ax.grid(True, alpha=0.2)

                handles, labels_leg = axes[0].get_legend_handles_labels()
                n_assets      = len(labels_leg)
                ncol          = min(10, n_assets)
                n_rows_leg    = -(-n_assets // ncol)       # 올림 나눗셈
                bottom_margin = 0.04 + 0.03 * n_rows_leg  # 레전드 행수에 비례

                fig.legend(handles, labels_leg,
                           loc="lower center", ncol=ncol,
                           bbox_to_anchor=(0.5, 0.0), fontsize=9,
                           markerscale=1.5,
                           handler_map={plt.matplotlib.patches.Polygon:
                                        plt.matplotlib.legend_handler.HandlerPatch()})

                fig.suptitle(
                    f"Portfolio Allocation | LB={lb} | delta={delta_val}, lam={lam_val}",
                    fontsize=13, fontweight="bold", y=1.01)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                plt.subplots_adjust(bottom=bottom_margin)

                alloc_path = os.path.join(
                    PLOT_DIR,
                    f"asset_allocation_{N_STOCKS}_inds_{lb}_{lam_val}.png")
                fig.savefig(alloc_path, bbox_inches="tight", dpi=450)
                print(f"  ✓ plot 저장: {alloc_path}")

                plt.show()
