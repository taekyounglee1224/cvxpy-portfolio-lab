"""
plot_utils.py
─────────────
포트폴리오 백테스트 결과 시각화 유틸리티.

사용법
------
    import importlib
    import plot_utils
    importlib.reload(plot_utils)
    from plot_utils import plot_multi_pnl, plot_overall_comparison
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from performance import build_equity_curve, compute_performance

__all__ = ["plot_multi_pnl", "plot_overall_comparison"]


def plot_multi_pnl(results_list, figsize=(14, 8), title="Cumulative PnL Comparison"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

    pv_store = []   # summary 계산용

    for (bt_results, label), color in zip(results_list, colors):
        pv = [1.0]
        for res in bt_results:
            w    = res["w_real"]
            base = pv[-1]
            pv.extend((base * (1 + w)).tolist())

        pv          = np.array(pv)
        running_max = np.maximum.accumulate(pv)
        drawdown    = (running_max - pv) / (running_max + 1e-10)
        total_ret   = pv[-1] - 1.0
        max_dd      = drawdown.max()
        calmar      = total_ret / (max_dd + 1e-10)

        full_label = f"{label}  R:{total_ret:.1%}  MDD:{max_dd:.1%}  Cal:{calmar:.2f}"
        ax1.plot(np.arange(len(pv)), pv, color=color, linewidth=1.5, label=full_label)
        ax2.plot(np.arange(len(pv)), -drawdown * 100, color=color, linewidth=1.0, alpha=0.7)
        pv_store.append((label, pv))

    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylabel("Portfolio Value")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)

    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trading Days (BT Period)")
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()

    # ── Summary Table ──
    print(f"\n{'─'*75}")
    print(f"  {'Label':<35}  {'Ann.Ret':>8}  {'Ann.Vol':>8}  {'MDD':>8}  {'Calmar':>7}")
    print(f"{'─'*75}")
    for label, pv in pv_store:
        daily_rets  = np.diff(pv) / (pv[:-1] + 1e-10)
        n_days      = len(daily_rets)
        ann_ret     = (pv[-1] ** (252 / n_days)) - 1
        ann_vol     = daily_rets.std() * np.sqrt(252)
        running_max = np.maximum.accumulate(pv)
        max_dd      = ((running_max - pv) / (running_max + 1e-10)).max()
        calmar      = ann_ret / (max_dd + 1e-10)
        print(f"  {label:<35}  {ann_ret:>8.2%}  {ann_vol:>8.2%}  {max_dd:>8.2%}  {calmar:>7.2f}")
    print(f"{'─'*75}")


def _plot_item(ax_pnl, ax_dd, res, lbl, color, linewidth, linestyle="-", x_vals=None):
    eq      = build_equity_curve(res)
    perf    = compute_performance(res)
    peak    = np.maximum.accumulate(eq)
    dd      = (eq - peak) / (peak + 1e-10)
    cum_ret = eq[-1] / eq[0] - 1
    calmar  = cum_ret / (perf['MDD'] + 1e-10)
    legend_lbl = (f"{lbl}  "
                  f"Ret={cum_ret:+.1%}  "
                  f"MDD={perf['MDD']:.1%}  "
                  f"Calmar={calmar:.2f}")
    xs = x_vals[:len(eq)] if x_vals is not None else np.arange(len(eq))
    ax_pnl.plot(xs, eq, label=legend_lbl, color=color, linewidth=linewidth, linestyle=linestyle)
    ax_dd.plot(xs, dd,                    color=color, linewidth=linewidth * 0.7, linestyle=linestyle)
    return dd, xs


def plot_overall_comparison(dfl_results_store, all_results_pto_mdd, all_results_mvo,
                            DELTA_LIST, LAM_LIST, LOOKBACK_LIST,
                            N_STOCKS, PLOT_DIR,
                            full_dates=None, test_start_idx=None):
    """
    Parameters
    ----------
    dfl_results_store    : dict  {(delta, lam): all_results_dfl_mdd}
    all_results_pto_mdd  : list of (results, label)
    all_results_mvo      : list of (results, label)
    DELTA_LIST, LAM_LIST : hyperparameter lists
    LOOKBACK_LIST        : list of lookback values
    N_STOCKS             : int  (파일명용)
    PLOT_DIR             : str  (저장 경로)
    """
    DFL_CMAP = plt.cm.Blues
    MDD_CMAP = plt.cm.Greens
    MVO_CMAP = plt.cm.Reds

    os.makedirs(PLOT_DIR, exist_ok=True)

    for delta_val in DELTA_LIST:
        for lam_val in LAM_LIST:
            if (delta_val, lam_val) not in dfl_results_store:
                print(f"  스킵: delta={delta_val}, lam={lam_val} (체크포인트 없음)")
                continue

            all_results_dfl_mdd = dfl_results_store[(delta_val, lam_val)]

            # PTO-MDD: n1별로 모두 표시
            pto_mdd_all = all_results_pto_mdd

            n_dfl = len(all_results_dfl_mdd)
            n_mdd = len(pto_mdd_all)
            n_mvo = len(all_results_mvo)

            dfl_colors = [DFL_CMAP(v) for v in np.linspace(0.4, 0.9, max(n_dfl, 1))]
            mdd_colors = [MDD_CMAP(v) for v in np.linspace(0.4, 0.9, max(n_mdd, 1))]
            mvo_colors = [MVO_CMAP(v) for v in np.linspace(0.5, 0.9, max(n_mvo, 1))]

            fig, (ax_pnl, ax_dd) = plt.subplots(
                2, 1, figsize=(16, 10),
                gridspec_kw={"height_ratios": [3, 1]},
                sharex=True
            )

            # x축 날짜 배열 구성
            if full_dates is not None and test_start_idx is not None:
                first_res = all_results_dfl_mdd[0][0]
                eq_len    = len(build_equity_curve(first_res))
                x_vals    = full_dates[test_start_idx:test_start_idx + eq_len]
            else:
                x_vals = None

            dd_last, xs_last = None, None
            for (res, lbl), color in zip(all_results_dfl_mdd, dfl_colors):
                dd_last, xs_last = _plot_item(ax_pnl, ax_dd, res, lbl, color,
                                              linewidth=1.5, x_vals=x_vals)

            for (res, lbl), color in zip(pto_mdd_all, mdd_colors):
                dd_last, xs_last = _plot_item(ax_pnl, ax_dd, res, lbl, color,
                                              linewidth=1.5, linestyle="--", x_vals=x_vals)

            for (res, lbl), color in zip(all_results_mvo, mvo_colors):
                dd_last, xs_last = _plot_item(ax_pnl, ax_dd, res, lbl, color,
                                              linewidth=2.0, linestyle=":", x_vals=x_vals)

            ax_pnl.set_title(
                f"DFL-MDD vs PTO-MDD vs PTO-MVO | delta={delta_val}, lam={lam_val}")
            ax_pnl.set_ylabel("Portfolio Value")
            ax_pnl.legend(loc="upper left", fontsize=7.0)
            # 양 옆 살짝 여백
            if x_vals is not None:
                import pandas as pd
                pad = pd.Timedelta(days=60)
                x_lo, x_hi = xs_last[0] - pad, xs_last[-1] + pad
            else:
                pad = len(xs_last) * 0.02
                x_lo, x_hi = xs_last[0] - pad, xs_last[-1] + pad

            ax_pnl.set_xlim(x_lo, x_hi)
            ax_pnl.grid(True, alpha=0.3)

            ax_dd.set_ylabel("Drawdown")
            ax_dd.set_xlabel("Date" if x_vals is not None else "Trading Days")
            ax_dd.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
            ax_dd.fill_between(xs_last, dd_last, 0, alpha=0.1, color="gray")
            ax_dd.set_xlim(x_lo, x_hi)

            if x_vals is not None:
                import matplotlib.dates as mdates
                ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
                ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
                plt.setp(ax_dd.xaxis.get_majorticklabels(), rotation=45, ha="right")

            ax_dd.grid(True, alpha=0.3)

            plt.tight_layout()

            plot_path = os.path.join(PLOT_DIR,
                                     f"overall_{N_STOCKS}_inds_{lam_val}.png")
            plt.savefig(plot_path, bbox_inches="tight", dpi=450)
            print(f"  ✓ plot 저장: {plot_path}")

            plt.show()
