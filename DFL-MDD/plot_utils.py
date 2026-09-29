"""
plot_utils.py
-------------
Plotting helpers for portfolio backtest results.

Usage
-----
    import importlib
    import plot_utils
    importlib.reload(plot_utils)
    from plot_utils import plot_multi_pnl, plot_overall_comparison
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from performance import build_equity_curve, compute_performance, apply_tc

__all__ = ["plot_multi_pnl", "plot_overall_comparison", "plot_lambda_grid",
           "plot_model_drawdown"]


# ---- shared styling, so the single-lambda and the four-lambda figures match ----
#
# Every model family gets a hue far from the others, and the drawdown budgets
# within a family are separated by lightness. The previous scheme put DFL-MVO on
# teal and GMV on purple, both of which sit next to the DFL-MDD blues and were
# hard to tell apart once thirteen series shared one panel.
DFL_COLORS = ["#9ECAE1", "#4292C6", "#08519C", "#08306B"]   # DFL-MDD, light to navy
MDD_COLORS = ["#A1D99B", "#41AB5D", "#006D2C", "#00441B"]   # PTO-MDD, light to forest
# only one PTO-MVO curve appears per lookback panel, so the strong red comes first
MVO_COLORS = ["#E31A1C", "#FB9A99"]                         # PTO-MVO

DFL_CMAP = plt.cm.Blues      # fallback when a family holds more entries
MDD_CMAP = plt.cm.Greens
MVO_CMAP = plt.cm.Reds


def _family_colors(explicit, cmap, n, lo=0.4, hi=0.9):
    """Explicit hand-picked colors while they last, then sample the colormap."""
    if n <= len(explicit):
        return explicit[:n]
    return [cmap(v) for v in np.linspace(lo, hi, max(n, 1))]


# fixed style per benchmark (color, linestyle)
BENCH_STYLE = {
    "DFL-MVO":  ("#E7298A", (0, (3, 1, 1, 1))),  # magenta dash-dot-dot
    "EW":       ("#000000", (0, (1, 1.2))),      # black dotted (reference line)
    "GMV":      ("#FF8C00", "-."),               # orange dash-dot
    "hist-MVO": ("#8C564B", (0, (5, 2))),        # brown dashed
}


def _bench_style(label):
    for key, style in BENCH_STYLE.items():
        if label.startswith(key):
            return style
    return ("gray", "-")


def plot_multi_pnl(results_list, figsize=(14, 8), title="Cumulative PnL Comparison"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                   gridspec_kw={"height_ratios": [3, 1]},
                                   sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_list)))

    pv_store = []   # collected for the summary table

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
        n_days      = len(pv) - 1
        ann_ret     = (pv[-1] ** (252 / n_days)) - 1 if n_days > 0 else float("nan")
        calmar      = ann_ret / (max_dd + 1e-10)   # Ann.Ret / MDD (consistent with performance.py)

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

    # ---- summary table ----
    print(f"\n{'-'*75}")
    print(f"  {'Label':<35}  {'Ann.Ret':>8}  {'Ann.Vol':>8}  {'MDD':>8}  {'Calmar':>7}")
    print(f"{'-'*75}")
    for label, pv in pv_store:
        daily_rets  = np.diff(pv) / (pv[:-1] + 1e-10)
        n_days      = len(daily_rets)
        ann_ret     = (pv[-1] ** (252 / n_days)) - 1
        ann_vol     = daily_rets.std() * np.sqrt(252)
        running_max = np.maximum.accumulate(pv)
        max_dd      = ((running_max - pv) / (running_max + 1e-10)).max()
        calmar      = ann_ret / (max_dd + 1e-10)
        print(f"  {label:<35}  {ann_ret:>8.2%}  {ann_vol:>8.2%}  {max_dd:>8.2%}  {calmar:>7.2f}")
    print(f"{'-'*75}")


def _fmt_n1_pct(label):
    """
    Tidy a label for the legend:
      - 'n1=0.1' becomes '10%'
      - lookback markers such as 'LB=252, ' or '(LB=252)' are dropped
    e.g. 'DFL-MDD (LB=252, n1=0.1)' -> 'DFL-MDD (10%)'
         'PTO-MVO (LB=252)'         -> 'PTO-MVO'
    """
    s = re.sub(r"n1=([0-9.]+)",
               lambda m: f"{float(m.group(1)) * 100:.0f}%", label)
    s = re.sub(r"LB=\d+\s*,\s*", "", s)   # drop 'LB=252, '
    s = re.sub(r"\s*\(\s*LB=\d+\s*\)", "", s)  # drop a standalone ' (LB=252)'
    s = re.sub(r"\(\s*", "(", s).replace("( ", "(")
    return s.strip()


def _plot_item(ax_pnl, ax_dd, res, lbl, color, linewidth, linestyle="-", x_vals=None):
    eq      = build_equity_curve(res)
    perf    = compute_performance(res)
    peak    = np.maximum.accumulate(eq)
    dd      = (eq - peak) / (peak + 1e-10)
    calmar  = perf['Calmar']   # Ann.Ret / MDD (consistent with performance.py)
    legend_lbl = (f"{_fmt_n1_pct(lbl)}  "
                  f"MDD={perf['MDD']:.1%}  "
                  f"Calmar={calmar:.2f}")
    xs = x_vals[:len(eq)] if x_vals is not None else np.arange(len(eq))
    ax_pnl.plot(xs, eq, label=legend_lbl, color=color, linewidth=linewidth, linestyle=linestyle)
    ax_dd.plot(xs, dd,                    color=color, linewidth=linewidth * 0.7, linestyle=linestyle)
    return dd, xs


def plot_overall_comparison(dfl_results_store, all_results_pto_mdd, all_results_mvo,
                            DELTA_LIST, LAM_LIST, LOOKBACK_LIST,
                            N_STOCKS, PLOT_DIR,
                            full_dates=None, test_start_idx=None,
                            tc_rate=0.0, bench_store=None,
                            dfl_mvo_store=None, full_np=None, REBAL=None,
                            horizon=None, show=True):
    """
    Parameters
    ----------
    dfl_results_store    : dict  {(delta, lam): all_results_dfl_mdd}
    all_results_pto_mdd  : list of (results, label)
    all_results_mvo      : list of (results, label)   PTO-MVO
    DELTA_LIST, LAM_LIST : hyperparameter lists
    LOOKBACK_LIST        : list of lookback values
    N_STOCKS             : int   used in the output filename
    PLOT_DIR             : str   output directory
    bench_store          : dict or None  {label: results}
                           e.g. {"EW": res, "GMV (LB=252)": res,
                                "hist-MVO (LB=252)": res, ...}
                           EW does not depend on LB and appears on every LB panel;
                           the others are matched on 'LB={lb}' in the label.
    dfl_mvo_store        : dict or None  {(delta, lam): [(results, label), ...]}
                           DFL-MVO depends on lambda, so it cannot be passed
                           through bench_store; it is matched on 'LB={lb}'.
    full_np, REBAL       : passed to apply_tc so that transaction costs use the
                           drift-adjusted turnover, matching the result tables.
    horizon              : int or None. When given, the filename carries an
                           'h{horizon}' tag.
    show                 : call plt.show() after each figure.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    for delta_val in DELTA_LIST:
        for lam_val in LAM_LIST:
            if (delta_val, lam_val) not in dfl_results_store:
                print(f"  skipped: delta={delta_val}, lam={lam_val} (no checkpoint)")
                continue

            # apply transaction cost (drift-adjusted when full_np/REBAL are given,
            # so the legend matches the numbers in the result tables)
            def _tc(pairs):
                if not tc_rate:
                    return list(pairs)
                return [(apply_tc(r, tc_rate, full_np=full_np, REBAL=REBAL), l)
                        for r, l in pairs]

            all_results_dfl_mdd = _tc(dfl_results_store[(delta_val, lam_val)])
            pto_mdd_all         = _tc(all_results_pto_mdd)
            all_results_mvo_tc  = _tc(all_results_mvo)
            dfl_mvo_tc          = _tc((dfl_mvo_store or {}).get((delta_val, lam_val), []))

            # ---- one panel per lookback ----
            for lb in LOOKBACK_LIST:
                dfl_lb = [(r, l) for r, l in all_results_dfl_mdd if f"LB={lb}" in l]
                mdd_lb = [(r, l) for r, l in pto_mdd_all          if f"LB={lb}" in l]
                mvo_lb = [(r, l) for r, l in all_results_mvo_tc   if f"LB={lb}" in l]

                if not dfl_lb:
                    continue

                dfl_colors = _family_colors(DFL_COLORS, DFL_CMAP, len(dfl_lb))
                mdd_colors = _family_colors(MDD_COLORS, MDD_CMAP, len(mdd_lb))
                mvo_colors = _family_colors(MVO_COLORS, MVO_CMAP, len(mvo_lb), 0.5, 0.9)

                fig, (ax_pnl, ax_dd) = plt.subplots(
                    2, 1, figsize=(14, 9),
                    gridspec_kw={"height_ratios": [3, 1]},
                    sharex=True
                )

                # build the x-axis date array
                if full_dates is not None and test_start_idx is not None:
                    eq_len = len(build_equity_curve(dfl_lb[0][0]))
                    x_vals = full_dates[test_start_idx:test_start_idx + eq_len]
                else:
                    x_vals = None

                dd_last, xs_last = None, None
                for (res, lbl), color in zip(dfl_lb, dfl_colors):
                    dd_last, xs_last = _plot_item(ax_pnl, ax_dd, res, lbl, color,
                                                  linewidth=1.5, x_vals=x_vals)
                for (res, lbl), color in zip(mdd_lb, mdd_colors):
                    dd_last, xs_last = _plot_item(ax_pnl, ax_dd, res, lbl, color,
                                                  linewidth=1.5, linestyle="--", x_vals=x_vals)
                for (res, lbl), color in zip(mvo_lb, mvo_colors):
                    dd_last, xs_last = _plot_item(ax_pnl, ax_dd, res, lbl, color,
                                                  linewidth=2.0, linestyle=":", x_vals=x_vals)

                # ---- overlay DFL-MVO (lambda-dependent, so not in bench_store) ----
                for res, lbl in dfl_mvo_tc:
                    if f"LB={lb}" not in lbl:
                        continue
                    bcolor, bstyle = _bench_style(lbl)
                    dd_last, xs_last = _plot_item(
                        ax_pnl, ax_dd, res, lbl, bcolor,
                        linewidth=1.8, linestyle=bstyle, x_vals=x_vals)

                # ---- overlay benchmarks (EW / GMV / hist-MVO) ----
                if bench_store is not None:
                    for blbl, bres in bench_store.items():
                        is_ew = blbl.startswith("EW")
                        # EW always, the others only for this lookback
                        if not is_ew and f"LB={lb}" not in blbl:
                            continue
                        bcolor, bstyle = _bench_style(blbl)
                        dd_last, xs_last = _plot_item(
                            ax_pnl, ax_dd, bres, blbl, bcolor,
                            linewidth=1.8, linestyle=bstyle, x_vals=x_vals)

                tc_str = f"  |  TC={int(round(tc_rate*10000))}bps" if tc_rate > 0 else ""
                ax_pnl.set_title(
                    f"Overall Comparison ({N_STOCKS} Industries, Lookback = {lb}){tc_str}")
                ax_pnl.set_ylabel("Portfolio Value")
                # with every model and benchmark overlaid the legend runs long,
                # so split it into two columns
                n_entries = len(ax_pnl.get_lines())
                ax_pnl.legend(loc="upper left", fontsize=7.5,
                              ncol=2 if n_entries > 8 else 1,
                              framealpha=0.9, borderpad=0.4, labelspacing=0.3)

                # small horizontal margin
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

                tc_suffix = f"_tc{int(round(tc_rate*10000))}bps" if tc_rate > 0 else ""
                h_tag     = f"_h{horizon}" if horizon is not None else ""
                plot_path = os.path.join(
                    PLOT_DIR,
                    f"overall_{N_STOCKS}_inds{h_tag}_LB{lb}_{lam_val}{tc_suffix}.png")
                plt.savefig(plot_path, bbox_inches="tight", dpi=450)
                print(f"  saved: {plot_path}")

                if show:
                    plt.show()
                else:
                    plt.close(fig)


def plot_lambda_grid(dfl_results_store, all_results_pto_mdd, all_results_mvo,
                     delta_val, LAM_LIST, lb, N_STOCKS, PLOT_DIR,
                     full_dates=None, test_start_idx=None,
                     tc_rate=0.0, bench_store=None, dfl_mvo_store=None,
                     full_np=None, REBAL=None, horizon=None, show=True,
                     metric="equity"):
    """
    All four loss weights on one figure, as a 2x2 grid for a single universe
    and lookback.

    metric="equity"   portfolio value
    metric="drawdown" running peak-to-trough decline of that same curve

    The series are identical across panels, so the legend is drawn once at the
    figure level and carries model names only; the per-configuration MDD and
    Calmar values belong in the result tables, not in four stacked legends.

    Parameters otherwise mirror plot_overall_comparison, except that `lb` is a
    single lookback and `delta_val` a single risk-aversion value.
    """
    if metric not in ("equity", "drawdown"):
        raise ValueError(f"metric must be 'equity' or 'drawdown', got {metric!r}")
    os.makedirs(PLOT_DIR, exist_ok=True)

    def _tc(pairs):
        if not tc_rate:
            return list(pairs)
        return [(apply_tc(r, tc_rate, full_np=full_np, REBAL=REBAL), l)
                for r, l in pairs]

    def _for_lb(pairs):
        return [(r, l) for r, l in pairs if f"LB={lb}" in l]

    lams = [l for l in LAM_LIST if (delta_val, l) in dfl_results_store]
    if not lams:
        print(f"  skipped: LB={lb} (no checkpoint)")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, sharey=True)
    handles, labels = [], []

    for ax, lam_val in zip(axes.ravel(), lams):
        dfl_lb = _for_lb(_tc(dfl_results_store[(delta_val, lam_val)]))
        mdd_lb = _for_lb(_tc(all_results_pto_mdd))
        mvo_lb = _for_lb(_tc(all_results_mvo))
        dmv_lb = _for_lb(_tc((dfl_mvo_store or {}).get((delta_val, lam_val), [])))

        dfl_colors = _family_colors(DFL_COLORS, DFL_CMAP, len(dfl_lb))
        mdd_colors = _family_colors(MDD_COLORS, MDD_CMAP, len(mdd_lb))
        mvo_colors = _family_colors(MVO_COLORS, MVO_CMAP, len(mvo_lb), 0.5, 0.9)

        if full_dates is not None and test_start_idx is not None:
            eq_len = len(build_equity_curve(dfl_lb[0][0]))
            xs = full_dates[test_start_idx:test_start_idx + eq_len]
        else:
            xs = None

        def _draw(pairs, colors, lw, ls):
            for (res, lbl), color in zip(pairs, colors):
                eq = build_equity_curve(res)
                if metric == "drawdown":
                    peak = np.maximum.accumulate(eq)
                    y = (eq - peak) / (peak + 1e-10)
                else:
                    y = eq
                x = xs[:len(y)] if xs is not None else np.arange(len(y))
                ax.plot(x, y, color=color, linewidth=lw, linestyle=ls,
                        label=_fmt_n1_pct(lbl))

        _draw(dfl_lb, dfl_colors, 1.5, "-")
        _draw(mdd_lb, mdd_colors, 1.5, "--")
        _draw(mvo_lb, mvo_colors, 2.0, ":")
        for res, lbl in dmv_lb:
            c, s = _bench_style(lbl)
            _draw([(res, lbl)], [c], 1.8, s)
        if bench_store is not None:
            for blbl, bres in bench_store.items():
                if not blbl.startswith("EW") and f"LB={lb}" not in blbl:
                    continue
                c, s = _bench_style(blbl)
                _draw([(bres, blbl)], [c], 1.8, s)

        ax.set_title(f"$\\lambda$ = {lam_val}", fontsize=11.5)
        ax.grid(True, alpha=0.3)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        if not handles:
            handles, labels = ax.get_legend_handles_labels()

    for ax in axes.ravel()[len(lams):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Drawdown" if metric == "drawdown" else "Portfolio Value")
    if metric == "drawdown":
        for ax in axes.ravel():
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for ax in axes[1, :]:
        ax.set_xlabel("Date" if full_dates is not None else "Trading Days")
        if full_dates is not None:
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    tc_str = f"  |  TC={int(round(tc_rate*10000))}bps" if tc_rate > 0 else ""
    what = "Drawdown" if metric == "drawdown" else "Overall Comparison"
    fig.suptitle(f"{what} across Loss Weights "
                 f"({N_STOCKS} Industries, Lookback = {lb}){tc_str}",
                 fontsize=13.5, fontweight="bold")
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    tc_suffix = f"_tc{int(round(tc_rate*10000))}bps" if tc_rate > 0 else ""
    h_tag     = f"_h{horizon}" if horizon is not None else ""
    stem = "overall_lamgrid" if metric == "equity" else "drawdown_lamgrid"
    path = os.path.join(
        PLOT_DIR,
        f"{stem}_{N_STOCKS}_inds{h_tag}_LB{lb}{tc_suffix}.png")
    fig.savefig(path, bbox_inches="tight", dpi=450)
    print(f"  saved: {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_model_drawdown(dfl_results_store, all_results_pto_mdd, all_results_mvo,
                        delta_val, lam_val, lb, N_STOCKS, PLOT_DIR,
                        full_dates=None, test_start_idx=None,
                        tc_rate=0.0, dfl_mvo_store=None,
                        full_np=None, REBAL=None, horizon=None, show=True):
    """
    Drawdown of the four models on one panel, for a single loss weight and
    lookback: DFL-MDD and PTO-MDD at each drawdown limit, plus DFL-MVO and
    PTO-MVO, ten curves in all. Benchmarks are left out so the comparison is
    between the models alone.

    The legend carries each curve's full-period maximum drawdown, since that is
    the number the panel is read for, and is ordered worst-first so it matches
    the vertical order of the curves at the trough.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    def _tc(pairs):
        if not tc_rate:
            return list(pairs)
        return [(apply_tc(r, tc_rate, full_np=full_np, REBAL=REBAL), l)
                for r, l in pairs]

    def _for_lb(pairs):
        return [(r, l) for r, l in pairs if f"LB={lb}" in l]

    key = (delta_val, lam_val)
    if key not in dfl_results_store:
        print(f"  skipped: lam={lam_val} (no checkpoint)")
        return None

    dfl_lb = _for_lb(_tc(dfl_results_store[key]))
    mdd_lb = _for_lb(_tc(all_results_pto_mdd))
    mvo_lb = _for_lb(_tc(all_results_mvo))
    dmv_lb = _for_lb(_tc((dfl_mvo_store or {}).get(key, [])))
    if not dfl_lb:
        print(f"  skipped: LB={lb} (no checkpoint)")
        return None

    series = []                      # (label, y, color, linestyle, lw, mdd)
    for (res, lbl), c in zip(dfl_lb, _family_colors(DFL_COLORS, DFL_CMAP, len(dfl_lb))):
        series.append((res, lbl, c, "-", 1.9))
    for (res, lbl), c in zip(mdd_lb, _family_colors(MDD_COLORS, MDD_CMAP, len(mdd_lb))):
        series.append((res, lbl, c, "--", 1.9))
    for res, lbl in dmv_lb:
        c, s = _bench_style(lbl)
        series.append((res, lbl, c, s, 2.1))
    for (res, lbl), c in zip(mvo_lb, _family_colors(MVO_COLORS, MVO_CMAP,
                                                    len(mvo_lb), 0.5, 0.9)):
        series.append((res, lbl, c, ":", 2.3))

    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    drawn, worst = [], 0.0
    for res, lbl, color, ls, lw in series:
        eq   = build_equity_curve(res)
        peak = np.maximum.accumulate(eq)
        y    = (eq - peak) / (peak + 1e-10)
        x    = (full_dates[test_start_idx:test_start_idx + len(y)]
                if full_dates is not None and test_start_idx is not None
                else np.arange(len(y)))
        line, = ax.plot(x[:len(y)], y, color=color, linestyle=ls, linewidth=lw,
                        solid_capstyle="round")
        mdd = -y.min()
        worst = max(worst, mdd)
        drawn.append((mdd, line, f"{_fmt_n1_pct(lbl)}   MDD {mdd:.1%}"))

    # Below the axes rather than inside it: ten entries cover the deepest curves
    # wherever they are placed, and those curves are the point of the panel.
    # Model order, not drawdown order, so the four families stay together.
    fig.legend([l for _, l, _ in drawn], [t for _, _, t in drawn],
               loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=5,
               fontsize=8.5, frameon=False, columnspacing=1.6,
               handlelength=2.6, handletextpad=0.6)

    ax.axhline(0, color="#444", lw=0.9)
    ax.set_ylim(-min(1.0, worst * 1.06), 0.015)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date" if full_dates is not None else "Trading days")
    ax.grid(alpha=0.25, lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    if full_dates is not None:
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())

    tc_str = f"  |  TC={int(round(tc_rate*10000))}bps" if tc_rate > 0 else ""
    ax.set_title(f"Drawdown by model  ({N_STOCKS} Industries, "
                 f"Lookback = {lb}, $\\lambda$ = {lam_val}){tc_str}",
                 fontsize=12.5, fontweight="bold", pad=10)
    fig.tight_layout(rect=[0, 0.10, 1, 1])

    tc_suffix = f"_tc{int(round(tc_rate*10000))}bps" if tc_rate > 0 else ""
    h_tag     = f"_h{horizon}" if horizon is not None else ""
    path = os.path.join(
        PLOT_DIR,
        f"drawdown_models_{N_STOCKS}_inds{h_tag}_LB{lb}_{lam_val}{tc_suffix}.png")
    fig.savefig(path, bbox_inches="tight", dpi=450)
    print(f"  saved: {path}")
    plt.show() if show else plt.close(fig)
    return path
