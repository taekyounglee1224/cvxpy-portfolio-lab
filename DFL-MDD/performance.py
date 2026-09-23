"""
performance.py
--------------
Compute and print portfolio performance metrics from a backtest result list.

Metrics
----
  Ann.Ret   : annualised return
  Sharpe    : annualised Sharpe ratio (rf = 0)
  CVaR(5%)  : conditional VaR at 5%, on daily returns, reported as a loss magnitude
  MDD       : maximum drawdown over the whole backtest period
  HHI       : mean Herfindahl-Hirschman index, a concentration measure

Usage
-----
  from performance import compute_performance, print_performance_table

  # all_results : [(results_list, label), ...]
  print_performance_table(all_results)
"""

import os
import numpy as np
import pandas as pd

__all__ = [
    "build_equity_curve",
    "apply_tc",
    "compute_performance",
    "print_performance_table",
    "print_tc_performance_table",
    "build_metrics_dataframe",
]


# ----------------------------------------------
# internal helpers
# ----------------------------------------------

def build_equity_curve(results):
    """
    results : list of dicts returned by a backtest_* function; every dict must
              carry 'w_real' (ndarray of shape (rebal,)).

    Returns
    -------
    equity : np.ndarray of shape (T+1,), cumulative portfolio value starting at 1.0
    """
    cum_pv = [1.0]
    for res in results:
        base = cum_pv[-1]
        cum_pv.extend((base * (1.0 + res["w_real"])).tolist())
    return np.array(cum_pv)


def _annualized_return(equity):
    n_days = len(equity) - 1
    if n_days <= 0:
        return float("nan")
    return float((equity[-1] / equity[0]) ** (252.0 / n_days) - 1.0)


def _sharpe(equity, rf=0.0):
    rets = np.diff(equity) / (equity[:-1] + 1e-10)
    excess = rets - rf / 252.0
    std = excess.std()
    if std < 1e-12:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(252.0))


def _cvar(equity, alpha=0.05):
    """
    CVaR (Expected Shortfall) at alpha level.
    Mean of the daily returns at or below the alpha quantile,
    returned as a positive loss magnitude.
    """
    rets = np.diff(equity) / (equity[:-1] + 1e-10)
    cutoff = np.quantile(rets, alpha)
    tail = rets[rets <= cutoff]
    if len(tail) == 0:
        return float("nan")
    return float(-tail.mean())   # sign flipped so the result is a loss magnitude


def _mdd(equity):
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / (peak + 1e-10)
    return float(np.max(drawdown))


def _calmar(equity):
    # standard Calmar: annualised return / MDD
    ann_ret = _annualized_return(equity)
    mdd     = _mdd(equity)
    return float(ann_ret / (mdd + 1e-10))


def _hhi(results):
    """
    Mean of HHI = sum(w_i^2) across rebalancing windows.
    Equals 1/m when fully diversified and 1 when fully concentrated.
    """
    values = []
    for res in results:
        w = np.array(res["weights"])
        values.append(float(np.sum(w ** 2)))
    return float(np.mean(values)) if values else float("nan")


def _mdd_uncompounded(results):
    """
    Absolute maximum drawdown on the uncompounded (additive) cumulative path
    (reviewer comment #1).

    Daily portfolio returns over the whole test period are accumulated
    arithmetically:
        cum_t = sum_{s<=t} r_p,s
    The drawdown is an absolute difference rather than a ratio:
        MDD_abs = max_t ( max_{s<=t} cum_s - cum_t )

    This matches the scale of the drawdown definition used in the optimization
    constraint.
    """
    daily = []
    for res in results:
        w_real = np.asarray(res["w_real"], dtype=float)   # cumulative path inside the window
        # convert the window path back to daily returns and concatenate
        daily.append(np.diff(np.concatenate([[0.0], w_real])))
    if not daily:
        return float("nan")
    r = np.concatenate(daily)
    cum      = np.concatenate([[0.0], np.cumsum(r)])
    run_max  = np.maximum.accumulate(cum)
    return float(np.max(run_max - cum))


def _turnover(results, full_np=None, REBAL=None, one_way=True):
    """
    Mean turnover (reviewer comment #21).

    When full_np and REBAL are supplied, drift is taken into account: turnover is
    measured against the weights actually held after drifting with realised
    returns since the previous rebalance.
        gross_j   = prod(1+r) over the previous holding period
        w_drift_j = w_prev_j gross_j / sum_l w_prev_l gross_l
        turnover  = sum_j |w_curr_j - w_drift_j|      (one-way)
    Without them, turnover is the plain difference of target weights, ignoring
    drift.
    """
    use_drift = (full_np is not None) and (REBAL is not None)
    tos, prev_w, prev_idx = [], None, None

    for res in results:
        w = np.asarray(res["weights"], dtype=float)
        if prev_w is not None:
            if use_drift and (prev_idx is not None):
                R_hold  = full_np[prev_idx : prev_idx + REBAL]
                gross   = np.prod(1.0 + R_hold, axis=0)
                drifted = prev_w * gross
                s       = drifted.sum()
                w_drift = drifted / s if s > 0 else prev_w
                to = float(np.sum(np.abs(w - w_drift)))
            else:
                to = float(np.sum(np.abs(w - prev_w)))
            tos.append(to if one_way else 0.5 * to)
        prev_w   = w
        prev_idx = res.get("date_idx")

    return float(np.mean(tos)) if tos else float("nan")


# ----------------------------------------------
# transaction cost helpers
# ----------------------------------------------

def apply_tc(results, tc_rate=0.0, full_np=None, REBAL=None):
    """
    Apply transaction cost post hoc (one-way).

    Turnover definition (reviewer comment #21):
      With full_np and REBAL, drift is included: turnover is computed against the
      weights actually held (w_drift) after drifting with realised returns since
      the previous rebalance.
          gross_j   = prod(1 + r) over the previous holding period  (per-asset gross)
          w_drift_j = w_prev_j gross_j / sum_l w_prev_l gross_l
          turnover  = sum_j |w_curr_j - w_drift_j|                  (one-way)
      Without them, the older definition is used (difference of target weights,
      ignoring drift), for backward compatibility.

    Parameters
    ----------
    results : backtest_* return list; each dict needs 'weights' and 'w_real',
              plus 'date_idx' when drift is used
    tc_rate : float
    full_np : (T, m) daily returns, required for the drift-aware definition
    REBAL   : int, holding period in days, required for the drift-aware definition

    Returns
    -------
    adjusted : list with the same structure, with transaction cost applied to w_real
    """
    if tc_rate == 0.0:
        return results

    use_drift = (full_np is not None) and (REBAL is not None)

    adjusted = []
    prev_w   = None
    prev_idx = None

    for res in results:
        w      = np.array(res["weights"], dtype=float)
        w_real = np.array(res["w_real"], dtype=float)

        if prev_w is not None:
            if use_drift and (prev_idx is not None):
                # drift over the holding period since the previous rebalance
                R_hold  = full_np[prev_idx : prev_idx + REBAL]
                gross   = np.prod(1.0 + R_hold, axis=0)
                drifted = prev_w * gross
                s       = drifted.sum()
                w_drift = drifted / s if s > 0 else prev_w
                turnover = float(np.sum(np.abs(w - w_drift)))
            else:
                turnover = float(np.sum(np.abs(w - prev_w)))   # older definition, ignoring drift
            tc = tc_rate * turnover
            w_real = (1.0 - tc) * (1.0 + w_real) - 1.0

        adjusted.append({**res, "w_real": w_real})
        prev_w   = w
        prev_idx = res.get("date_idx")   # present for benchmarks, may be missing for DFL

    return adjusted


# ----------------------------------------------
# public API
# ----------------------------------------------

def compute_performance(results, label="", full_np=None, REBAL=None):
    """
    results : backtest_* return list
    label   : strategy name, used in the printed output
    full_np, REBAL : used for drift-aware turnover (reviewer comment #21);
                     without them turnover is approximated

    Returns
    -------
    dict : { label, Ann.Ret, Sharpe, CVaR(5%), MDD, MDD_abs, Calmar, HHI, Turnover }
      - MDD      : relative drawdown on compounded wealth
      - MDD_abs  : absolute drawdown on uncompounded cumulative return
                   (the definition used by the optimization model)
      - Turnover : mean one-way turnover (actual traded amount when drift is used)
    """
    equity = build_equity_curve(results)
    return {
        "label"    : label,
        "Ann.Ret"  : _annualized_return(equity),
        "Sharpe"   : _sharpe(equity),
        "CVaR(5%)" : _cvar(equity),
        "MDD"      : _mdd(equity),
        "MDD_abs"  : _mdd_uncompounded(results),
        "Calmar"   : _calmar(equity),
        "HHI"      : _hhi(results),
        "Turnover" : _turnover(results, full_np=full_np, REBAL=REBAL),
    }


def print_performance_table(all_results, title=None, full_np=None, REBAL=None):
    """
    Parameters
    ----------
    all_results : list of (results_list, label) tuples
    title       : optional heading printed above the table

    Returns
    -------
    df : pd.DataFrame of the raw numeric values, before formatting
    """
    rows = []
    for results, label in all_results:
        rows.append(compute_performance(results, label,
                                        full_np=full_np, REBAL=REBAL))

    df_raw = pd.DataFrame(rows).set_index("label")

    # formatted copy, for display only
    df_fmt = df_raw.copy()
    df_fmt["Ann.Ret"]   = df_raw["Ann.Ret"].map("{:+.2%}".format)
    df_fmt["Sharpe"]    = df_raw["Sharpe"].map("{:.3f}".format)
    df_fmt["CVaR(5%)"]  = df_raw["CVaR(5%)"].map("{:.2%}".format)
    df_fmt["MDD"]       = df_raw["MDD"].map("{:.2%}".format)
    df_fmt["MDD_abs"]   = df_raw["MDD_abs"].map("{:.4f}".format)
    df_fmt["Turnover"]  = df_raw["Turnover"].map("{:.4f}".format)
    df_fmt["HHI"]       = df_raw["HHI"].map("{:.4f}".format)

    if title:
        print(f"\n{'-'*60}")
        print(f"  {title}")
        print(f"{'-'*60}")
    print(df_fmt.to_string())
    print()

    return df_raw   # numeric values, for further analysis


def print_tc_performance_table(all_results,
                                tc_rates=(0.0, 0.10, 0.20, 0.40),
                                title=None, full_np=None, REBAL=None):
    """
    Print one performance table per transaction-cost rate.

    Parameters
    ----------
    all_results : list of (results_list, label) tuples
                  drift-aware TC needs 'date_idx' in every results dict
                  (attach it to DFL/PTO results with benchmarks.attach_date_idx).
    tc_rates    : iterable of float
    title       : optional heading
    full_np, REBAL : used for drift-aware TC (reviewer comment #21);
                     without them the older definition is used

    Returns
    -------
    dict : { tc_rate: pd.DataFrame }
    """
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}  --  transaction cost sensitivity")
        print(f"{'='*70}")

    dfs = {}
    for tc_rate in tc_rates:
        tc_label = f"TC={int(round(tc_rate*10000))}bps"
        rows = []
        for results, label in all_results:
            adj = apply_tc(results, tc_rate, full_np=full_np, REBAL=REBAL)
            rows.append(compute_performance(adj, label, full_np=full_np, REBAL=REBAL))

        df_raw = pd.DataFrame(rows).set_index("label")
        df_fmt = df_raw.copy()
        df_fmt["Ann.Ret"]   = df_raw["Ann.Ret"].map("{:+.2%}".format)
        df_fmt["Sharpe"]    = df_raw["Sharpe"].map("{:.3f}".format)
        df_fmt["CVaR(5%)"]  = df_raw["CVaR(5%)"].map("{:.2%}".format)
        df_fmt["MDD"]       = df_raw["MDD"].map("{:.2%}".format)
        df_fmt["MDD_abs"]   = df_raw["MDD_abs"].map("{:.4f}".format)
        df_fmt["Turnover"]  = df_raw["Turnover"].map("{:.4f}".format)
        df_fmt["HHI"]       = df_raw["HHI"].map("{:.4f}".format)

        print(f"\n  -- {tc_label} --")
        print(df_fmt.to_string())

        dfs[tc_rate] = df_raw

    print()
    return dfs


def _parse_lb_n1(label):
    """Extract (lookback, n1) from a label; None when absent."""
    lb, n1 = None, None
    if "LB=" in label:
        try:
            lb = int(label.split("LB=")[1].split(",")[0].split(")")[0].strip())
        except Exception:
            lb = None
    if "n1=" in label:
        try:
            n1 = float(label.split("n1=")[1].split(")")[0].strip())
        except Exception:
            n1 = None
    return lb, n1


def build_metrics_dataframe(dfl_results_store,
                             all_results_pto_mdd,
                             all_results_mvo,
                             tc_rate=0.0,
                             save_dir=None,
                             N_STOCKS="",
                             bench_store=None,
                             full_np=None,
                             REBAL=None):
    """
    Build one DataFrame per lambda and write it to CSV.
    Each CSV holds the DFL-MDD results for that lambda plus every benchmark.

    Parameters
    ----------
    dfl_results_store   : dict  {(delta, lam): [(results, label), ...]}
    all_results_pto_mdd : list of (results, label)
    all_results_mvo     : list of (results, label)
    tc_rate             : float  transaction cost rate (default 0.0)
    save_dir            : str or None, output directory (e.g. "./csv")
    N_STOCKS            : int or str, used in the filename (e.g. 10, 30)
    bench_store         : dict or None  {label: results}
                          EW / GMV / hist-MVO; the model name and LB are parsed
                          from the label

    Returns
    -------
    dfs : dict  { lam_val: pd.DataFrame }
    """

    # ---- benchmark rows first; they appear in every lambda CSV ----
    benchmark_rows = []

    # EW / GMV / hist-MVO
    if bench_store is not None:
        for blabel, bres in bench_store.items():
            lb, _ = _parse_lb_n1(blabel)
            model = blabel.split("(")[0].strip()   # "GMV (LB=252)" -> "GMV"
            perf = compute_performance(apply_tc(bres, tc_rate, full_np=full_np, REBAL=REBAL),
                                        full_np=full_np, REBAL=REBAL)
            benchmark_rows.append({
                "Model"    : model,
                "lam"      : None,
                "Lookback" : lb,
                "n1"       : None,
                "Ann.Ret"  : perf["Ann.Ret"],
                "Sharpe"   : perf["Sharpe"],
                "CVaR(5%)" : perf["CVaR(5%)"],
                "MDD"      : perf["MDD"],
                "Calmar"   : perf["Calmar"],
                "MDD_abs"  : perf["MDD_abs"],
                "Turnover" : perf["Turnover"],
                "HHI"      : perf["HHI"],
            })

    for results, label in all_results_pto_mdd:
        try:
            lb = int(label.split("LB=")[1].split(",")[0].strip())
            n1 = float(label.split("n1=")[1].split(")")[0].strip())
        except Exception:
            lb, n1 = None, None
        perf = compute_performance(apply_tc(results, tc_rate, full_np=full_np, REBAL=REBAL),
                                   full_np=full_np, REBAL=REBAL)
        benchmark_rows.append({
            "Model"    : "PTO-MDD",
            "lam"      : None,
            "Lookback" : lb,
            "n1"       : n1,
            "Ann.Ret"  : perf["Ann.Ret"],
            "Sharpe"   : perf["Sharpe"],
            "CVaR(5%)" : perf["CVaR(5%)"],
            "MDD"      : perf["MDD"],
            "Calmar"   : perf["Calmar"],
            "MDD_abs"  : perf["MDD_abs"],
            "Turnover" : perf["Turnover"],
            "HHI"      : perf["HHI"],
        })

    for results, label in all_results_mvo:
        try:
            lb = int(label.split("LB=")[1].split(")")[0].strip())
        except Exception:
            lb = None
        perf = compute_performance(apply_tc(results, tc_rate, full_np=full_np, REBAL=REBAL),
                                   full_np=full_np, REBAL=REBAL)
        benchmark_rows.append({
            "Model"    : "PTO-MVO",
            "lam"      : None,
            "Lookback" : lb,
            "n1"       : None,
            "Ann.Ret"  : perf["Ann.Ret"],
            "Sharpe"   : perf["Sharpe"],
            "CVaR(5%)" : perf["CVaR(5%)"],
            "MDD"      : perf["MDD"],
            "Calmar"   : perf["Calmar"],
            "MDD_abs"  : perf["MDD_abs"],
            "Turnover" : perf["Turnover"],
            "HHI"      : perf["HHI"],
        })

    # ---- combine DFL-MDD and benchmarks per lambda, then save ----
    dfs = {}

    for (delta_val, lam_val), results_list in dfl_results_store.items():
        dfl_rows = []
        for results, label in results_list:
            try:
                lb = int(label.split("LB=")[1].split(",")[0].strip())
                n1 = float(label.split("n1=")[1].split(")")[0].strip())
            except Exception:
                lb, n1 = None, None
            perf = compute_performance(apply_tc(results, tc_rate, full_np=full_np, REBAL=REBAL),
                                   full_np=full_np, REBAL=REBAL)
            dfl_rows.append({
                "Model"    : "DFL-MDD",
                "lam"      : lam_val,
                "Lookback" : lb,
                "n1"       : n1,
                "Ann.Ret"  : perf["Ann.Ret"],
                "Sharpe"   : perf["Sharpe"],
                "CVaR(5%)" : perf["CVaR(5%)"],
                "MDD"      : perf["MDD"],
                "Calmar"   : perf["Calmar"],
                "MDD_abs"  : perf["MDD_abs"],
                "Turnover" : perf["Turnover"],
                "HHI"      : perf["HHI"],
            })

        df = pd.DataFrame(dfl_rows + benchmark_rows)
        dfs[lam_val] = df

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            tc_suffix = f"_tc{int(round(tc_rate*10000))}bps" if tc_rate > 0 else ""
            fname = f"{N_STOCKS}_inds_lam{lam_val}{tc_suffix}.csv"
            fpath = os.path.join(save_dir, fname)
            df.to_csv(fpath, index=False)
            print(f"  saved: {fpath}")

    return dfs
