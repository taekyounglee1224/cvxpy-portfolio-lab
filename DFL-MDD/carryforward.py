"""
carryforward.py
---------------
Replace the portfolio weights at infeasible dates with the weights from the
previous rebalancing date and recompute the performance metrics (a no-trade
rule). This is pure post-processing; no retraining is required.

Background
----
When the optimization fails, dfl_mdd.py falls back to a softmax allocation over
the predicted returns. That fallback is not guaranteed to satisfy the drawdown
constraint and is not an operational rule. Applying a no-trade rule instead --
"if it cannot be solved, do not trade" -- means that
 - turnover at that date is zero, and
 - the portfolio stays at the last allocation that did satisfy the constraint.

The failing dates come from the window indices that backtest_dfl_mdd records in
infeas_summary["failed_windows"] (0-based within each fold).

Usage
------
  import importlib, carryforward
  importlib.reload(carryforward)
  from carryforward import apply_carryforward

  dfl_store_cf = apply_carryforward(
      dfl_results_store, infeas_store,
      folds=folds, full_np=full_np,
      HORIZON=HORIZON, REBAL=REBAL, d=d, C=C)
"""

import re
import numpy as np

from benchmarks import attach_date_idx

__all__ = ["carryforward_fallback", "apply_carryforward",
           "parse_lb", "parse_n1"]


# ----------------------------------------------
# label parsing
# ----------------------------------------------

def parse_lb(label, default=252):
    """'DFL-MDD (LB=252, n1=0.1)' -> 252"""
    m = re.search(r"LB=(\d+)", label)
    return int(m.group(1)) if m else default


def parse_n1(label, default=None):
    """'DFL-MDD (LB=252, n1=0.1)' -> 0.1"""
    m = re.search(r"n1=([0-9.]+)", label)
    return float(m.group(1).rstrip(".")) if m else default


# ----------------------------------------------
# core routine: a single configuration
# ----------------------------------------------

def carryforward_fallback(results, infeas_logs, full_np, REBAL, LOOKBACK,
                          d=1.0, C=1.0, ridge=1e-4, verbose=True):
    """
    Parameters
    ----------
    results     : backtest results with the folds concatenated; every element
                  needs a 'date_idx' (attached by attach_date_idx)
    infeas_logs : infeas_map[(LB, n1)], a list of per-fold dicts, each holding
                  'fold', 'n_windows' and 'failed_windows'
    full_np     : (T, m) raw daily returns, before standardisation
    REBAL       : holding period in trading days
    LOOKBACK    : covariance estimation window, used for the Sharpe ratio
    d, C        : return scaling constants (same values as the backtest)
    ridge       : covariance ridge (same as the backtest: 1e-4)

    Returns
    -------
    list : same shape as results, with weights / w_real / R_real / M_real /
           Sharpe updated.

    Notes
    -----
    - Consecutive failures keep the weights of the last successful date
      (w_prev is only updated when the solve is feasible).
    - If the very first window fails there is nothing to carry, so it starts
      from equal weights (EW).
    - Weights carry across fold boundaries, since the portfolio is continuous
      in time.
    - Metric definitions are kept identical to backtest_dfl_mdd.
    """
    # within-fold index -> global index
    failed, offset = set(), 0
    for e in sorted(infeas_logs, key=lambda x: x["fold"]):
        failed |= {offset + w for w in e.get("failed_windows", [])}
        offset += e["n_windows"]

    m      = full_np.shape[1]
    out    = []
    w_prev = np.full(m, 1.0 / m)      # EW if the first window fails
    n_sub  = 0

    for i, r in enumerate(results):
        t = r["date_idx"]

        if i in failed:
            w = w_prev.copy()                       # no-trade
            n_sub += 1
        else:
            w = np.asarray(r["weights"], dtype=float)
            w_prev = w.copy()                       # only remember feasible solutions

        # ---- recompute realised performance ----
        p_ret  = full_np[t:t + REBAL] @ w           # daily portfolio return
        w_real = np.cumsum(p_ret)                   # additive cumulative path
        pv     = 1.0 + w_real
        rmax   = np.maximum.accumulate(pv)

        S   = np.cov(full_np[t - LOOKBACK:t].T) + ridge * np.eye(m)
        sig = np.sqrt(max(float(w @ S @ w), 0.0) + 1e-8)

        out.append({**r,
                    "weights": w,
                    "w_real" : w_real,
                    "R_real" : float(w_real[-1] / (d * C)),
                    "M_real" : float(np.max((rmax - pv) / (rmax + 1e-10))),
                    "Sharpe" : float(p_ret.mean() / sig)})

    if verbose:
        rate = n_sub / len(results) if results else float("nan")
        print(f"    substituted {n_sub}/{len(results)} ({rate:.1%})")
    return out


# ----------------------------------------------
# apply to every configuration
# ----------------------------------------------

def apply_carryforward(results_store, infeas_store, folds, full_np,
                       HORIZON, REBAL, d=1.0, C=1.0, ridge=1e-4,
                       verbose=True):
    """
    Parameters
    ----------
    results_store : {(delta, lam): [(results, label), ...]}
    infeas_store  : {(delta, lam): {(LB, n1): [per-fold dict, ...]}}
    folds         : list of fold definitions, used to rebuild date_idx

    Returns
    -------
    dict : same structure as results_store, with carry-forward applied.
    """
    cf_store = {}
    for key, lst in results_store.items():
        infeas_map = infeas_store[key]
        out = []
        for res, label in lst:
            lb, n1 = parse_lb(label), parse_n1(label)
            logs   = infeas_map.get((lb, n1), infeas_map.get(lb, []))
            if verbose:
                print(f"  [delta={key[0]}, lam={key[1]}] {label}")
            res_idx = attach_date_idx(res, folds, lb, HORIZON, REBAL)
            out.append((carryforward_fallback(
                res_idx, logs, full_np, REBAL, lb,
                d=d, C=C, ridge=ridge, verbose=verbose), label))
        cf_store[key] = out

    if verbose:
        print(f"\ncarry-forward applied to {len(cf_store)} (delta, lam) pairs")
    return cf_store
