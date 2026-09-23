"""
benchmarks.py
-------------
Benchmarks and diagnostics that need no retraining (reviewer comments #8, #21,
#2 and #3).

Contents
--------
1. Benchmark backtests, on the same rebalancing dates as DFL-MDD:
     - EW       : equally-weighted
     - GMV      : global minimum variance (long-only)
     - hist-MVO : sample mean-variance (long-only)
   They return the same result-dict structure as the DFL backtests.
     (window, weights, w_real, R_real, M_real, MDD_abs, Sharpe)

2. Drift-aware turnover (reviewer comment #21):
     one-way L1 against the weights as they drifted with realised returns
     since the previous rebalancing date.

3. uncompounded absolute drawdown (Reviewer #1, #2):
     Absolute drawdown on the additive (cumulative-sum) path, reported
     alongside the compounded relative drawdown.

4. Solver infeasibility rate (reviewer comment #4): how often the GMV / MVO
   optimization fails.

Assumptions
----
  full_np  : (T, m) daily simple returns, as decimals
  folds    : list of dicts, each with test_start_idx and test_end_idx
  rebalance: i = test_start_idx + k*REBAL, provided i + HORIZON <= test_end_idx
  lookback : full_np[i-LOOKBACK : i], the LOOKBACK days before i
  holding  : i to i+REBAL; only the first REBAL days are realised

Usage
-----
  import importlib, benchmarks
  importlib.reload(benchmarks)
  from benchmarks import backtest_benchmark, w_equal, w_gmv, w_mvo, compute_turnover
"""

import numpy as np
import cvxpy as cp

__all__ = [
    "rebalance_dates",
    "attach_date_idx",
    "w_equal", "w_gmv", "w_mvo",
    "backtest_benchmark",
    "build_bench_store",
    "compute_turnover",
    "infeasibility_rate",
]


# ----------------------------------------------
# rebalancing dates (identical to the DFL backtests)
# ----------------------------------------------

def rebalance_dates(fold, LOOKBACK, HORIZON, REBAL):
    """Return the list of rebalancing indices i for one fold."""
    idxs = []
    i = fold["test_start_idx"]
    while i + HORIZON <= fold["test_end_idx"]:
        if i - LOOKBACK >= 0:
            idxs.append(i)
        i += REBAL
    return idxs


def attach_date_idx(results, folds, LOOKBACK, HORIZON, REBAL):
    """
    Attach rebalancing indices to a result list that has no date_idx
    (a DFL or PTO checkpoint). Results are assumed to be stored fold by fold
    and window by window, the order in which the backtest produced them.

    Returns
    -------
    a new list, with 'date_idx' added to every dict
    """
    dates = []
    for fold in folds:
        dates += rebalance_dates(fold, LOOKBACK, HORIZON, REBAL)
    if len(dates) != len(results):
        print(f"  warning: attach_date_idx length mismatch (dates={len(dates)}, "
              f"results={len(results)}); check the ordering and structure")
    return [{**r, "date_idx": d} for r, d in zip(results, dates)]


# ----------------------------------------------
# weight functions: weight_fn(R_lb) -> w, where R_lb is (LOOKBACK, m)
# ----------------------------------------------

def w_equal(R_lb, **kwargs):
    m = R_lb.shape[1]
    return np.ones(m) / m, True   # (weights, feasible)


def _solve_long_only(objective, x, m, x_max=1.0):
    """Shared long-only, fully-invested QP solver. Returns (w, feasible).

    When x_max < 1 a per-asset weight cap is added (the default 1.0 means no cap).
    With a cap in place the failure fallback is the equal allocation that still
    respects the cap, rather than plain EW.
    """
    constraints = [cp.sum(x) == 1, x >= 0]
    if x_max < 1.0:
        constraints.append(x <= x_max)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
        if x.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            raise ValueError(prob.status)
        w = np.clip(np.array(x.value).flatten(), 0, x_max if x_max < 1.0 else None)
        s = w.sum()
        return (w / s if s > 0 else np.ones(m) / m), True
    except Exception:
        return np.ones(m) / m, False   # fallback = EW, which respects the cap whenever 1/m <= x_max


def w_gmv(R_lb, ridge=1e-4, x_max=1.0, **kwargs):
    """Global minimum variance, long-only: min x' Sigma x."""
    m = R_lb.shape[1]
    Sigma = np.cov(R_lb.T) + ridge * np.eye(m)
    x = cp.Variable(m)
    obj = cp.Minimize(cp.quad_form(x, cp.psd_wrap(Sigma)))
    return _solve_long_only(obj, x, m, x_max)


def w_mvo(R_lb, delta=20.0, ridge=1e-4, x_max=1.0, **kwargs):
    """
    Historical mean-variance, long-only: max mu'x - (delta/2) x' Sigma x.

    Standard MVO: mu and Sigma are taken over the same horizon (the daily
    lookback), which keeps the objective internally consistent.
      mu    = daily sample mean
      Sigma = daily lookback covariance (plus ridge)
    Scaling mu and Sigma by the same factor leaves the argmax unchanged, so
    daily-daily is the standard choice. Scaling only mu up to the horizon would
    let the return term dominate risk and produce error maximization (extreme
    concentration), so it is not used.
    """
    m = R_lb.shape[1]
    mu    = R_lb.mean(axis=0)
    Sigma = np.cov(R_lb.T) + ridge * np.eye(m)
    x = cp.Variable(m)
    obj = cp.Maximize(mu @ x - (delta / 2.0) * cp.quad_form(x, cp.psd_wrap(Sigma)))
    return _solve_long_only(obj, x, m, x_max)


# ----------------------------------------------
# benchmark backtests (same result structure as the DFL runs)
# ----------------------------------------------

def backtest_benchmark(full_np, folds, weight_fn, LOOKBACK, HORIZON, REBAL,
                       stock_names=None, weight_kwargs=None):
    """
    Returns
    -------
    results : list of dict
        each dict: window, date_idx, weights, w_real, R_real,
                   M_real (relative drawdown, DFL definition),
                   MDD_abs (uncompounded absolute drawdown), Sharpe
    n_infeasible : int   number of optimization failures
    """
    weight_kwargs = weight_kwargs or {}
    m = full_np.shape[1]
    names = stock_names if stock_names else [f"S{j+1}" for j in range(m)]
    results = []
    n_infeasible = 0
    win = 0

    for fold in folds:
        for i in rebalance_dates(fold, LOOKBACK, HORIZON, REBAL):
            win += 1
            R_lb = full_np[i - LOOKBACK : i]              # (LOOKBACK, m)
            w, feasible = weight_fn(R_lb, **weight_kwargs)
            if not feasible:
                n_infeasible += 1

            # realised: only the first REBAL days are held
            R_hold = full_np[i : i + REBAL]              # (REBAL, m)
            p_daily = R_hold @ w                          # (REBAL,) daily portfolio returns
            w_real  = np.cumsum(p_daily)                  # uncompounded cumulative path

            R_real = w_real[-1]

            # relative drawdown, DFL definition (kept for comparability)
            pv_w   = 1.0 + w_real
            rmax_w = np.maximum.accumulate(pv_w)
            M_real = float(np.max((rmax_w - pv_w) / (rmax_w + 1e-10)))

            # uncompounded absolute drawdown (reviewer comments #1 and #2)
            cum      = np.concatenate([[0.0], w_real])
            run_max  = np.maximum.accumulate(cum)
            MDD_abs  = float(np.max(run_max - cum))

            # Sharpe on realised daily returns
            sig = p_daily.std()
            sharpe = float(p_daily.mean() / (sig + 1e-12))

            results.append({
                "window":   win,
                "date_idx": i,
                "weights":  w.astype(np.float32),
                "w_real":   w_real.astype(np.float32),
                "R_real":   float(R_real),
                "M_real":   M_real,
                "MDD_abs":  MDD_abs,
                "Sharpe":   sharpe,
            })

    return results, n_infeasible


# ----------------------------------------------
# drift-aware turnover (reviewer comment #21)
# ----------------------------------------------

def build_bench_store(full_np, folds, stock_names, LOOKBACK_LIST,
                      HORIZON, REBAL, delta=20.0, x_max=1.0, verbose=True):
    """Backtest the EW / GMV / hist-MVO benchmarks together and return a dict.

    Setting x_max once applies it to both GMV and hist-MVO (EW is 1/m and is
    unaffected by the cap), which avoids forgetting x_max on an individual call.

    Returns
    -------
    store : {label: results}
    infeas: {label: n_infeasible}
    """
    kw = {} if x_max >= 1.0 else {"x_max": x_max}
    store, infeas = {}, {}
    r, n = backtest_benchmark(full_np, folds, w_equal, LOOKBACK_LIST[0],
                              HORIZON, REBAL, stock_names)
    store["EW"], infeas["EW"] = r, n
    for lb in LOOKBACK_LIST:
        r_g, n_g = backtest_benchmark(full_np, folds, w_gmv, lb, HORIZON, REBAL,
                                      stock_names, weight_kwargs=dict(kw))
        r_m, n_m = backtest_benchmark(full_np, folds, w_mvo, lb, HORIZON, REBAL,
                                      stock_names,
                                      weight_kwargs={"delta": delta, **kw})
        store[f"GMV (LB={lb})"], infeas[f"GMV (LB={lb})"] = r_g, n_g
        store[f"hist-MVO (LB={lb})"], infeas[f"hist-MVO (LB={lb})"] = r_m, n_m

    if verbose:
        mx = {k: max(float(np.max(x["weights"])) for x in v) for k, v in store.items()}
        print(f"  {len(store)} benchmarks, x_max={x_max}, {len(store['EW'])} windows")
        for k, v in mx.items():
            flag = "  <-- cap violated" if v > x_max + 1e-6 else ""
            print(f"    {k:<20} max weight {v:.4f}{flag}")
    return store, infeas


def compute_turnover(results, full_np, REBAL, one_way=True):
    """
    Turnover measured against the weights as they drifted with realised returns
    since the previous rebalancing date.

    For each rebalance k (>= 2):
      gross_j    = prod_{d=0}^{REBAL-1} (1 + r_{i_{k-1}+d, j})   per-asset gross return
      w_drift_j  = w_{k-1,j} gross_j / sum_l w_{k-1,l} gross_l   drifted weight
      turnover_k = sum_j |w_{k,j} - w_drift_j|                   one-way L1

    Returns
    -------
    dict : { "turnovers": np.ndarray, "mean": float, "median": float }
    """
    turnovers = []
    for k in range(1, len(results)):
        prev = results[k - 1]
        i_prev = prev["date_idx"]
        w_prev = np.array(prev["weights"], dtype=float)
        w_curr = np.array(results[k]["weights"], dtype=float)

        R_hold = full_np[i_prev : i_prev + REBAL]         # (REBAL, m)
        gross  = np.prod(1.0 + R_hold, axis=0)            # (m,) per-asset gross return
        drifted = w_prev * gross
        s = drifted.sum()
        w_drift = drifted / s if s > 0 else w_prev

        to = np.sum(np.abs(w_curr - w_drift))
        turnovers.append(to if one_way else 0.5 * to)

    turnovers = np.array(turnovers)
    return {
        "turnovers": turnovers,
        "mean":   float(turnovers.mean())   if len(turnovers) else float("nan"),
        "median": float(np.median(turnovers)) if len(turnovers) else float("nan"),
    }


# ----------------------------------------------
# infeasibility rate (Reviewer #4)
# ----------------------------------------------

def infeasibility_rate(n_infeasible, results):
    n = len(results)
    return {
        "n_infeasible": n_infeasible,
        "n_total": n,
        "rate": (n_infeasible / n) if n else float("nan"),
    }
