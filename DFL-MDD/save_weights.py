"""
save_weights.py
---------------
Write the optimal portfolio weights at every rebalancing date to CSV
(addresses reviewer comments #11, #19 and #22).

Layout (wide)
----------
  date_idx, date, <asset_1>, <asset_2>, ..., <asset_m>
  4528, 2018-01-02, 0.1234, 0.0000, ...

Filename
------
  {N_STOCKS}_inds_{model}[_lam{lam}][_LB{lb}][_n1{n1}].csv
  e.g. 10_inds_DFL-MDD_lam0.5_LB252_n10.1.csv
      10_inds_PTO-MVO_LB252.csv
      10_inds_EW.csv

Usage
-----
  import importlib, save_weights
  importlib.reload(save_weights)
  from save_weights import save_all_weights

  save_all_weights(
      dfl_store=dfl_store_idx, pto_mdd=pto_mdd_idx, pto_mvo=mvo_idx,
      dfl_mvo_store=dfl_mvo_idx, bench_store=bench_store,
      full_dates=full_dates, stock_names=stock_names,
      folds=folds, HORIZON=HORIZON, REBAL=REBAL,
      N_STOCKS=N_STOCKS, save_dir="./weights")
"""

import os
import re
import numpy as np
import pandas as pd

__all__ = ["save_weights_csv", "save_all_weights", "summarize_concentration"]


# ----------------------------------------------
# internal helpers
# ----------------------------------------------

def _slug(label, lam=None):
    """
    Turn a label into a filename fragment.
      'DFL-MDD (LB=252, n1=0.1)' with lam=0.5 -> 'DFL-MDD_lam0.5_LB252_n10.1'
      'PTO-MVO (LB=252)'                       -> 'PTO-MVO_LB252'
      'EW'                                     -> 'EW'
    """
    model = label.split("(")[0].strip()
    parts = [model]
    if lam is not None:
        parts.append(f"lam{lam}")
    m_lb = re.search(r"LB=(\d+)", label)
    if m_lb:
        parts.append(f"LB{m_lb.group(1)}")
    m_n1 = re.search(r"n1=([0-9.]+)", label)
    if m_n1:
        parts.append(f"n1{m_n1.group(1)}")
    return "_".join(parts)


def _rebal_dates_from_folds(folds, LOOKBACK, HORIZON, REBAL):
    """For results without date_idx: rebuild rebalancing indices from the folds."""
    idxs = []
    for fold in folds:
        i = fold["test_start_idx"]
        while i + HORIZON <= fold["test_end_idx"]:
            if i - LOOKBACK >= 0:
                idxs.append(i)
            i += REBAL
    return idxs


# ----------------------------------------------
# save a single model
# ----------------------------------------------

def save_weights_csv(results, label, stock_names, save_dir,
                     N_STOCKS, full_dates=None, lam=None,
                     folds=None, HORIZON=None, REBAL=None, LOOKBACK=252,
                     verbose=True):
    """
    Write one configuration's weight history to a wide CSV.

    Parameters
    ----------
    results     : list of backtest dicts, each with 'weights' (and 'date_idx' when available)
    label       : e.g. 'DFL-MDD (LB=252, n1=0.1)'
    stock_names : asset names, used as column headers
    save_dir    : output directory
    N_STOCKS    : used in the output filename
    full_dates  : pd.DatetimeIndex; when given, a real date column is added
    lam         : lambda value, included in the filename for the DFL models
    folds, HORIZON, REBAL, LOOKBACK
                : used to rebuild rebalancing indices when date_idx is absent

    Returns
    -------
    df : the DataFrame that was written
    """
    os.makedirs(save_dir, exist_ok=True)

    # obtain date_idx
    date_idx = [r.get("date_idx") for r in results]
    if any(d is None for d in date_idx):
        if folds is not None and HORIZON is not None and REBAL is not None:
            recon = _rebal_dates_from_folds(folds, LOOKBACK, HORIZON, REBAL)
            if len(recon) == len(results):
                date_idx = recon
            else:
                if verbose:
                    print(f"  warning: {label}: rebuilt date_idx has the wrong length "
                          f"({len(recon)} vs {len(results)}); falling back to position")
                date_idx = list(range(len(results)))
        else:
            date_idx = list(range(len(results)))

    W = np.vstack([np.asarray(r["weights"], dtype=float) for r in results])
    # clean up solver noise (long-only solutions can carry values like -1e-13)
    W = np.where(np.abs(W) < 1e-8, 0.0, W)

    df = pd.DataFrame(W, columns=stock_names)
    df.insert(0, "date_idx", date_idx)

    if full_dates is not None:
        try:
            df.insert(1, "date", [str(full_dates[i])[:10] for i in date_idx])
        except Exception:
            pass

    fname = f"{N_STOCKS}_inds_{_slug(label, lam)}.csv"
    fpath = os.path.join(save_dir, fname)
    df.to_csv(fpath, index=False, float_format="%.6f")
    if verbose:
        print(f"  saved weights: {fpath}  ({len(df)} rebalances)")
    return df


# ----------------------------------------------
# save every model at once
# ----------------------------------------------

def save_all_weights(dfl_store=None, pto_mdd=None, pto_mvo=None,
                     dfl_mvo_store=None, bench_store=None,
                     full_dates=None, stock_names=None,
                     folds=None, HORIZON=None, REBAL=None,
                     N_STOCKS="", save_dir="./weights", verbose=True):
    """
    Write the weight CSV for every model in one pass.

    Parameters
    ----------
    dfl_store     : dict {(delta, lam): [(results, label), ...]}   DFL-MDD
    pto_mdd       : list [(results, label), ...]
    pto_mvo       : list [(results, label), ...]
    dfl_mvo_store : dict {(delta, lam): [(results, label), ...]}   DFL-MVO
    bench_store   : dict {label: results}                          EW/GMV/hist-MVO
    others        : see save_weights_csv

    Returns
    -------
    n_saved : int
    """
    os.makedirs(save_dir, exist_ok=True)
    n = 0

    def _lb_of(label, default=252):
        m = re.search(r"LB=(\d+)", label)
        return int(m.group(1)) if m else default

    # DFL-MDD, one per lambda
    if dfl_store:
        for (delta_val, lam_val), results_list in dfl_store.items():
            for results, label in results_list:
                save_weights_csv(results, label, stock_names, save_dir,
                                 N_STOCKS, full_dates=full_dates, lam=lam_val,
                                 folds=folds, HORIZON=HORIZON, REBAL=REBAL,
                                 LOOKBACK=_lb_of(label), verbose=verbose)
                n += 1

    # DFL-MVO, one per lambda
    if dfl_mvo_store:
        for (delta_val, lam_val), results_list in dfl_mvo_store.items():
            for results, label in results_list:
                save_weights_csv(results, label, stock_names, save_dir,
                                 N_STOCKS, full_dates=full_dates, lam=lam_val,
                                 folds=folds, HORIZON=HORIZON, REBAL=REBAL,
                                 LOOKBACK=_lb_of(label), verbose=verbose)
                n += 1

    # PTO-MDD / PTO-MVO (independent of lambda)
    for lst in (pto_mdd, pto_mvo):
        if lst:
            for results, label in lst:
                save_weights_csv(results, label, stock_names, save_dir,
                                 N_STOCKS, full_dates=full_dates, lam=None,
                                 folds=folds, HORIZON=HORIZON, REBAL=REBAL,
                                 LOOKBACK=_lb_of(label), verbose=verbose)
                n += 1

    # static benchmarks (EW / GMV / hist-MVO)
    if bench_store:
        for label, results in bench_store.items():
            save_weights_csv(results, label, stock_names, save_dir,
                             N_STOCKS, full_dates=full_dates, lam=None,
                             folds=folds, HORIZON=HORIZON, REBAL=REBAL,
                             LOOKBACK=_lb_of(label), verbose=verbose)
            n += 1

    print(f"\n  wrote {n} weight CSV files to {save_dir}")
    return n


# ----------------------------------------------
# concentration summary (reviewer comment #22)
# ----------------------------------------------

def summarize_concentration(results, label=""):
    """
    Summarise portfolio concentration.

    Returns
    -------
    dict : label, HHI, EffN, MaxWeight, AvgMaxWeight, AvgNActive
      - EffN         : 1/HHI  (effective number of assets)
      - MaxWeight    : largest single weight over the whole period
      - AvgMaxWeight : mean of the per-rebalance maximum weight
      - AvgNActive   : mean number of assets holding more than 1%
    """
    W = np.vstack([np.asarray(r["weights"], dtype=float) for r in results])
    W = np.where(np.abs(W) < 1e-8, 0.0, W)   # clean up solver noise
    hhi_t = (W ** 2).sum(axis=1)
    hhi   = float(hhi_t.mean())
    return {
        "label"        : label,
        "HHI"          : hhi,
        "EffN"         : float(1.0 / hhi) if hhi > 0 else float("nan"),
        "MaxWeight"    : float(W.max()),
        "AvgMaxWeight" : float(W.max(axis=1).mean()),
        "AvgNActive"   : float((W > 0.01).sum(axis=1).mean()),
    }
