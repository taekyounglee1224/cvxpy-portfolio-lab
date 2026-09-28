"""
make_overall_plots.py
---------------------
Regenerate the "Overall Comparison" figures with every model and benchmark on
one panel: DFL-MDD (4 budgets), PTO-MDD (4 budgets), DFL-MVO, PTO-MVO, EW, GMV
and hist-MVO.

The stores are assembled exactly as the notebook assembles them for the result
tables, so the legend values reproduce the published CSVs:

  DFL-MDD   carry-forward applied
  DFL-MVO   raw (carry-forward is not applied to it in the tables either)
  PTO-MDD   raw
  PTO-MVO   raw
  EW / GMV / hist-MVO   from the benchmark checkpoint

After plotting, every legend value is checked against
results/{N}_inds_h{H}_tc_full_cf.csv and any mismatch is reported.

Usage
-----
  python make_overall_plots.py --data 10
  python make_overall_plots.py --data 10 --data 30 --tc 0 --tc 0.001
"""

import argparse
import os
import pickle

import matplotlib
matplotlib.use("Agg")                      # write files, never open a window

import numpy as np
import pandas as pd

from benchmarks import attach_date_idx
from carryforward import apply_carryforward, parse_lb
from performance import apply_tc, build_equity_curve, compute_performance
from plot_utils import plot_overall_comparison

CKPT_DIR = "./checkpoint"
RSLT_DIR = "./results"
PLOT_DIR = "./plots/overall_plot"

# -- fixed experiment grid, mirroring the notebooks --
REBAL, VAL_YEARS, TEST_YEARS, N_FOLDS = 21, 5, 1, 8
LAM_LIST      = [0.3, 0.5, 0.7, 1.0]
DELTA_LIST    = [20]
LOOKBACK_LIST = [252, 504]
N1_LIST       = [0.1, 0.2, 0.3, 0.4]
SOLVER        = "CLARABEL"
C, d          = 1.0, 1.0


def load_universe(n_stocks):
    inds = pd.read_csv(f"csv/{n_stocks}_industry.csv")
    inds["Date"] = pd.to_datetime(inds["Date"])
    inds = inds.set_index("Date").sort_index()
    inds = inds[~inds.index.duplicated(keep="first")] / 100.0
    return inds.values, inds.index


def make_folds(full_np, full_dates, horizon):
    idx = lambda s: full_dates.searchsorted(pd.Timestamp(s), side="left")
    return [{
        "fold"          : f + 1,
        "train_end_idx" : idx(f"{2018 + f - VAL_YEARS}-01-01"),
        "val_start_idx" : idx(f"{2018 + f - VAL_YEARS}-01-01"),
        "val_end_idx"   : idx(f"{2018 + f}-01-01"),
        "test_start_idx": idx(f"{2018 + f}-01-01"),
        "test_end_idx"  : min(idx(f"{2018 + f + TEST_YEARS}-01-01") + horizon,
                              len(full_np)),
        "test_year"     : 2018 + f,
    } for f in range(N_FOLDS)]


def _ck(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def build_stores(n_stocks, horizon, full_np, folds):
    """Return (dfl_store_cf, dfl_mvo_idx, pto_mdd_idx, mvo_idx, bench_store)."""
    configs = [(lb, n1) for lb in LOOKBACK_LIST for n1 in N1_LIST]

    # -- DFL-MDD, then carry-forward --
    raw, infeas = {}, {}
    for dv in DELTA_LIST:
        for lv in LAM_LIST:
            ck = _ck(f"{CKPT_DIR}/dfl_mdd_{n_stocks}_inds_h{horizon}"
                     f"_d{dv}_l{lv}_{SOLVER}.pkl")
            if ck is None:
                continue
            raw[(dv, lv)] = [(ck["fold_results_map"][(lb, n1)],
                              f"DFL-MDD (LB={lb}, n1={n1})") for lb, n1 in configs]
            infeas[(dv, lv)] = ck["infeas_map"]
    if not raw:
        raise RuntimeError(f"no DFL-MDD checkpoint for {n_stocks} inds, H={horizon}")
    dfl_store_cf = apply_carryforward(raw, infeas, folds=folds, full_np=full_np,
                                      HORIZON=horizon, REBAL=REBAL, d=d, C=C,
                                      verbose=False)

    # -- DFL-MVO (raw, as in the tables) --
    dfl_mvo_idx = {}
    for dv in DELTA_LIST:
        for lv in LAM_LIST:
            ck = _ck(f"{CKPT_DIR}/dfl_mvo_{n_stocks}_inds_h{horizon}"
                     f"_d{dv}_l{lv}_{SOLVER}.pkl")
            if ck is None:
                continue
            dfl_mvo_idx[(dv, lv)] = [
                (attach_date_idx(ck["fold_results_map"][lb], folds, lb, horizon, REBAL),
                 f"DFL-MVO (LB={lb})")
                for lb in LOOKBACK_LIST if lb in ck["fold_results_map"]]

    # -- PTO-MDD / PTO-MVO --
    ck = _ck(f"{CKPT_DIR}/pto_mdd_{n_stocks}_inds_h{horizon}"
             f"_d{DELTA_LIST[0]}_{SOLVER}.pkl")
    pto_mdd_idx = [(attach_date_idx(ck["fold_results_map"][(lb, n1)], folds,
                                    lb, horizon, REBAL),
                    f"PTO-MDD (LB={lb}, n1={n1})") for lb, n1 in configs]

    ck = _ck(f"{CKPT_DIR}/pto_mvo_{n_stocks}_inds_h{horizon}"
             f"_d{DELTA_LIST[0]}_{SOLVER}.pkl")
    mvo_idx = [(attach_date_idx(ck["fold_results_map"][lb], folds, lb, horizon, REBAL),
                f"PTO-MVO (LB={lb})")
               for lb in LOOKBACK_LIST if lb in ck["fold_results_map"]]

    # -- EW / GMV / hist-MVO --
    ck = _ck(f"{CKPT_DIR}/bench_{n_stocks}_inds_h{horizon}.pkl")
    bench_store = ck["bench_store"]

    return dfl_store_cf, dfl_mvo_idx, pto_mdd_idx, mvo_idx, bench_store


def verify(n_stocks, horizon, stores, full_np, tc_rate, tol=5e-4):
    """Recompute every legend value and compare with the published CSV."""
    dfl_store_cf, dfl_mvo_idx, pto_mdd_idx, mvo_idx, bench_store = stores
    ref_path = f"{RSLT_DIR}/{n_stocks}_inds_h{horizon}_tc_full_cf.csv"
    if not os.path.exists(ref_path):
        print(f"  [verify] {ref_path} not found - skipped")
        return True
    ref = pd.read_csv(ref_path)
    tc_bps = int(round(tc_rate * 1e4))
    ref = ref[ref.tc_bps == tc_bps]

    bad = []
    for lam in LAM_LIST:
        entries = list(dfl_store_cf.get((DELTA_LIST[0], lam), []))
        entries += list(pto_mdd_idx) + list(mvo_idx)
        entries += list(dfl_mvo_idx.get((DELTA_LIST[0], lam), []))
        entries += [(r, l) for l, r in bench_store.items()]

        for res, lbl in entries:
            res_tc = (apply_tc(res, tc_rate, full_np=full_np, REBAL=REBAL)
                      if tc_rate else res)
            perf = compute_performance(res_tc)
            row = ref[(ref.lam == lam) & (ref.label == lbl)]
            if row.empty:
                bad.append((lam, lbl, "missing in CSV", None, None))
                continue
            for key, col, scale in (("MDD", "MDD(%)", 100.0),
                                    ("Calmar", "Calmar", 1.0)):
                got, want = perf[key] * scale, float(row.iloc[0][col])
                if abs(got - want) > (tol * scale if scale == 1 else 5e-2):
                    bad.append((lam, lbl, key, got, want))

    if bad:
        print(f"  [verify] {len(bad)} mismatches at tc={tc_bps}bps")
        for lam, lbl, key, got, want in bad[:15]:
            print(f"     lam={lam} {lbl:<28} {key}: plot={got} csv={want}")
        return False
    print(f"  [verify] OK - every legend value matches {os.path.basename(ref_path)} "
          f"at tc={tc_bps}bps")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", type=int, choices=[10, 30])
    ap.add_argument("--horizon", type=int, default=126)
    ap.add_argument("--tc", action="append", type=float)
    args = ap.parse_args()

    universes = args.data or [10, 30]
    tc_rates  = args.tc if args.tc is not None else [0.0]
    horizon   = args.horizon
    os.makedirs(PLOT_DIR, exist_ok=True)

    ok = True
    for n_stocks in universes:
        full_np, full_dates = load_universe(n_stocks)
        folds  = make_folds(full_np, full_dates, horizon)
        stores = build_stores(n_stocks, horizon, full_np, folds)
        dfl_store_cf, dfl_mvo_idx, pto_mdd_idx, mvo_idx, bench_store = stores

        print(f"\n{'='*72}\n  {n_stocks} industries, H={horizon}\n{'='*72}")
        print(f"  DFL-MDD {len(dfl_store_cf[(20, LAM_LIST[0])])} configs | "
              f"DFL-MVO {len(dfl_mvo_idx.get((20, LAM_LIST[0]), []))} | "
              f"PTO-MDD {len(pto_mdd_idx)} | PTO-MVO {len(mvo_idx)} | "
              f"benchmarks {len(bench_store)}")

        for tc in tc_rates:
            ok &= verify(n_stocks, horizon, stores, full_np, tc)
            plot_overall_comparison(
                dfl_store_cf, pto_mdd_idx, mvo_idx,
                DELTA_LIST, LAM_LIST, LOOKBACK_LIST,
                n_stocks, PLOT_DIR,
                full_dates=full_dates,
                test_start_idx=folds[0]["test_start_idx"],
                tc_rate=tc, bench_store=bench_store,
                dfl_mvo_store=dfl_mvo_idx,
                full_np=full_np, REBAL=REBAL,
                horizon=horizon, show=False)

    print("\nall figures verified" if ok else "\nMISMATCHES FOUND - see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
