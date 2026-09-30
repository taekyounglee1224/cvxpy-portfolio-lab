"""
Drawdown and Calmar tests under a per-asset weight cap.

The weight-cap sweep was run for 10 industries only, so this covers that
universe. For each cap the capped DFL-MDD is compared against the capped
versions of every alternative: the ablations and PTO baselines come from their
own capped checkpoints, and GMV and hist-MVO are re-solved with the same cap,
since comparing a capped model against uncapped benchmarks would confound the
cap with the model. Equal weighting is 1/m, which already satisfies every cap
considered here, so it is unchanged.

Windows are the 126-day evaluation windows sliding by 21 days, as in
analyze_risk_by_window, with the Newey-West statistic reported alongside the
ordinary one.

Usage
-----
    python analyze_xmax_tests.py
    python analyze_xmax_tests.py --xmax 0.3 --horizon 126
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from benchmarks import attach_date_idx, build_bench_store
from carryforward import apply_carryforward, parse_lb, parse_n1
from make_overall_plots import (DELTA_LIST, LAM_LIST, LOOKBACK_LIST, N1_LIST,
                                REBAL, _ck, load_universe, make_folds)
from analyze_risk_by_window import (ALPHAS, MDD_FLOOR, ROUND, daily_path,
                                    nw_tstat, rolling_metric, verdict)

RSLT_DIR = "./results"
XMAXES   = [0.2, 0.3, 0.6]
METRICS  = ["MDD", "Calmar"]
EVAL_WIN = 126
SOLVER   = "CLARABEL"
C, d     = 1.0, 1.0


def build_capped_stores(n_stocks, horizon, xmax, full_np, full_dates, folds,
                        stock_names):
    """The same five stores build_stores returns, but every model capped."""
    tag = f"_xm{xmax:g}"
    configs = [(lb, n1) for lb in LOOKBACK_LIST for n1 in N1_LIST]

    raw, infeas = {}, {}
    for lam in LAM_LIST:
        ck = _ck(f"./checkpoint/dfl_mdd_{n_stocks}_inds_h{horizon}{tag}"
                 f"_d{DELTA_LIST[0]}_l{lam}_{SOLVER}.pkl")
        if ck is None:
            continue
        raw[(DELTA_LIST[0], lam)] = [
            (ck["fold_results_map"][(lb, n1)], f"DFL-MDD (LB={lb}, n1={n1})")
            for lb, n1 in configs if (lb, n1) in ck["fold_results_map"]]
        infeas[(DELTA_LIST[0], lam)] = ck["infeas_map"]
    if not raw:
        raise RuntimeError(f"no capped DFL-MDD checkpoint for x_max={xmax}")
    dfl_cf = apply_carryforward(raw, infeas, folds=folds, full_np=full_np,
                                HORIZON=horizon, REBAL=REBAL, d=d, C=C,
                                verbose=False)

    dfl_mvo = {}
    for lam in LAM_LIST:
        ck = _ck(f"./checkpoint/dfl_mvo_{n_stocks}_inds_h{horizon}{tag}"
                 f"_d{DELTA_LIST[0]}_l{lam}_{SOLVER}.pkl")
        if ck is None:
            continue
        dfl_mvo[(DELTA_LIST[0], lam)] = [
            (attach_date_idx(ck["fold_results_map"][lb], folds, lb, horizon, REBAL),
             f"DFL-MVO (LB={lb})")
            for lb in LOOKBACK_LIST if lb in ck["fold_results_map"]]

    ck = _ck(f"./checkpoint/pto_mdd_{n_stocks}_inds_h{horizon}{tag}"
             f"_d{DELTA_LIST[0]}_{SOLVER}.pkl")
    pto_mdd = [(attach_date_idx(ck["fold_results_map"][(lb, n1)], folds, lb,
                                horizon, REBAL),
                f"PTO-MDD (LB={lb}, n1={n1})")
               for lb, n1 in configs if (lb, n1) in ck["fold_results_map"]]

    ck = _ck(f"./checkpoint/pto_mvo_{n_stocks}_inds_h{horizon}{tag}"
             f"_d{DELTA_LIST[0]}_{SOLVER}.pkl")
    pto_mvo = [(attach_date_idx(ck["fold_results_map"][lb], folds, lb, horizon, REBAL),
                f"PTO-MVO (LB={lb})")
               for lb in LOOKBACK_LIST if lb in ck["fold_results_map"]]

    # GMV and hist-MVO are cheap to re-solve, and must carry the same cap
    bench, _ = build_bench_store(full_np, folds, stock_names, LOOKBACK_LIST,
                                 horizon, REBAL, delta=DELTA_LIST[0],
                                 x_max=xmax, verbose=False)
    return dfl_cf, dfl_mvo, pto_mdd, pto_mvo, bench


def analyse(n_stocks, horizon, xmax, stores, metric, win=EVAL_WIN):
    dfl_cf, dfl_mvo, pto_mdd, pto_mvo, bench = stores
    by_lb = lambda pairs, lb: [(r, l) for r, l in pairs if parse_lb(l) == lb]
    lags  = max(win // REBAL - 1, 0)

    rows = []
    for lb in LOOKBACK_LIST:
        for lam in LAM_LIST:
            key = (DELTA_LIST[0], lam)
            if key not in dfl_cf:
                continue
            for n1 in N1_LIST:
                base_pairs = [(r, l) for r, l in dfl_cf[key]
                              if parse_lb(l) == lb and parse_n1(l) == n1]
                if not base_pairs:
                    continue
                starts = [r["date_idx"] for r in base_pairs[0][0]]
                a = rolling_metric(daily_path(base_pairs), starts, win, metric)

                comps = {
                    "DFL-MVO" : by_lb(dfl_mvo.get(key, []), lb),
                    "PTO-MDD" : [(r, l) for r, l in pto_mdd
                                 if parse_lb(l) == lb and parse_n1(l) == n1],
                    "PTO-MVO" : by_lb(pto_mvo, lb),
                    "EW"      : [(bench["EW"], "EW")],
                    "GMV"     : [(bench[f"GMV (LB={lb})"], f"GMV (LB={lb})")],
                    "hist-MVO": [(bench[f"hist-MVO (LB={lb})"],
                                  f"hist-MVO (LB={lb})")],
                }
                for name, pairs in comps.items():
                    if not pairs:
                        continue
                    b = rolling_metric(daily_path(pairs), starts, win, metric)
                    common = a.index.intersection(b.index)
                    if len(common) < 5:
                        continue
                    x, y = a.loc[common].values, b.loc[common].values
                    diff = x - y
                    sgn  = 1.0 if metric == "MDD" else -1.0

                    t_st, p2 = stats.ttest_rel(x, y)
                    t_st *= sgn
                    p_low = p2 / 2 if t_st < 0 else 1 - p2 / 2
                    t_nw, p_nw = nw_tstat(sgn * diff, lags)

                    row = {"N": n_stocks, "H_train": horizon, "x_max": xmax,
                           "metric": metric, "eval_window": win,
                           "overlap_lags": lags, "LB": lb, "lam": lam,
                           "d_bar": n1, "comparison": name, "n": int(len(x)),
                           "DFL-MDD": x.mean(), "other": y.mean(),
                           "difference": diff.mean(),
                           "cohen_d": abs(diff.mean() / diff.std(ddof=1)),
                           "t": t_st, "p_onesided": p_low,
                           "t_nw": t_nw, "p_nw": p_nw}
                    for al in ALPHAS:
                        row[f"significant({al:.2f})"] = verdict(p_low, 1 - p_low, al)
                    for al in ALPHAS:
                        row[f"significant_nw({al:.2f})"] = verdict(p_nw, 1 - p_nw, al)
                    rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=int, default=10, choices=[10, 30])
    ap.add_argument("--horizon", type=int, default=126)
    ap.add_argument("--xmax", action="append", type=float)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(RSLT_DIR, exist_ok=True)

    inds = pd.read_csv(f"csv/{args.data}_industry.csv")
    stock_names = [c for c in inds.columns if c != "Date"]
    full_np, full_dates = load_universe(args.data)
    folds = make_folds(full_np, full_dates, args.horizon)

    summary = []
    for xmax in (args.xmax or XMAXES):
        stores = build_capped_stores(args.data, args.horizon, xmax, full_np,
                                     full_dates, folds, stock_names)
        for metric in METRICS:
            df = analyse(args.data, args.horizon, xmax, stores, metric)
            num = df.select_dtypes("number").columns.difference(
                ["N", "H_train", "eval_window", "overlap_lags", "LB", "n"])
            df[num] = df[num].round(ROUND)
            out = (f"{RSLT_DIR}/{args.data}_inds_h{args.horizon}"
                   f"_xm{xmax:g}_w{EVAL_WIN}_{metric.lower()}.csv")
            df.to_csv(out, index=False, encoding="utf-8-sig")
            print(f"  saved: {out}  ({len(df)} rows)")

            col = f"significant_nw({args.alpha:.2f})"
            summary.append(df.assign(win=df[col].eq("O"), loss=df[col].eq("X")))

    s = pd.concat(summary, ignore_index=True)
    print(f"\n{'='*72}")
    print(f"  {args.data} industries, H={args.horizon}, {EVAL_WIN}-day window | "
          f"alpha={args.alpha}, Newey-West")
    print(f"{'='*72}")
    print(s.groupby(["metric", "x_max", "comparison"])
           .agg(favourable=("win", "sum"), unfavourable=("loss", "sum"),
                cells=("win", "size")).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
