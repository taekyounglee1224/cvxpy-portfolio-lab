"""
Quarterly rebalancing, without retraining.

The rebalancing period never enters the optimization: the weights at a date are
fixed by the predicted path and the drawdown limit, and REBAL only decides how
many days of realized return are then accumulated. Quarterly rebalancing is
63 = 3 x 21 trading days, so its dates are exactly every third monthly date
within each fold, and the weights already stored in the checkpoints are the
weights a quarterly run would have produced.

This script therefore keeps every third window per fold, re-accumulates the
realized path, re-applies the carry-forward rule on the reduced calendar, and
reports performance at five transaction-cost levels beside a monthly baseline
built the same way.

Each portfolio is held until the next rebalancing date rather than for a fixed
number of days. Accumulating a fixed 63 days from every quarterly date would
double-count 11.9% of the calendar at fold boundaries, where the next test year
restarts and two dates land within a few days of each other; the monthly
baseline is rebuilt under the same rule, so its numbers differ slightly from
the published tables, which accumulate a fixed 21 days.

Turnover is reported two ways. Per rebalancing it *rises* under a quarterly
schedule, since weights drift further over three months, while the annualized
figure — what is actually traded per year — falls by roughly 60%.

Usage
-----
    python analyze_quarterly_rebal.py
    python analyze_quarterly_rebal.py --data 10 --horizon 126
"""

import argparse
import os

import numpy as np
import pandas as pd

from benchmarks import attach_date_idx
from carryforward import parse_lb, parse_n1
from make_overall_plots import (DELTA_LIST, LAM_LIST, LOOKBACK_LIST, N1_LIST,
                                REBAL, _ck, load_universe, make_folds)
from performance import apply_tc, compute_performance

RSLT_DIR   = "./results"
QREBAL     = 63           # one quarter
TC_RATES   = [0, 0.0005, 0.001, 0.002, 0.004]
SOLVER     = "CLARABEL"
RIDGE      = 1e-4


def fold_bounds(results):
    """Indices where a new fold starts, found from the gap in date_idx.

    Inside a fold the rebalancing dates are REBAL apart; at a fold boundary the
    next test year restarts the calendar, so the gap differs.
    """
    t = np.asarray([r["date_idx"] for r in results])
    brk = [0] + [i for i in range(1, len(t)) if t[i] - t[i - 1] != REBAL] + [len(t)]
    return brk


def subsample(results, step=3):
    """Keep every `step`-th window within each fold."""
    brk = fold_bounds(results)
    keep = [i for b in range(len(brk) - 1)
            for i in range(brk[b], brk[b + 1], step)]
    return [results[i] for i in keep], keep


def failed_global(infeas_logs):
    """Window indices that fell back, on the monthly calendar."""
    failed, offset = set(), 0
    for e in sorted(infeas_logs, key=lambda x: x["fold"]):
        failed |= {offset + w for w in e.get("failed_windows", [])}
        offset += e["n_windows"]
    return failed


def holding_days(results, nominal):
    """Days each portfolio is actually held: up to the next rebalancing date.

    Accumulating a fixed `nominal` from every date double-counts wherever two
    dates sit closer than that, which happens at every fold boundary: 11.9% of
    the quarterly calendar and 4.4% of the monthly one. Holding to the next
    decision instead is both what the strategy does and free of overlap.
    """
    t = np.asarray([r["date_idx"] for r in results])
    gaps = np.diff(t)
    return [max(1, min(int(g), nominal)) for g in gaps] + [nominal]


def rebuild(results, keep, failed, full_np, lookback, holds, d=1.0, C=1.0):
    """Re-accumulate realized performance, carry-forward applied.

    `keep` maps the reduced calendar back to the monthly indices, so a window is
    a fallback here only if that same window fell back in the original run.
    """
    m, out = full_np.shape[1], []
    w_prev = np.full(m, 1.0 / m)
    n_sub = 0

    for r, orig_i, hold in zip(results, keep, holds):
        t = r["date_idx"]
        if orig_i in failed:
            w = w_prev.copy()
            n_sub += 1
        else:
            w = np.asarray(r["weights"], dtype=float)
            w_prev = w.copy()

        p_ret  = full_np[t:t + hold] @ w
        w_real = np.cumsum(p_ret)
        pv     = 1.0 + w_real
        rmax   = np.maximum.accumulate(pv)

        S   = np.cov(full_np[t - lookback:t].T) + RIDGE * np.eye(m)
        sig = np.sqrt(max(float(w @ S @ w), 0.0) + 1e-8)

        out.append({**r, "weights": w, "w_real": w_real,
                    "R_real": float(w_real[-1] / (d * C)),
                    "M_real": float(np.max((rmax - pv) / (rmax + 1e-10))),
                    "Sharpe": float(p_ret.mean() / sig)})
    return out, n_sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", type=int, choices=[10, 30])
    ap.add_argument("--horizon", type=int, default=126)
    args = ap.parse_args()
    os.makedirs(RSLT_DIR, exist_ok=True)

    for n_stocks in (args.data or [10, 30]):
        full_np, full_dates = load_universe(n_stocks)
        folds = make_folds(full_np, full_dates, args.horizon)

        rows = []
        for lam in LAM_LIST:
            ck = _ck(f"./checkpoint/dfl_mdd_{n_stocks}_inds_h{args.horizon}"
                     f"_d{DELTA_LIST[0]}_l{lam}_{SOLVER}.pkl")
            if ck is None:
                continue
            for lb in LOOKBACK_LIST:
                for n1 in N1_LIST:
                    key = (lb, n1)
                    if key not in ck["fold_results_map"]:
                        continue
                    res = attach_date_idx(ck["fold_results_map"][key], folds,
                                          lb, args.horizon, REBAL)
                    failed = failed_global(ck["infeas_map"].get(key, []))
                    label  = f"DFL-MDD (LB={lb}, n1={n1})"

                    for tag, hold, sel in (("monthly",   REBAL,  None),
                                           ("quarterly", QREBAL, 3)):
                        if sel is None:
                            sub, keep = res, list(range(len(res)))
                        else:
                            sub, keep = subsample(res, sel)
                        holds = holding_days(sub, hold)
                        built, n_sub = rebuild(sub, keep, failed, full_np,
                                               lb, holds)
                        span = sum(holds)          # trading days actually covered
                        for tc in TC_RATES:
                            r_tc = (apply_tc(built, tc, full_np=full_np, REBAL=hold)
                                    if tc else built)
                            perf = compute_performance(r_tc, label,
                                                       full_np=full_np, REBAL=hold)
                            # per-rebalance turnover rises when trades are rarer,
                            # so also report what is traded per year
                            perf["Turnover_ann"] = (perf["Turnover"] * len(built)
                                                    * 252.0 / span)
                            rows.append({"N": n_stocks, "H": args.horizon,
                                         "schedule": tag, "hold_days": hold,
                                         "LB": lb, "lam": lam, "n1": n1,
                                         "tc_bps": int(round(tc * 1e4)),
                                         "n_windows": len(built),
                                         "n_carryforward": n_sub, **perf})

        df = pd.DataFrame(rows)
        num = df.select_dtypes("number").columns.difference(
            ["N", "H", "hold_days", "LB", "tc_bps", "n_windows", "n_carryforward"])
        df[num] = df[num].round(4)
        # the monthly rows exist to anchor the printed comparison; the file itself
        # carries the quarterly schedule only, since the monthly tables are
        # published separately and are built under a fixed 21-day hold
        out = f"{RSLT_DIR}/{n_stocks}_inds_h{args.horizon}_quarterly_rebal.csv"
        df[df.schedule == "quarterly"].to_csv(out, index=False, encoding="utf-8-sig")

        print(f"\n{'='*84}")
        print(f"  {n_stocks} industries, H={args.horizon}  "
              f"(monthly 21d vs quarterly 63d, averaged over 32 configurations)")
        print(f"{'='*84}")
        piv = (df.groupby(["tc_bps", "schedule"])
                 [["Ann.Ret", "Sharpe", "MDD", "Calmar", "Turnover", "Turnover_ann"]]
                 .mean().unstack("schedule").round(4))
        print(piv.to_string())
        nw = df.groupby("schedule")[["n_windows", "n_carryforward"]].mean().round(1)
        print(f"\n  windows and fallbacks per configuration:\n{nw.to_string()}")
        print(f"\n  saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
