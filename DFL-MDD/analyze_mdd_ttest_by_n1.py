"""
Per-window drawdown tests, one test per drawdown budget.

The tests reported so far average the four budgets into a single DFL-MDD series
before pairing, which answers "is DFL-MDD as a family better" but not "is the
model with this budget better". A budget is what an investor actually picks, so
this script keeps d-bar fixed and tests each one on its own.

For a fixed (universe, H, lookback, lambda, n1) the comparison set is matched on
everything the comparison model also has:

  DFL-MVO    same lookback and lambda (it has no budget)
  PTO-MDD    same lookback and the same budget
  PTO-MVO    same lookback
  EW         lookback-independent
  GMV        same lookback
  hist-MVO   same lookback

Windows are paired on rebalancing date over the dates common to every model, so
each test is a repeated-measures comparison on identical decision dates.

  H0: mean per-window drawdown of DFL-MDD >= that of the comparison
  H1: strictly less

Usage
-----
    python analyze_mdd_ttest_by_n1.py
    python analyze_mdd_ttest_by_n1.py --data 10 --horizon 126
    python analyze_mdd_ttest_by_n1.py --out results/custom.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from carryforward import parse_lb, parse_n1
from make_overall_plots import (DELTA_LIST, LAM_LIST, LOOKBACK_LIST, N1_LIST,
                                REBAL, build_stores, load_universe, make_folds)

RSLT_DIR = "./results"
ALPHAS   = (0.10, 0.05, 0.01)


def verdict(p_low, p_high, alpha):
    """'+' DFL-MDD better, '-' comparison better, '=' neither, at this alpha."""
    if p_low < alpha:
        return "+"
    if p_high < alpha:
        return "-"
    return "="


def per_date(pairs, common):
    """Mean per-window drawdown (%) by rebalancing date, restricted to `common`."""
    rec = {}
    for res, _ in pairs:
        for r in res:
            rec.setdefault(r["date_idx"], []).append(r["M_real"] * 100)
    s = pd.Series({k: float(np.mean(v)) for k, v in rec.items()})
    return s.reindex(common)


def common_dates(stores):
    dfl_store_cf, dfl_mvo_idx, pto_mdd_idx, mvo_idx, bench_store = stores
    key = (DELTA_LIST[0], LAM_LIST[0])
    sets = [
        {r["date_idx"] for res, _ in dfl_store_cf[key] for r in res},
        {r["date_idx"] for res, _ in dfl_mvo_idx.get(key, []) for r in res},
        {r["date_idx"] for res, _ in pto_mdd_idx for r in res},
        {r["date_idx"] for res, _ in mvo_idx for r in res},
    ]
    sets += [{r["date_idx"] for r in res} for res in bench_store.values()]
    return sorted(set.intersection(*[s for s in sets if s]))


def analyse(n_stocks, horizon, stores, common):
    dfl_store_cf, dfl_mvo_idx, pto_mdd_idx, mvo_idx, bench_store = stores
    by_lb = lambda pairs, lb: [(r, l) for r, l in pairs if parse_lb(l) == lb]

    rows = []
    for lb in LOOKBACK_LIST:
        for lam in LAM_LIST:
            key = (DELTA_LIST[0], lam)
            if key not in dfl_store_cf:
                continue
            for n1 in N1_LIST:
                base_pairs = [(r, l) for r, l in dfl_store_cf[key]
                              if parse_lb(l) == lb and parse_n1(l) == n1]
                if not base_pairs:
                    continue
                base = per_date(base_pairs, common)

                comps = {
                    "DFL-MVO" : by_lb(dfl_mvo_idx.get(key, []), lb),
                    "PTO-MDD" : [(r, l) for r, l in pto_mdd_idx
                                 if parse_lb(l) == lb and parse_n1(l) == n1],
                    "PTO-MVO" : by_lb(mvo_idx, lb),
                    "EW"      : [(bench_store["EW"], "EW")],
                    "GMV"     : [(bench_store[f"GMV (LB={lb})"], f"GMV (LB={lb})")],
                    "hist-MVO": [(bench_store[f"hist-MVO (LB={lb})"],
                                  f"hist-MVO (LB={lb})")],
                }

                for name, pairs in comps.items():
                    if not pairs:
                        continue
                    other = per_date(pairs, common)
                    ok    = base.notna() & other.notna()
                    x, y  = base[ok].values, other[ok].values
                    if len(x) < 3:
                        continue
                    diff = x - y

                    t_st, p2 = stats.ttest_rel(x, y)
                    p_low  = p2 / 2 if t_st < 0 else 1 - p2 / 2   # H1: DFL-MDD lower
                    p_high = 1 - p_low
                    try:
                        _, wp2 = stats.wilcoxon(x, y)
                        wp = wp2 / 2 if np.median(diff) < 0 else 1 - wp2 / 2
                    except ValueError:
                        wp = float("nan")

                    row = {"N": n_stocks, "H": horizon, "LB": lb, "lam": lam,
                           "n1": n1, "comparison": name, "n": int(len(x)),
                           "DFL-MDD": x.mean(), "other": y.mean(),
                           "difference": diff.mean(),
                           "cohen_d": abs(diff.mean() / diff.std(ddof=1)),
                           "t": t_st, "p_onesided": p_low, "wilcoxon_p": wp}
                    for a in ALPHAS:
                        row[f"significant({a:.2f})"] = verdict(p_low, p_high, a)
                    rows.append(row)
    return pd.DataFrame(rows)


def summarise(df, alpha=0.05):
    """How many (LB, lambda) cells favour DFL-MDD, per budget and comparison."""
    col = f"significant({alpha:.2f})"
    tab = (df.assign(win=df[col].eq("+"), loss=df[col].eq("-"))
             .groupby(["comparison", "n1"])
             .agg(wins=("win", "sum"), losses=("loss", "sum"), cells=("win", "size")))
    tab["win_rate"] = (tab["wins"] / tab["cells"]).round(3)
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", type=int, choices=[10, 30])
    ap.add_argument("--horizon", type=int, default=126)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default=None,
                    help="output path; the default is "
                         "results/{N}_inds_h{H}_mdd_ttest_by_n1.csv per universe")
    args = ap.parse_args()

    os.makedirs(RSLT_DIR, exist_ok=True)
    for n_stocks in (args.data or [10, 30]):
        full_np, full_dates = load_universe(n_stocks)
        folds  = make_folds(full_np, full_dates, args.horizon)
        stores = build_stores(n_stocks, args.horizon, full_np, folds)
        common = common_dates(stores)

        df = analyse(n_stocks, args.horizon, stores, common)
        out = args.out or f"{RSLT_DIR}/{n_stocks}_inds_h{args.horizon}_mdd_ttest_by_n1.csv"
        df.to_csv(out, index=False, encoding="utf-8-sig")

        print(f"\n{'='*78}")
        print(f"  {n_stocks} industries, H={args.horizon} | "
              f"{len(common)} common rebalancing dates | {len(df)} tests")
        print(f"{'='*78}")
        print(summarise(df, args.alpha).to_string())
        print(f"\n  saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
