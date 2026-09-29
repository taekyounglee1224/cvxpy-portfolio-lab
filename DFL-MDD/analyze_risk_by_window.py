"""
Drawdown and Calmar tests at several evaluation horizons, on the monthly strategy.

The strategy is unchanged: weights are set every 21 trading days and the daily
path is the one the backtest produces. What changes is the window the drawdown
is measured over. For each rebalancing date the maximum drawdown of the realized
path is taken over the next H trading days, H in {21, 126, 252}, so the windows
slide by 21 days and overlap once H exceeds that.

  H = 21   the holding period, non-overlapping, what earlier tests reported
  H = 126  the constraint horizon
  H = 252  one year

Overlap makes the paired observations dependent — at H = 252 two neighbouring
windows share 231 of 252 days — and the ordinary paired t-test then understates
its p-values. Every test is therefore reported twice: the ordinary statistic and
a Newey-West statistic with H/21 - 1 lags, which is the overlap length in units
of the sliding step.

Two metrics are tested per window. Drawdown is better when lower, Calmar when
higher, so the one-sided alternative flips between them:

  MDD     H0: mean drawdown of DFL-MDD >= comparison    H1: strictly less
  Calmar  H0: mean Calmar of DFL-MDD <= comparison      H1: strictly greater

Calmar over a window is the annualized return implied by that window divided by
its drawdown. At 21 days that annualizes a single month and is correspondingly
noisy; the 126- and 252-day windows are the ones to read.

The drawdown is measured on the compounded path, and a date touched by two
windows at a fold boundary is counted once. Both differ from the per-window
figure the earlier tables report, which is additive and accumulates a fixed 21
days; at H = 21 the two agree to within about two percent and give the same
verdicts, but they are not the same number.

Usage
-----
    python analyze_mdd_by_window.py
    python analyze_mdd_by_window.py --data 10 --horizon 126
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from carryforward import parse_lb, parse_n1
from make_overall_plots import (DELTA_LIST, LAM_LIST, LOOKBACK_LIST, N1_LIST,
                                REBAL, build_stores, load_universe, make_folds)

RSLT_DIR   = "./results"
EVAL_WINS  = [21, 126, 252]
METRICS    = ["MDD", "Calmar"]
MDD_FLOOR  = 0.005          # 0.5%: below this the Calmar denominator is noise
ALPHAS     = (0.10, 0.05, 0.01)
ROUND      = 4


def verdict(p_low, p_high, alpha):
    if p_low < alpha:
        return "O"
    if p_high < alpha:
        return "X"
    return ""


def daily_path(pairs):
    """Daily returns of the strategy, by date index, averaged over configs.

    Each window contributes the days it is actually held, so a date that two
    windows touch at a fold boundary is not counted twice.
    """
    acc = {}
    for res, _ in pairs:
        t = [r["date_idx"] for r in res]
        for i, r in enumerate(res):
            w_real = np.asarray(r["w_real"], dtype=float)
            daily  = np.diff(np.concatenate([[0.0], w_real]))
            hold   = min(len(daily), t[i + 1] - t[i]) if i + 1 < len(t) else len(daily)
            for k in range(max(hold, 0)):
                acc.setdefault(t[i] + k, []).append(daily[k])
    s = pd.Series({k: float(np.mean(v)) for k, v in acc.items()})
    return s.sort_index()


def rolling_metric(path, starts, win, metric):
    """`metric` of the compounded path over `win` days from each start.

    MDD is a percentage; Calmar is the annualized return of the window over its
    drawdown, which is why a window whose drawdown is negligible is dropped
    rather than allowed to produce an unbounded ratio.
    """
    val = path.values
    pos = {d: i for i, d in enumerate(path.index.values)}
    out = {}
    for t in starts:
        i = pos.get(t)
        if i is None or i + win > len(val):
            continue
        eq   = np.cumprod(1.0 + val[i:i + win])
        peak = np.maximum.accumulate(eq)
        mdd  = float(np.max((peak - eq) / peak))
        if metric == "MDD":
            out[t] = mdd * 100.0
        else:
            if mdd < MDD_FLOOR:
                continue
            out[t] = (float(eq[-1]) ** (252.0 / win) - 1.0) / mdd
    return pd.Series(out)


def nw_tstat(diff, lags):
    """One-sided t-statistic for mean(diff) < 0 with Newey-West standard error."""
    n = len(diff)
    x = diff - diff.mean()
    gamma0 = float(x @ x) / n
    s = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        g = float(x[l:] @ x[:-l]) / n
        s += 2.0 * (1.0 - l / (lags + 1.0)) * g
    s = max(s, 1e-18)
    se = np.sqrt(s / n)
    t = diff.mean() / se
    return t, stats.t.cdf(t, df=n - 1)          # P(T <= t): one-sided, DFL lower


def analyse(n_stocks, horizon, stores, win, metric):
    dfl_store_cf, dfl_mvo_idx, pto_mdd_idx, mvo_idx, bench_store = stores
    by_lb = lambda pairs, lb: [(r, l) for r, l in pairs if parse_lb(l) == lb]
    lags  = max(win // REBAL - 1, 0)

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
                base_path  = daily_path(base_pairs)
                starts     = [r["date_idx"] for r in base_pairs[0][0]]

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

                a = rolling_metric(base_path, starts, win, metric)
                for name, pairs in comps.items():
                    if not pairs:
                        continue
                    b = rolling_metric(daily_path(pairs), starts, win, metric)
                    common = a.index.intersection(b.index)
                    if len(common) < 5:
                        continue
                    x, y = a.loc[common].values, b.loc[common].values
                    diff = x - y

                    # DFL-MDD is better with a lower drawdown but a higher Calmar,
                    # so the statistic is signed to favour it either way
                    sgn = 1.0 if metric == "MDD" else -1.0
                    t_st, p2 = stats.ttest_rel(x, y)
                    t_st *= sgn
                    p_low  = p2 / 2 if t_st < 0 else 1 - p2 / 2
                    t_nw, p_nw = nw_tstat(sgn * diff, lags)

                    row = {"N": n_stocks, "H_train": horizon, "metric": metric,
                           "eval_window": win,
                           "overlap_lags": lags, "LB": lb, "lam": lam, "d_bar": n1,
                           "comparison": name, "n": int(len(x)),
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
    ap.add_argument("--data", action="append", type=int, choices=[10, 30])
    ap.add_argument("--horizon", type=int, default=126)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(RSLT_DIR, exist_ok=True)

    for n_stocks in (args.data or [10, 30]):
        full_np, full_dates = load_universe(n_stocks)
        folds  = make_folds(full_np, full_dates, args.horizon)
        stores = build_stores(n_stocks, args.horizon, full_np, folds)

        # one file per (universe, evaluation window, metric): each is a table a
        # reader compares within, and mixing metrics in one file would put two
        # different one-sided alternatives in the same column
        parts, outs = [], []
        for m in METRICS:
            for w in EVAL_WINS:
                part = analyse(n_stocks, args.horizon, stores, w, m)
                num = part.select_dtypes("number").columns.difference(
                    ["N", "H_train", "eval_window", "overlap_lags", "LB", "n"])
                part[num] = part[num].round(ROUND)
                out = (f"{RSLT_DIR}/{n_stocks}_inds_h{args.horizon}"
                       f"_w{w}_{m.lower()}.csv")
                part.to_csv(out, index=False, encoding="utf-8-sig")
                parts.append(part); outs.append(out)
        df = pd.concat(parts, ignore_index=True)

        col, colnw = f"significant({args.alpha:.2f})", f"significant_nw({args.alpha:.2f})"
        summ = (df.assign(win=df[col].eq("O"), loss=df[col].eq("X"),
                          win_nw=df[colnw].eq("O"), loss_nw=df[colnw].eq("X"))
                  .groupby(["metric", "eval_window", "comparison"])
                  .agg(n_obs=("n", "mean"), cells=("win", "size"),
                       wins=("win", "sum"), losses=("loss", "sum"),
                       wins_nw=("win_nw", "sum"), losses_nw=("loss_nw", "sum")))
        print(f"\n{'='*88}")
        print(f"  {n_stocks} industries, trained at H={args.horizon} | "
              f"alpha={args.alpha} | plain vs Newey-West")
        print(f"{'='*88}")
        print(summ.to_string())
        print("\n  saved:")
        for o in outs:
            print(f"    {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
