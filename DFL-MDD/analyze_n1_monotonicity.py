"""
Reviewer #2, comment 13 -- does a tighter drawdown budget (n1) actually reduce
realized risk?

The aggregate 8-year maximum drawdown shows no ordering in n1. This script tests
the ordering at the level the constraint actually operates on: the per-window,
H-day, additive drawdown (`M_real`), which is the same quantity the optimization
constrains and the training loss penalises.

Windows are paired across n1: within a fixed (universe, H, LB, lam) the four n1
runs share the same rebalancing dates in the same order, so window i is the same
decision date under four different budgets. That pairing makes this a repeated-
measures design, and lets us use:

  * Page's L      - trend test against the ordered alternative
                    theta(0.1) <= theta(0.2) <= theta(0.3) <= theta(0.4)
  * paired t      - consecutive levels and the 0.1 vs 0.4 endpoints
  * Wilcoxon      - distribution-free counterpart of the same pairs

Usage
-----
    python analyze_n1_monotonicity.py
    python analyze_n1_monotonicity.py --feasible-only
    python analyze_n1_monotonicity.py --out results/n1_monotonicity.csv
"""

import os
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy import stats

CKPT_DIR = "./checkpoint"
SOLVER = "CLARABEL"
DELTA = 20
LAM_LIST = [0.3, 0.5, 0.7, 1.0]
N1_LIST = [0.1, 0.2, 0.3, 0.4]
LB_LIST = [252, 504]


# ----------------------------------------------------------------------------
# Page's L trend test (repeated measures, ordered alternative)
# ----------------------------------------------------------------------------
def page_trend_test(X):
    """X: (n_blocks, k) with columns already in the hypothesised increasing order.

    Returns (L, z, p_one_sided). Ranks run 1..k within each block, so a perfectly
    increasing block contributes ranks 1,2,...,k.
    """
    n, k = X.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, X)      # 1 = smallest in block
    R = ranks.sum(axis=0)                                   # rank total per level
    L = float(np.sum(np.arange(1, k + 1) * R))
    EL = n * k * (k + 1) ** 2 / 4.0
    VL = n * k ** 2 * (k + 1) * (k ** 2 - 1) / 144.0
    z = (L - EL) / np.sqrt(VL)
    return L, z, stats.norm.sf(z)                           # H1: increasing


# ----------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------
def fold_offsets(infeas_map, key):
    """Cumulative start index of each fold inside the concatenated window list."""
    logs = sorted(infeas_map.get(key, []), key=lambda e: e["fold"])
    off, acc = {}, 0
    for e in logs:
        off[e["fold"]] = acc
        acc += e["n_windows"]
    return off, logs


def load_long(n_stocks, horizon):
    """Long frame: one row per (LB, lam, window, n1)."""
    rows = []
    for lam in LAM_LIST:
        p = (f"{CKPT_DIR}/dfl_mdd_{n_stocks}_inds_h{horizon}"
             f"_d{DELTA}_l{lam}_{SOLVER}.pkl")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        with open(p, "rb") as f:
            ck = pickle.load(f)
        fm, im = ck["fold_results_map"], ck["infeas_map"]

        for lb, n1 in itertools.product(LB_LIST, N1_LIST):
            key = (lb, n1)
            off, logs = fold_offsets(im, key)
            failed = {off[e["fold"]] + w
                      for e in logs for w in e.get("failed_windows", [])}
            for i, r in enumerate(fm[key]):
                rows.append({"LB": lb, "lam": lam, "n1": n1, "w": i,
                             "M_real": float(r["M_real"]),
                             "infeasible": i in failed})
    d = pd.DataFrame(rows)
    d["N"], d["H"] = n_stocks, horizon
    return d


def drop_infeasible(d):
    """Remove every (LB, lam, window) block that was infeasible under any n1."""
    bad = d[d.infeasible].set_index(["LB", "lam", "w"]).index.unique()
    return (d.set_index(["LB", "lam", "w"])
             .drop(index=bad, errors="ignore").reset_index())


def to_blocks(d):
    """Pivot to a (block x n1) matrix, dropping blocks with any missing cell."""
    wide = d.pivot_table(index=["LB", "lam", "w"], columns="n1", values="M_real")
    return wide.dropna()[N1_LIST]


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
def analyse(X, label):
    n = len(X)
    A = X.values
    L, z, p = page_trend_test(A)

    out = {"group": label, "blocks": n,
           **{f"mean_n1={c}": A[:, j].mean() for j, c in enumerate(N1_LIST)},
           "Page_L": L, "Page_z": z, "Page_p": p}

    # endpoint and consecutive paired comparisons, H1: higher n1 => larger M_real
    pairs = list(zip(N1_LIST[:-1], N1_LIST[1:])) + [(N1_LIST[0], N1_LIST[-1])]
    for a, b in pairs:
        lo, hi = A[:, N1_LIST.index(a)], A[:, N1_LIST.index(b)]
        diff = hi - lo
        t, p2 = stats.ttest_rel(hi, lo)
        out[f"d({b}-{a})"] = diff.mean()
        out[f"t({b}>{a})"] = t
        out[f"p_t({b}>{a})"] = p2 / 2 if t > 0 else 1 - p2 / 2
        try:
            _, w2 = stats.wilcoxon(hi, lo)
            out[f"p_W({b}>{a})"] = w2 / 2 if np.median(diff) > 0 else 1 - w2 / 2
        except ValueError:
            out[f"p_W({b}>{a})"] = np.nan
        out[f"win%({b}>{a})"] = 100 * (diff > 0).mean()
    return out


def violation_table(d):
    """Share of windows whose realized drawdown exceeds its own budget."""
    out = []
    for n1, g in d.groupby("n1"):
        v = g.M_real.values
        out.append({"n1": n1, "mean_M_real": v.mean(), "median_M_real": np.median(v),
                    "p90_M_real": np.percentile(v, 90),
                    "violation_rate": (v > n1).mean(), "windows": len(v)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feasible-only", action="store_true",
                    help="drop every window that was infeasible under any n1")
    ap.add_argument("--outdir", default="results",
                    help="directory for the per-universe CSV files")
    args = ap.parse_args()

    suffix = "_feasible_only" if args.feasible_only else ""
    os.makedirs(args.outdir, exist_ok=True)

    for n_stocks in (10, 30):
        recs, viol = [], []
        for horizon in (126, 252):
            d = load_long(n_stocks, horizon)
            if args.feasible_only:
                d = drop_infeasible(d)      # both tables must see the same rows
            X = to_blocks(d)
            recs.append({"N": n_stocks, "H": horizon,
                         **analyse(X, f"{n_stocks} inds, H={horizon}")})
            v = violation_table(d)
            v.insert(0, "H", horizon)
            v.insert(0, "N", n_stocks)
            viol.append(v)

            print(f"\n=== {n_stocks} industries, H={horizon} "
                  f"({len(X)} paired windows) ===")
            m = X.mean()
            print("  mean per-window drawdown (M_real) by budget")
            for n1 in N1_LIST:
                print(f"     n1={n1}   {m[n1]*100:6.3f}%")
            L, z, p = page_trend_test(X.values)
            print(f"  Page's L trend test:  L={L:.0f}  z={z:+.2f}  p={p:.3e}")
            lo, hi = X[N1_LIST[0]].values, X[N1_LIST[-1]].values
            t, p2 = stats.ttest_rel(hi, lo)
            print(f"  paired t (0.4 > 0.1):  t={t:+.2f}  "
                  f"p={p2/2 if t>0 else 1-p2/2:.3e}  "
                  f"mean diff={100*(hi-lo).mean():+.3f}%p")

        base = f"{args.outdir}/{n_stocks}_inds_n1_monotonicity{suffix}"
        pd.DataFrame(recs).to_csv(f"{base}.csv", index=False, encoding="utf-8-sig")
        pd.concat(viol).to_csv(f"{base}_violations.csv", index=False,
                               encoding="utf-8-sig")
        print(f"\n  saved: {base}.csv")
        print(f"  saved: {base}_violations.csv")


if __name__ == "__main__":
    main()
