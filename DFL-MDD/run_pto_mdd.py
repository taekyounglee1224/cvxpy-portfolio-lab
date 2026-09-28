"""
run_pto_mdd.py
--------------
Train the PTO-MDD baseline for one lookback as a single process, so that the
lookbacks can run on separate cores.

PTO-MDD has no lambda, so a run is identified by (data, horizon, lookback).
The drawdown budgets n1 are looped inside the process because they share the
training set; pass --n1 to shard them across processes as well.

Usage
-----
  # one process per universe, all four budgets inside
  python run_pto_mdd.py --data 10 --lb 1260 > logs/pto_mdd_10_LB1260.txt 2>&1 &
  python run_pto_mdd.py --data 30 --lb 1260 > logs/pto_mdd_30_LB1260.txt 2>&1 &
  wait

  # shard the budgets too, then merge
  for B in 0.1 0.2 0.3 0.4; do
      python run_pto_mdd.py --data 10 --lb 1260 --n1 $B &
  done; wait
  python run_pto_mdd.py --data 10 --lb 1260 --merge

Checkpoints use the same format the notebooks read:
  pto_mdd_{N}_inds_h{H}[_LB{lb}][_n1{n1}]_d{delta}_{solver}.pkl
The _LB tag is written whenever --lb is given, so a lookback outside the default
grid stays a separate experiment.
"""

# -- pin BLAS threads before importing numpy/torch --
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import glob
import pickle
import random
import time
import warnings

warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)


# ══════════════════════════════════════════════════════════
# arguments
# ══════════════════════════════════════════════════════════
ap = argparse.ArgumentParser()
ap.add_argument("--data",    default="10", choices=["10", "30"])
ap.add_argument("--lb", type=int, nargs="+", default=None,
                help="LOOKBACK values. When given, the checkpoint carries a _LB tag")
ap.add_argument("--n1", type=float, nargs="+", default=None,
                help="drawdown budgets. A single value shards the run and tags the file")
ap.add_argument("--delta",   type=float, default=20.0)
ap.add_argument("--horizon", type=int,   default=126)
ap.add_argument("--solver",  default="CLARABEL")
ap.add_argument("--merge",   action="store_true",
                help="merge the _n1 shards for this (data, horizon, lb) and exit")
args = ap.parse_args()

SOLVER    = args.solver
DELTA_VAL = int(args.delta) if float(args.delta).is_integer() else args.delta
HORIZON   = args.horizon

LOOKBACK_LIST = args.lb if args.lb else [252, 504]
N1_LIST       = args.n1 if args.n1 else [0.1, 0.2, 0.3, 0.4]
_LBTAG        = f"_LB{'-'.join(map(str, LOOKBACK_LIST))}" if args.lb else ""
# a shard tag only when a single budget was requested
_N1TAG        = f"_n1{N1_LIST[0]:g}" if (args.n1 and len(args.n1) == 1) else ""

TAG = f"[PTO-MDD h{HORIZON} LB{args.lb or 'all'}{_N1TAG}]"
_T0 = time.time()


def _el():
    s = int(time.time() - _T0)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')} +{_el()}] {TAG} {msg}", flush=True)


CKPT_DIR = "./checkpoint"
os.makedirs(CKPT_DIR, exist_ok=True)
N_STOCKS_HINT = int(args.data) if args.data == "10" else None   # 30_industry has 30 columns


def ckpt_name(n_stocks, n1tag):
    return os.path.join(
        CKPT_DIR,
        f"pto_mdd_{n_stocks}_inds_h{HORIZON}{_LBTAG}{n1tag}_d{DELTA_VAL}_{SOLVER}.pkl")


# ══════════════════════════════════════════════════════════
# data (identical to the notebooks)
# ══════════════════════════════════════════════════════════
inds = pd.read_csv(f"csv/{args.data}_industry.csv")
inds["Date"] = pd.to_datetime(inds["Date"])
inds = inds.set_index("Date").sort_index()
inds = inds[~inds.index.duplicated(keep="first")] / 100.0

stock_names = inds.columns.tolist()
full_np     = inds.values
full_dates  = inds.index

gamma, x_min, x_max = 0.0, 0.0, 1.0
N_STOCKS   = len(inds.columns)
REBAL      = 21
N          = HORIZON
M          = N_STOCKS
C, d       = 1.0, 1.0
HIDDEN_DIM = 128
EPOCHS     = 100
BATCH_SIZE = 32
LR         = 1e-4
PATIENCE   = 20

VAL_YEARS, TEST_YEARS, N_FOLDS = 5, 1, 8
configs = [{"LOOKBACK": lb, "n1": n1}
           for lb in LOOKBACK_LIST for n1 in N1_LIST]


# ══════════════════════════════════════════════════════════
# merge mode: combine the _n1 shards and exit
# ══════════════════════════════════════════════════════════
if args.merge:
    out = ckpt_name(N_STOCKS, "")
    pattern = os.path.join(
        CKPT_DIR,
        f"pto_mdd_{N_STOCKS}_inds_h{HORIZON}{_LBTAG}_n1*_d{DELTA_VAL}_{SOLVER}.pkl")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise SystemExit(f"no shards matching {pattern}")

    merged, completed = {}, []
    for sp in shards:
        with open(sp, "rb") as f:
            ck = pickle.load(f)
        for k, v in ck["fold_results_map"].items():
            if k in merged:
                raise SystemExit(f"duplicate config {k} in {os.path.basename(sp)}")
            merged[k] = v
        completed.append(ck["completed_fold"])
        log(f"  + {os.path.basename(sp)}  configs {sorted(ck['fold_results_map'])}")

    if len(set(completed)) != 1:
        log(f"  warning: shards stopped at different folds {sorted(set(completed))}")
    with open(out, "wb") as f:
        pickle.dump({"fold_results_map": merged,
                     "completed_fold"  : min(completed),
                     "delta_val"       : DELTA_VAL,
                     "horizon"         : HORIZON,
                     "x_max"           : x_max,
                     "solver"          : SOLVER}, f)
    log(f"merged {len(shards)} shards -> {out}  ({len(merged)} configs)")
    raise SystemExit(0)


def date_to_idx(s):
    return full_dates.searchsorted(pd.Timestamp(s), side="left")


folds = []
for f in range(N_FOLDS):
    ty = 2018 + f
    folds.append({
        "fold"          : f + 1,
        "train_end_idx" : date_to_idx(f"{ty - VAL_YEARS}-01-01"),
        "val_start_idx" : date_to_idx(f"{ty - VAL_YEARS}-01-01"),
        "val_end_idx"   : date_to_idx(f"{ty}-01-01"),
        "test_start_idx": date_to_idx(f"{ty}-01-01"),
        "test_end_idx"  : min(date_to_idx(f"{ty + TEST_YEARS}-01-01") + HORIZON,
                              len(full_np)),
        "test_year"     : ty,
    })

init_train_end = date_to_idx("2013-01-01")
is_mean = full_np[:init_train_end].mean(axis=0)
is_std  = full_np[:init_train_end].std(axis=0)


def make_windows(data, lookback, horizon, start, end):
    samples = []
    for t in range(max(start, lookback), end - horizon + 1):
        z_raw  = data[t - lookback:t]
        z_norm = (z_raw - is_mean) / (is_std + 1e-8)
        samples.append((z_norm.flatten(), data[t:t + horizon]))
    return samples


# ══════════════════════════════════════════════════════════
# training
# ══════════════════════════════════════════════════════════
from dfl_mdd import PredictionModel
from pto_mdd import train_pto_mdd, backtest_pto_mdd

ckpt_path = ckpt_name(N_STOCKS, _N1TAG)

log(f"data {args.data} industries ({N_STOCKS} assets, {len(full_np)} days)")
log(f"HORIZON={HORIZON}, solver={SOLVER}, delta={DELTA_VAL}, "
    f"config {len(configs)} x fold {N_FOLDS}")
log(f"checkpoint: {ckpt_path}")

if os.path.exists(ckpt_path):
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    fold_results_map = ck["fold_results_map"]
    start_fold       = ck["completed_fold"] + 1
    log(f"checkpoint loaded: through fold {ck['completed_fold']}")
else:
    fold_results_map = {(c["LOOKBACK"], c["n1"]): [] for c in configs}
    start_fold       = 1

t_start = time.time()

for fold_info in folds:
    fold_id = fold_info["fold"]
    if fold_id < start_fold:
        log(f"fold {fold_id} skipped")
        continue

    log(f"-- fold {fold_id} (test={fold_info['test_year']}) --")
    # seeded per fold, matching the PTO-MDD cell in the notebooks
    torch.manual_seed(123); np.random.seed(123); random.seed(123)

    for cfg in configs:
        LOOKBACK, n1 = cfg["LOOKBACK"], cfg["n1"]
        t0 = time.time()

        train_samples = make_windows(full_np, LOOKBACK, HORIZON,
                                     start=LOOKBACK,
                                     end=fold_info["train_end_idx"])
        val_samples   = make_windows(full_np, LOOKBACK, HORIZON,
                                     start=fold_info["val_start_idx"],
                                     end=fold_info["val_end_idx"])[::HORIZON]
        rebal_samples = make_windows(full_np, LOOKBACK, HORIZON,
                                     start=fold_info["test_start_idx"],
                                     end=fold_info["test_end_idx"])[::REBAL]

        # a long HORIZON can leave the later folds with no rebalancing window
        if len(rebal_samples) == 0:
            log(f"   LB={LOOKBACK}, n1={n1}  no rebalancing window - skipped")
            continue

        model = PredictionModel(LOOKBACK * N_STOCKS, HIDDEN_DIM, N, M)
        model = train_pto_mdd(model, train_samples, val_samples,
                              EPOCHS, BATCH_SIZE, LR, patience=PATIENCE)

        bt = backtest_pto_mdd(
            model, rebal_samples, N, d, C,
            n1=n1, x_min=x_min, x_max=x_max, gamma=gamma,
            delta=DELTA_VAL, is_mean=is_mean, is_std=is_std,
            stock_names=stock_names, rebal=REBAL, solve_method=SOLVER)

        fold_results_map[(LOOKBACK, n1)].extend(bt)
        log(f"   LB={LOOKBACK}, n1={n1}  windows {len(bt)}  [{time.time() - t0:.0f}s]")

    with open(ckpt_path, "wb") as f:
        pickle.dump({"fold_results_map": fold_results_map,
                     "completed_fold"  : fold_id,
                     "delta_val"       : DELTA_VAL,
                     "horizon"         : HORIZON,
                     "x_max"           : x_max,
                     "solver"          : SOLVER}, f)
    log(f"checkpoint saved (fold {fold_id}) - elapsed {time.time() - t_start:.0f}s")

n_win = sum(len(v) for v in fold_results_map.values())
log()
log(f"done - {time.time() - t_start:.0f}s total, {n_win} backtest windows")
