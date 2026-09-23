"""
run_dfl_mdd.py
--------------
Run DFL-MDD training in a single process that owns one shard along the
(lambda, LOOKBACK, n1) axes.

One process uses one core, since BLAS and torch are pinned to a single thread, so
the finer the sharding the more cores can be kept busy. Checkpoints are written per
shard, which keeps concurrent runs from interfering with each other.

Usage
------
  # (A) single run: one lambda processes all 8 configs (2 LB x 4 n1) in sequence
  python run_dfl_mdd.py --data 30 --lam 0.3

  # (B) sharded run: one (lam, LB, n1) each. Recommended on a 24-core desktop.
  python run_dfl_mdd.py --data 30 --lam 0.3 --lb 504 --n1 0.2
  -> checkpoint/dfl_mdd_30_inds_h126_LB504_n10.2_d20_l0.3_CLARABEL.pkl

  # to launch every shard at once with a cap on concurrency
  python launch_dfl_mdd.py --data 30 --horizon 126 --jobs 18

  # once the shards finish, merge them into the checkpoint the notebooks read
  python merge_ckpt.py --data 30 --horizon 126

Reproducibility
------
  --seed-mode config (the default) derives the seed from (fold, LOOKBACK, n1)
  alone, so a single run and a sharded run give identical results.
  --seed-mode fold reproduces the older behaviour, where the seed is per fold and
  the results depend on the order in which configs are processed.

  Note that existing checkpoints were produced with the fold seed, so the two modes
  do not give the same numbers. Keep a single mode within any one table.

Merged checkpoints have the same format as before and load directly.
"""

# Pin BLAS to one thread so concurrent processes do not fight over cores.
# This must happen before the numeric libraries are imported.
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import pickle
import random
import time
import warnings

# silence the CSR-tensor beta warning from inside cvxpylayers (no effect on results)
warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)


# ==========================================================
# arguments
# ==========================================================
ap = argparse.ArgumentParser()
ap.add_argument("--data",   default="30", choices=["10", "30"],
                help="industry dataset (10 or 30)")
ap.add_argument("--lam",    type=float, required=True,
                help="the lambda value this process owns")
ap.add_argument("--solver", default="CLARABEL",
                help="forward solver: CLARABEL | ECOS | SCS")
ap.add_argument("--delta",  type=float, default=20.0)
ap.add_argument("--horizon", type=int, default=126,
                help="prediction and drawdown-constraint horizon in trading days (default 126)")
ap.add_argument("--lb", type=int, nargs="+", default=None,
                help="LOOKBACK values; when given, an _LB tag is added to the checkpoint name")
ap.add_argument("--n1", type=float, nargs="+", default=None,
                help="n1 values; when given, an _n1 tag is added to the checkpoint name "
                     "(merge the shards afterwards with merge_ckpt.py)")
ap.add_argument("--xmax", type=float, default=1.0,
                help="per-asset weight cap (0 < xmax <= 1). Anything other than 1.0 adds an "
                     "_xm tag to the checkpoint name, keeping it separate from the "
                     "uncapped results")
ap.add_argument("--seed-mode", default="config", choices=["config", "fold"],
                help="config: seed fixed per (fold, LB, n1), so results do not depend on how "
                     "the work is sharded (default). fold: older behaviour, seed per "
                     "fold, results depend on config order")
args = ap.parse_args()

LAM_VAL   = args.lam
SOLVER    = args.solver
DELTA_VAL = int(args.delta) if float(args.delta).is_integer() else args.delta

HORIZON_ARG = args.horizon

TAG = (f"[h{HORIZON_ARG} LB{args.lb or 'all'} "
       f"n1{args.n1 or 'all'} lam={LAM_VAL}]")
_T0 = time.time()

def _el():
    """Elapsed time since start, as H:MM:SS."""
    s = int(time.time() - _T0)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"

def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')} +{_el()}] {TAG} {msg}", flush=True)


# ==========================================================
# data loading (identical to the notebook setup cells)
# ==========================================================
inds = pd.read_csv(f"csv/{args.data}_industry.csv")
inds["Date"] = pd.to_datetime(inds["Date"])
inds = inds.set_index("Date").sort_index()
inds = inds[~inds.index.duplicated(keep="first")] / 100.0

stock_names = inds.columns.tolist()
full_np     = inds.values
full_dates  = inds.index

gamma, x_min, x_max = 0.0, 0.0, args.xmax
N_STOCKS   = len(inds.columns)
HORIZON    = HORIZON_ARG
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
LOOKBACK_LIST = args.lb if args.lb else [252, 504]
_LBTAG        = f"_LB{'-'.join(map(str, LOOKBACK_LIST))}" if args.lb else ""
_XMTAG        = "" if args.xmax >= 1.0 else f"_xm{args.xmax:g}"
N1_LIST       = args.n1 if args.n1 else [0.1, 0.2, 0.3, 0.4]
_N1TAG        = f"_n1{'-'.join(f'{v:g}' for v in N1_LIST)}" if args.n1 else ""
configs = [{"LOOKBACK": lb, "n1": n1}
           for lb in LOOKBACK_LIST for n1 in N1_LIST]


def config_seed(fold_id, lookback, n1):
    """Seed derived only from (fold, LOOKBACK, n1), so sharding does not change it."""
    return 42 + 100_000 * fold_id + 10 * int(lookback) + int(round(n1 * 100))


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
        "val_year"      : f"{ty - VAL_YEARS}~{ty - 1}",
        "test_year"     : ty,
    })

init_train_end = date_to_idx("2013-01-01")
is_mean = full_np[:init_train_end].mean(axis=0)
is_std  = full_np[:init_train_end].std(axis=0)


class WindowSet:
    """A set of (z, r) samples.

    The samples are held as two contiguous arrays Z and R rather than as a list,
    for memory reasons. The earlier implementation kept the same data three times
    over -- a float64 list, a float64 np.array copy, then a float32 tensor --
    costing 12 bytes per element. Storing float32 once and wrapping it with
    torch.from_numpy costs 4.

    It supports len, indexing, slicing and iteration like a list, so call sites are
    unchanged. Slicing (for example [::REBAL]) returns a view and copies nothing.
    """
    __slots__ = ("Z", "R")

    def __init__(self, Z, R):
        self.Z, self.R = Z, R

    def __len__(self):
        return len(self.Z)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return WindowSet(self.Z[i], self.R[i])
        return (self.Z[i], self.R[i])

    def __iter__(self):
        for i in range(len(self.Z)):
            yield (self.Z[i], self.R[i])


def make_windows(data, lookback, horizon, start, end, dtype=np.float32):
    """
    dtype=float32 : for training and validation. The data becomes a float32 tensor
                    anyway, so the values are identical and memory drops to a third.
    dtype=float64 : for the backtest. backtest_dfl_mdd undoes the standardisation in
                    float64 to estimate Sigma, so float64 is needed to reproduce the
                    existing numbers exactly. There are only a dozen or so rebalancing
                    samples, so the memory cost is negligible.
    """
    ts = range(max(start, lookback), end - horizon + 1)
    m  = data.shape[1]
    Z  = np.empty((len(ts), lookback * m), dtype=dtype)
    R  = np.empty((len(ts), horizon, m),   dtype=dtype)
    for i, t in enumerate(ts):
        z_raw  = data[t - lookback:t]
        z_norm = (z_raw - is_mean) / (is_std + 1e-8)   # standardise in float64
        Z[i]   = z_norm.ravel()                        # cast only at the end
        R[i]   = data[t:t + horizon]
    return WindowSet(Z, R)


# ==========================================================
# training
# ==========================================================
from dfl_mdd import (PredictionModel, build_optimization_layer,
                     train_dfl_mdd, backtest_dfl_mdd)

CKPT_DIR = "./checkpoint"
os.makedirs(CKPT_DIR, exist_ok=True)
ckpt_path = os.path.join(
    CKPT_DIR,
    f"dfl_mdd_{N_STOCKS}_inds_h{HORIZON}{_XMTAG}{_LBTAG}{_N1TAG}"
    f"_d{DELTA_VAL}_l{LAM_VAL}_{SOLVER}.pkl")

log(f"data: {args.data} industries ({N_STOCKS} assets, {len(full_np)} days)")
log(f"x_max={x_max}")
log(f"HORIZON={HORIZON}, solver={SOLVER}, delta={DELTA_VAL}, "
    f"{len(configs)} configs x {N_FOLDS} folds")
log(f"checkpoint: {ckpt_path}")

if os.path.exists(ckpt_path):
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    fold_results_map = ck["fold_results_map"]
    infeas_map       = ck.get("infeas_map",
                              {(c["LOOKBACK"], c["n1"]): [] for c in configs})
    start_fold       = ck["completed_fold"] + 1
    log(f"checkpoint loaded: complete through fold {ck['completed_fold']}")
else:
    fold_results_map = {(c["LOOKBACK"], c["n1"]): [] for c in configs}
    infeas_map       = {(c["LOOKBACK"], c["n1"]): [] for c in configs}
    start_fold       = 1

t_start = time.time()

for fold_info in folds:
    fold_id = fold_info["fold"]
    if fold_id < start_fold:
        log(f"fold {fold_id} skipped")
        continue

    log(f"-- fold {fold_id} (test={fold_info['test_year']}) --")
    if args.seed_mode == "fold":
        torch.manual_seed(42); np.random.seed(42); random.seed(42)

    for cfg in configs:
        LOOKBACK, n1 = cfg["LOOKBACK"], cfg["n1"]
        t0 = time.time()

        if args.seed_mode == "config":
            _sd = config_seed(fold_id, LOOKBACK, n1)
            torch.manual_seed(_sd); np.random.seed(_sd); random.seed(_sd)

        train_samples = make_windows(full_np, LOOKBACK, HORIZON,
                                     start=LOOKBACK,
                                     end=fold_info["train_end_idx"])
        val_samples   = make_windows(full_np, LOOKBACK, HORIZON,
                                     start=fold_info["val_start_idx"],
                                     end=fold_info["val_end_idx"])[::HORIZON]
        rebal_samples = make_windows(full_np, LOOKBACK, HORIZON,
                                     start=fold_info["test_start_idx"],
                                     end=fold_info["test_end_idx"],
                                     dtype=np.float64)[::REBAL]
        # With a long HORIZON the later folds can have no rebalancing window at all
        # (e.g. H=252 with the 2025 test year: there are not 252 trading days of
        # future data left after the rebalance)
        if len(rebal_samples) == 0:
            log(f"   LB={LOOKBACK}, n1={n1}: no rebalancing windows, skipped")
            continue

        train_dates = [(str(full_dates[LOOKBACK + i])[:10],
                        str(full_dates[LOOKBACK + i + HORIZON - 1])[:10])
                       for i in range(len(train_samples))]

        pred_model = PredictionModel(LOOKBACK * N_STOCKS, HIDDEN_DIM, N, M)
        opt_layer  = build_optimization_layer(N, M, gamma, delta=DELTA_VAL)

        pred_model, _ = train_dfl_mdd(
            pred_model, opt_layer, train_samples, val_samples,
            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            n1=n1, C=C, d=d, x_min=x_min, x_max=x_max, lam=LAM_VAL,
            is_mean=is_mean, is_std=is_std, delta=DELTA_VAL,
            patience=PATIENCE, lr_patience=10, lr_factor=0.5,
            train_dates=train_dates, solve_method=SOLVER)

        bt, _, infeas = backtest_dfl_mdd(
            pred_model=pred_model, opt_layer=opt_layer,
            rebal_samples=rebal_samples, N=HORIZON, d=d, C=C,
            n1=n1, x_min=x_min, x_max=x_max, delta=DELTA_VAL,
            is_mean=is_mean, is_std=is_std,
            stock_names=stock_names, rebal=REBAL, solve_method=SOLVER)

        fold_results_map[(LOOKBACK, n1)].extend(bt)
        infeas_map[(LOOKBACK, n1)].append({"fold": fold_id, **infeas})

        log(f"   LB={LOOKBACK}, n1={n1}  "
            f"fallback {infeas['n_infeasible']}/{infeas['n_windows']} "
            f"({infeas['rate']:.1%})  "
            f"[{time.time() - t0:.0f}s]")

    with open(ckpt_path, "wb") as f:
        pickle.dump({"fold_results_map": fold_results_map,
                     "infeas_map"      : infeas_map,
                     "completed_fold"  : fold_id,
                     "delta_val"       : DELTA_VAL,
                     "lam_val"         : LAM_VAL,
                     "solver"          : SOLVER}, f)
    log(f"checkpoint saved (fold {fold_id}) -- elapsed {time.time() - t_start:.0f}s")

# ---- summary ----
n_inf = sum(e["n_infeasible"] for v in infeas_map.values() for e in v)
n_win = sum(e["n_windows"]    for v in infeas_map.values() for e in v)
n_tru = sum(e.get("n_true_infeas", 0) for v in infeas_map.values() for e in v)
log()
log(f"done -- {time.time() - t_start:.0f}s total")
if n_win:
    log(f"windows {n_win}  fallbacks {n_inf} ({n_inf / n_win:.1%})  "
        f"truly infeasible {n_tru}  numerical failures {n_inf - n_tru}")
else:
    log("no backtest windows, nothing to summarise")
