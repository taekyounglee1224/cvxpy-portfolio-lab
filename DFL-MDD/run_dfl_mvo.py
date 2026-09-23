"""
run_dfl_mvo.py
--------------
Run DFL-MVO training -- the ablation with the drawdown constraint removed -- in a
single process that owns one (delta, lambda) pair. Launching one process per pair
puts the cores to work in parallel.

Unlike DFL-MDD there is no n1, so LOOKBACK is the only remaining config axis.

Usage
------
  # four lambdas in parallel, for a single delta
  python run_dfl_mvo.py --data 10 --delta 20 --lam 0.3 &
  python run_dfl_mvo.py --data 10 --delta 20 --lam 0.5 &
  python run_dfl_mvo.py --data 10 --delta 20 --lam 0.7 &
  python run_dfl_mvo.py --data 10 --delta 20 --lam 1.0 &
  wait

  # the delta sweep in parallel too, when cores allow
  for D in 20 50 100; do for L in 0.3 0.5 0.7 1.0; do
      python run_dfl_mvo.py --data 10 --delta $D --lam $L > logs/mvo_d${D}_l${L}.txt 2>&1 &
  done; done

Checkpoints use the same format as the notebooks and load directly once finished.
"""

# Pin BLAS to one thread; this must happen before the numeric libraries load.
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
ap.add_argument("--data",    default="10", choices=["10", "30"])
ap.add_argument("--lam",     type=float, required=True)
ap.add_argument("--delta",   type=float, default=20.0)
ap.add_argument("--horizon", type=int,   default=126)
ap.add_argument("--solver",  default="CLARABEL")
ap.add_argument("--xmax", type=float, default=1.0,
                help="per-asset weight cap (0 < xmax <= 1). Anything other than 1.0 adds an "
                     "_xm tag to the checkpoint name, keeping it separate from the "
                     "uncapped results")
ap.add_argument("--lb", type=int, nargs="+", default=None,
                help="LOOKBACK values; when given, an _LB tag is added to the checkpoint name")
args = ap.parse_args()

LAM_VAL   = args.lam
SOLVER    = args.solver
DELTA_VAL = int(args.delta) if float(args.delta).is_integer() else args.delta

TAG = f"[MVO h{args.horizon} d{DELTA_VAL} lam={LAM_VAL}]"
_T0 = time.time()

def _el():
    """Elapsed time since start, as H:MM:SS."""
    s = int(time.time() - _T0)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"

def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')} +{_el()}] {TAG} {msg}", flush=True)


# ==========================================================
# data loading (identical to the notebooks)
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
HORIZON    = args.horizon
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
LOOKBACK_LIST = args.lb if args.lb else [252, 504]   # DFL-MVO has no n1
_LBTAG        = f"_LB{'-'.join(map(str, LOOKBACK_LIST))}" if args.lb else ""
_XMTAG        = "" if args.xmax >= 1.0 else f"_xm{args.xmax:g}"


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
    """A set of (z, r) samples; identical to run_dfl_mdd.WindowSet.

    The samples are held as two contiguous arrays Z and R rather than as a list.
    The earlier implementation kept the same data three times over -- a float64
    list, a float64 np.array copy, then a float32 tensor -- costing 12 bytes per
    element. Storing float32 once and wrapping it with torch.from_numpy costs 4.

    It supports len, indexing, slicing and iteration like a list, so call sites are
    unchanged.
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
    dtype=float64 : for the backtest. backtest_dfl_mvo undoes the standardisation in
                    float64 to estimate Sigma, so float64 is needed to reproduce the
                    existing numbers exactly.
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
from dfl_mdd import PredictionModel
from dfl_mvo import build_mvo_layer, train_dfl_mvo, backtest_dfl_mvo

CKPT_DIR = "./checkpoint"
os.makedirs(CKPT_DIR, exist_ok=True)
ckpt_path = os.path.join(
    CKPT_DIR,
    f"dfl_mvo_{N_STOCKS}_inds_h{HORIZON}{_XMTAG}{_LBTAG}"
    f"_d{DELTA_VAL}_l{LAM_VAL}_{SOLVER}.pkl")

log(f"data: {args.data} industries ({N_STOCKS} assets, {len(full_np)} days)")
log(f"HORIZON={HORIZON}, solver={SOLVER}, delta={DELTA_VAL}, "
    f"LOOKBACK {LOOKBACK_LIST} x {N_FOLDS} folds")
log(f"checkpoint: {ckpt_path}")

if os.path.exists(ckpt_path):
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    fold_results_map = ck["fold_results_map"]
    infeas_map       = ck.get("infeas_map", {lb: [] for lb in LOOKBACK_LIST})
    start_fold       = ck["completed_fold"] + 1
    log(f"checkpoint loaded: complete through fold {ck['completed_fold']}")
else:
    fold_results_map = {lb: [] for lb in LOOKBACK_LIST}
    infeas_map       = {lb: [] for lb in LOOKBACK_LIST}
    start_fold       = 1

t_start = time.time()

for fold_info in folds:
    fold_id = fold_info["fold"]
    if fold_id < start_fold:
        log(f"fold {fold_id} skipped")
        continue

    log(f"-- fold {fold_id} (test={fold_info['test_year']}) --")
    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    for LOOKBACK in LOOKBACK_LIST:
        t0 = time.time()

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
        if len(rebal_samples) == 0:
            log(f"   LB={LOOKBACK}: no rebalancing windows, skipped")
            continue

        train_dates = [(str(full_dates[LOOKBACK + i])[:10],
                        str(full_dates[LOOKBACK + i + HORIZON - 1])[:10])
                       for i in range(len(train_samples))]

        pred_model = PredictionModel(LOOKBACK * N_STOCKS, HIDDEN_DIM, N, M)
        opt_layer  = build_mvo_layer(N, M, gamma, delta=DELTA_VAL)

        pred_model, _ = train_dfl_mvo(
            pred_model, opt_layer, train_samples, val_samples,
            epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
            C=C, d=d, x_min=x_min, x_max=x_max, lam=LAM_VAL,
            is_mean=is_mean, is_std=is_std, delta=DELTA_VAL,
            patience=PATIENCE, lr_patience=10, lr_factor=0.5,
            train_dates=train_dates, solve_method=SOLVER)

        bt, _, infeas = backtest_dfl_mvo(
            pred_model=pred_model, opt_layer=opt_layer,
            rebal_samples=rebal_samples, N=HORIZON, d=d, C=C,
            x_min=x_min, x_max=x_max, delta=DELTA_VAL,
            is_mean=is_mean, is_std=is_std,
            stock_names=stock_names, rebal=REBAL, solve_method=SOLVER)

        fold_results_map[LOOKBACK].extend(bt)
        infeas_map[LOOKBACK].append({"fold": fold_id, **infeas})

        log(f"   LB={LOOKBACK}  "
            f"fallback {infeas['n_infeasible']}/{infeas['n_windows']} "
            f"({infeas['rate']:.1%})  [{time.time() - t0:.0f}s]")

    with open(ckpt_path, "wb") as f:
        pickle.dump({"fold_results_map": fold_results_map,
                     "infeas_map"      : infeas_map,
                     "completed_fold"  : fold_id,
                     "delta_val"       : DELTA_VAL,
                     "lam_val"         : LAM_VAL,
                     "horizon"         : HORIZON,
                     "solver"          : SOLVER}, f)
    log(f"checkpoint saved (fold {fold_id}) -- elapsed {time.time() - t_start:.0f}s")

# ---- summary ----
n_inf = sum(e["n_infeasible"] for v in infeas_map.values() for e in v)
n_win = sum(e["n_windows"]    for v in infeas_map.values() for e in v)
log()
log(f"done -- {time.time() - t_start:.0f}s total")
if n_win:
    log(f"windows {n_win}  fallbacks {n_inf} ({n_inf / n_win:.1%})")
else:
    log("no backtest windows, nothing to summarise")
