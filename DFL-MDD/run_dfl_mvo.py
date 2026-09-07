"""
run_dfl_mvo.py
──────────────
DFL-MVO(drawdown 제약 제거 ablation) 학습을 (delta, lambda) 하나만 담당하는
단일 프로세스로 실행한다. 조합별로 프로세스를 띄우면 코어를 병렬로 쓸 수 있다.

DFL-MDD와 달리 n1이 없으므로 config는 LOOKBACK 뿐이다.

사용법
------
  # lambda 4개 병렬 (delta 하나)
  python run_dfl_mvo.py --data 10 --delta 20 --lam 0.3 &
  python run_dfl_mvo.py --data 10 --delta 20 --lam 0.5 &
  python run_dfl_mvo.py --data 10 --delta 20 --lam 0.7 &
  python run_dfl_mvo.py --data 10 --delta 20 --lam 1.0 &
  wait

  # delta sweep까지 병렬로 (코어가 넉넉할 때)
  for D in 20 50 100; do for L in 0.3 0.5 0.7 1.0; do
      python run_dfl_mvo.py --data 10 --delta $D --lam $L > logs/mvo_d${D}_l${L}.txt 2>&1 &
  done; done

체크포인트는 노트북과 동일한 형식이므로 끝난 뒤 그대로 로드하면 된다.
"""

# ── BLAS 스레드 고정 (import 전에 설정) ──
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import pickle
import random
import time
import warnings

# cvxpylayers 내부 CSR 텐서 beta 경고 억제 (동작·결과에는 영향 없음)
warnings.filterwarnings("ignore", message=".*Sparse CSR tensor support.*")

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)


# ══════════════════════════════════════════════════════════
# 인자
# ══════════════════════════════════════════════════════════
ap = argparse.ArgumentParser()
ap.add_argument("--data",    default="10", choices=["10", "30"])
ap.add_argument("--lam",     type=float, required=True)
ap.add_argument("--delta",   type=float, default=20.0)
ap.add_argument("--horizon", type=int,   default=126)
ap.add_argument("--solver",  default="CLARABEL")
ap.add_argument("--lb", type=int, nargs="+", default=None,
                help="LOOKBACK 목록. 지정 시 체크포인트 이름에 _LB 태그가 붙는다")
args = ap.parse_args()

LAM_VAL   = args.lam
SOLVER    = args.solver
DELTA_VAL = int(args.delta) if float(args.delta).is_integer() else args.delta

TAG = f"[MVO h{args.horizon} d{DELTA_VAL} λ={LAM_VAL}]"
_T0 = time.time()

def _el():
    """시작 이후 경과시간 H:MM:SS."""
    s = int(time.time() - _T0)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"

def log(msg=""):
    print(f"[{time.strftime('%H:%M:%S')} +{_el()}] {TAG} {msg}", flush=True)


# ══════════════════════════════════════════════════════════
# 데이터 (노트북과 동일)
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
LOOKBACK_LIST = args.lb if args.lb else [252, 504]   # ★ DFL-MVO는 n1 없음
_LBTAG        = f"_LB{'-'.join(map(str, LOOKBACK_LIST))}" if args.lb else ""


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


def make_windows(data, lookback, horizon, start, end):
    samples = []
    for t in range(max(start, lookback), end - horizon + 1):
        z_raw  = data[t - lookback:t]
        z_norm = (z_raw - is_mean) / (is_std + 1e-8)
        samples.append((z_norm.flatten(), data[t:t + horizon]))
    return samples


# ══════════════════════════════════════════════════════════
# 학습
# ══════════════════════════════════════════════════════════
from dfl_mdd import PredictionModel
from dfl_mvo import build_mvo_layer, train_dfl_mvo, backtest_dfl_mvo

CKPT_DIR = "./checkpoint"
os.makedirs(CKPT_DIR, exist_ok=True)
ckpt_path = os.path.join(
    CKPT_DIR,
    f"dfl_mvo_{N_STOCKS}_inds_h{HORIZON}{_LBTAG}_d{DELTA_VAL}_l{LAM_VAL}_{SOLVER}.pkl")

log(f"데이터 {args.data} industries ({N_STOCKS}개 자산, {len(full_np)}일)")
log(f"HORIZON={HORIZON}, solver={SOLVER}, delta={DELTA_VAL}, "
    f"LOOKBACK {LOOKBACK_LIST} × fold {N_FOLDS}개")
log(f"체크포인트: {ckpt_path}")

if os.path.exists(ckpt_path):
    with open(ckpt_path, "rb") as f:
        ck = pickle.load(f)
    fold_results_map = ck["fold_results_map"]
    infeas_map       = ck.get("infeas_map", {lb: [] for lb in LOOKBACK_LIST})
    start_fold       = ck["completed_fold"] + 1
    log(f"체크포인트 로드: fold {ck['completed_fold']}까지 완료")
else:
    fold_results_map = {lb: [] for lb in LOOKBACK_LIST}
    infeas_map       = {lb: [] for lb in LOOKBACK_LIST}
    start_fold       = 1

t_start = time.time()

for fold_info in folds:
    fold_id = fold_info["fold"]
    if fold_id < start_fold:
        log(f"fold {fold_id} 스킵")
        continue

    log(f"── fold {fold_id} (test={fold_info['test_year']}) ──")
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
                                     end=fold_info["test_end_idx"])[::REBAL]

        # HORIZON이 크면 뒤쪽 fold에 리밸런싱 윈도우가 없을 수 있음
        if len(rebal_samples) == 0:
            log(f"   LB={LOOKBACK}  리밸런싱 윈도우 0개 — 건너뜀")
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
    log(f"체크포인트 저장 (fold {fold_id}) — 누적 {time.time() - t_start:.0f}s")

# ── 요약 ──
n_inf = sum(e["n_infeasible"] for v in infeas_map.values() for e in v)
n_win = sum(e["n_windows"]    for v in infeas_map.values() for e in v)
log()
log(f"완료 — 총 {time.time() - t_start:.0f}초")
if n_win:
    log(f"윈도우 {n_win}  fallback {n_inf} ({n_inf / n_win:.1%})")
else:
    log("백테스트 윈도우 0개 — 요약 없음")
