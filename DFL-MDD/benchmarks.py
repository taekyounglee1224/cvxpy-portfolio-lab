"""
benchmarks.py
─────────────
재학습 불필요한 벤치마크 & 진단 유틸 (Reviewer #8, #21, #2, #3 대응).

제공 항목
--------
1. 벤치마크 백테스트 (DFL-MDD와 동일한 리밸런싱 시점):
     - EW       : equally-weighted
     - GMV      : global minimum variance (long-only)
     - hist-MVO : sample mean-variance (long-only)
   → DFL 백테스트와 같은 결과 dict 구조 반환
     (window, weights, w_real, R_real, M_real, MDD_abs, Sharpe)

2. drift 반영 turnover (Reviewer #21):
     리밸런싱 직전 실현수익으로 표류한 weight 기준 one-way L1.

3. uncompounded absolute drawdown (Reviewer #1, #2):
     가법(누적합) 경로 기준 절대 낙폭. compounded(상대) 낙폭과 병행 제공.

4. solver infeasibility rate (Reviewer #4) — GMV/MVO 최적화 실패율.

가정
----
  full_np  : (T, m) 일별 단순수익률 (decimal)
  folds    : dict 리스트, 각 원소에 test_start_idx, test_end_idx
  리밸런싱 : i = test_start_idx + k*REBAL,  단 i + HORIZON <= test_end_idx
  lookback : full_np[i-LOOKBACK : i]  (i 직전 LOOKBACK일)
  보유      : i ~ i+REBAL (첫 REBAL일만 실현)

사용법
------
  import importlib, benchmarks
  importlib.reload(benchmarks)
  from benchmarks import backtest_benchmark, w_equal, w_gmv, w_mvo, compute_turnover
"""

import numpy as np
import cvxpy as cp

__all__ = [
    "rebalance_dates",
    "attach_date_idx",
    "w_equal", "w_gmv", "w_mvo",
    "backtest_benchmark",
    "compute_turnover",
    "infeasibility_rate",
]


# ──────────────────────────────────────────────
# 리밸런싱 시점 (DFL 백테스트와 동일)
# ──────────────────────────────────────────────

def rebalance_dates(fold, LOOKBACK, HORIZON, REBAL):
    """fold 하나에서 리밸런싱 인덱스 i 리스트 반환."""
    idxs = []
    i = fold["test_start_idx"]
    while i + HORIZON <= fold["test_end_idx"]:
        if i - LOOKBACK >= 0:
            idxs.append(i)
        i += REBAL
    return idxs


def attach_date_idx(results, folds, LOOKBACK, HORIZON, REBAL):
    """
    date_idx 없는 결과 리스트(DFL/PTO checkpoint)에 리밸런싱 인덱스 부착.
    결과는 fold별·윈도우 순서로 저장돼 있다고 가정 (백테스트 생성 순서).

    Returns
    -------
    새 리스트 (각 dict에 'date_idx' 추가)
    """
    dates = []
    for fold in folds:
        dates += rebalance_dates(fold, LOOKBACK, HORIZON, REBAL)
    if len(dates) != len(results):
        print(f"  ⚠ attach_date_idx: 길이 불일치 (dates={len(dates)}, "
              f"results={len(results)}) — 순서/구조 확인 필요")
    return [{**r, "date_idx": d} for r, d in zip(results, dates)]


# ──────────────────────────────────────────────
# 가중치 함수  weight_fn(R_lb) -> w   (R_lb: (LOOKBACK, m))
# ──────────────────────────────────────────────

def w_equal(R_lb, **kwargs):
    m = R_lb.shape[1]
    return np.ones(m) / m, True   # (weights, feasible)


def _solve_long_only(objective, x, m):
    """long-only + full-investment QP 공통 solver. (w, feasible) 반환."""
    constraints = [cp.sum(x) == 1, x >= 0]
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
        if x.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            raise ValueError(prob.status)
        w = np.clip(np.array(x.value).flatten(), 0, None)
        s = w.sum()
        return (w / s if s > 0 else np.ones(m) / m), True
    except Exception:
        return np.ones(m) / m, False   # fallback = EW, infeasible 플래그


def w_gmv(R_lb, ridge=1e-4, **kwargs):
    """Global Minimum Variance (long-only): min x'Σx."""
    m = R_lb.shape[1]
    Sigma = np.cov(R_lb.T) + ridge * np.eye(m)
    x = cp.Variable(m)
    obj = cp.Minimize(cp.quad_form(x, cp.psd_wrap(Sigma)))
    return _solve_long_only(obj, x, m)


def w_mvo(R_lb, delta=20.0, ridge=1e-4, **kwargs):
    """
    Historical Mean-Variance (long-only): max mu'x - (delta/2) x'Σx.

    표준 MVO — mu와 Σ를 동일 horizon(일별 lookback)으로 사용해 내부 일관성 유지.
      mu    = 일별 sample mean
      Σ     = 일별 lookback 공분산 (+ ridge)
    (mu·Σ를 같은 배수로 스케일해도 argmax 불변이므로 일별-일별이 표준.
     mu만 horizon 배수로 키우면 return이 risk를 과대 지배해 error-maximization
     (극단 집중)이 발생하므로 사용하지 않음.)
    """
    m = R_lb.shape[1]
    mu    = R_lb.mean(axis=0)
    Sigma = np.cov(R_lb.T) + ridge * np.eye(m)
    x = cp.Variable(m)
    obj = cp.Maximize(mu @ x - (delta / 2.0) * cp.quad_form(x, cp.psd_wrap(Sigma)))
    return _solve_long_only(obj, x, m)


# ──────────────────────────────────────────────
# 벤치마크 백테스트 (DFL 결과 구조와 동일)
# ──────────────────────────────────────────────

def backtest_benchmark(full_np, folds, weight_fn, LOOKBACK, HORIZON, REBAL,
                       stock_names=None, weight_kwargs=None):
    """
    Returns
    -------
    results : list of dict
        각 dict: window, date_idx, weights, w_real, R_real,
                 M_real(=DFL식 상대낙폭), MDD_abs(uncompounded 절대낙폭), Sharpe
    n_infeasible : int   (최적화 실패 횟수)
    """
    weight_kwargs = weight_kwargs or {}
    m = full_np.shape[1]
    names = stock_names if stock_names else [f"S{j+1}" for j in range(m)]
    results = []
    n_infeasible = 0
    win = 0

    for fold in folds:
        for i in rebalance_dates(fold, LOOKBACK, HORIZON, REBAL):
            win += 1
            R_lb = full_np[i - LOOKBACK : i]              # (LOOKBACK, m)
            w, feasible = weight_fn(R_lb, **weight_kwargs)
            if not feasible:
                n_infeasible += 1

            # 실현: 첫 REBAL일만 보유
            R_hold = full_np[i : i + REBAL]              # (REBAL, m)
            p_daily = R_hold @ w                          # (REBAL,) 일별 포트폴리오 수익
            w_real  = np.cumsum(p_daily)                  # uncompounded 누적 경로

            R_real = w_real[-1]

            # DFL식 상대 낙폭 (기존 비교용)
            pv_w   = 1.0 + w_real
            rmax_w = np.maximum.accumulate(pv_w)
            M_real = float(np.max((rmax_w - pv_w) / (rmax_w + 1e-10)))

            # uncompounded 절대 낙폭 (Reviewer #1, #2)
            cum      = np.concatenate([[0.0], w_real])
            run_max  = np.maximum.accumulate(cum)
            MDD_abs  = float(np.max(run_max - cum))

            # Sharpe (일별 실현 기준)
            sig = p_daily.std()
            sharpe = float(p_daily.mean() / (sig + 1e-12))

            results.append({
                "window":   win,
                "date_idx": i,
                "weights":  w.astype(np.float32),
                "w_real":   w_real.astype(np.float32),
                "R_real":   float(R_real),
                "M_real":   M_real,
                "MDD_abs":  MDD_abs,
                "Sharpe":   sharpe,
            })

    return results, n_infeasible


# ──────────────────────────────────────────────
# drift 반영 turnover (Reviewer #21)
# ──────────────────────────────────────────────

def compute_turnover(results, full_np, REBAL, one_way=True):
    """
    리밸런싱 직전 실현수익으로 표류한 weight 기준 turnover.

    각 리밸런싱 k (>=2):
      gross_j   = Π_{d=0}^{REBAL-1} (1 + r_{i_{k-1}+d, j})     자산별 보유기간 총수익
      w_drift_j = w_{k-1,j} gross_j / Σ_l w_{k-1,l} gross_l     표류 weight
      turnover_k = Σ_j |w_{k,j} - w_drift_j|                    one-way L1

    Returns
    -------
    dict : { "turnovers": np.ndarray, "mean": float, "median": float }
    """
    turnovers = []
    for k in range(1, len(results)):
        prev = results[k - 1]
        i_prev = prev["date_idx"]
        w_prev = np.array(prev["weights"], dtype=float)
        w_curr = np.array(results[k]["weights"], dtype=float)

        R_hold = full_np[i_prev : i_prev + REBAL]         # (REBAL, m)
        gross  = np.prod(1.0 + R_hold, axis=0)            # (m,) 자산별 총수익
        drifted = w_prev * gross
        s = drifted.sum()
        w_drift = drifted / s if s > 0 else w_prev

        to = np.sum(np.abs(w_curr - w_drift))
        turnovers.append(to if one_way else 0.5 * to)

    turnovers = np.array(turnovers)
    return {
        "turnovers": turnovers,
        "mean":   float(turnovers.mean())   if len(turnovers) else float("nan"),
        "median": float(np.median(turnovers)) if len(turnovers) else float("nan"),
    }


# ──────────────────────────────────────────────
# infeasibility rate (Reviewer #4)
# ──────────────────────────────────────────────

def infeasibility_rate(n_infeasible, results):
    n = len(results)
    return {
        "n_infeasible": n_infeasible,
        "n_total": n,
        "rate": (n_infeasible / n) if n else float("nan"),
    }
