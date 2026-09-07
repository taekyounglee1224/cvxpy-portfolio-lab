"""
carryforward.py
───────────────
infeasible 시점의 포트폴리오 비중을 '직전 리밸런싱 시점의 비중'으로 대체하고
성과 지표를 재계산한다 (no-trade 규칙). 재학습 불필요 — 후처리만 수행.

배경
----
최적화가 실패한 시점에서 dfl_mdd.py는 예측값 기반 softmax 배분을 대체 규칙으로
사용한다. 이는 drawdown 제약을 만족한다는 보장이 없고 실제 운용 규칙도 아니다.
대신 "풀 수 없으면 거래하지 않는다"는 no-trade 규칙을 적용하면
 - 해당 시점의 turnover가 0이 되고
 - 직전에 제약을 만족했던 포트폴리오를 유지하게 된다.

실패 시점은 backtest_dfl_mdd가 infeas_summary["failed_windows"]에 기록해 둔
윈도우 인덱스(fold 내 0-based)를 사용한다.

사용법
------
  import importlib, carryforward
  importlib.reload(carryforward)
  from carryforward import apply_carryforward

  dfl_store_cf = apply_carryforward(
      dfl_results_store, infeas_store,
      folds=folds, full_np=full_np,
      HORIZON=HORIZON, REBAL=REBAL, d=d, C=C)
"""

import re
import numpy as np

from benchmarks import attach_date_idx

__all__ = ["carryforward_fallback", "apply_carryforward",
           "parse_lb", "parse_n1"]


# ──────────────────────────────────────────────
# 라벨 파서
# ──────────────────────────────────────────────

def parse_lb(label, default=252):
    """'DFL-MDD (LB=252, n1=0.1)' → 252"""
    m = re.search(r"LB=(\d+)", label)
    return int(m.group(1)) if m else default


def parse_n1(label, default=None):
    """'DFL-MDD (LB=252, n1=0.1)' → 0.1"""
    m = re.search(r"n1=([0-9.]+)", label)
    return float(m.group(1).rstrip(".")) if m else default


# ──────────────────────────────────────────────
# 핵심: 단일 config
# ──────────────────────────────────────────────

def carryforward_fallback(results, infeas_logs, full_np, REBAL, LOOKBACK,
                          d=1.0, C=1.0, ridge=1e-4, verbose=True):
    """
    Parameters
    ----------
    results     : fold들을 이어붙인 백테스트 결과. 각 원소에 'date_idx' 필요
                  (attach_date_idx로 부착)
    infeas_logs : infeas_map[(LB, n1)] — fold별 dict 리스트.
                  각 dict에 'fold', 'n_windows', 'failed_windows' 포함
    full_np     : (T, m) 일별 수익률 원본 (정규화 이전)
    REBAL       : 보유 기간(거래일)
    LOOKBACK    : 공분산 추정 구간 — Sharpe 계산에 사용
    d, C        : 수익률 정규화 상수 (backtest와 동일 값)
    ridge       : 공분산 ridge (backtest와 동일: 1e-4)

    Returns
    -------
    list : results와 같은 형식. weights / w_real / R_real / M_real / Sharpe 갱신.

    Notes
    -----
    - 실패가 연속되면 마지막으로 성공한 시점의 비중이 계속 유지된다
      (w_prev는 feasible일 때만 갱신).
    - 첫 윈도우가 실패하면 직전이 없으므로 동일가중(EW)으로 시작한다.
    - fold 경계를 넘어 비중을 이어받는다 (포트폴리오는 시간상 연속).
    - 지표 정의는 backtest_dfl_mdd와 동일하게 유지했다.
    """
    # fold 내 인덱스 → 전체 인덱스
    failed, offset = set(), 0
    for e in sorted(infeas_logs, key=lambda x: x["fold"]):
        failed |= {offset + w for w in e.get("failed_windows", [])}
        offset += e["n_windows"]

    m      = full_np.shape[1]
    out    = []
    w_prev = np.full(m, 1.0 / m)      # 첫 윈도우 실패 시 EW
    n_sub  = 0

    for i, r in enumerate(results):
        t = r["date_idx"]

        if i in failed:
            w = w_prev.copy()                       # no-trade
            n_sub += 1
        else:
            w = np.asarray(r["weights"], dtype=float)
            w_prev = w.copy()                       # feasible 해만 기억

        # ── 실현 성과 재계산 ──
        p_ret  = full_np[t:t + REBAL] @ w           # 일별 포트폴리오 수익
        w_real = np.cumsum(p_ret)                   # 가산 누적 경로
        pv     = 1.0 + w_real
        rmax   = np.maximum.accumulate(pv)

        S   = np.cov(full_np[t - LOOKBACK:t].T) + ridge * np.eye(m)
        sig = np.sqrt(max(float(w @ S @ w), 0.0) + 1e-8)

        out.append({**r,
                    "weights": w,
                    "w_real" : w_real,
                    "R_real" : float(w_real[-1] / (d * C)),
                    "M_real" : float(np.max((rmax - pv) / (rmax + 1e-10))),
                    "Sharpe" : float(p_ret.mean() / sig)})

    if verbose:
        rate = n_sub / len(results) if results else float("nan")
        print(f"    대체 {n_sub}/{len(results)} ({rate:.1%})")
    return out


# ──────────────────────────────────────────────
# 전 config 일괄 적용
# ──────────────────────────────────────────────

def apply_carryforward(results_store, infeas_store, folds, full_np,
                       HORIZON, REBAL, d=1.0, C=1.0, ridge=1e-4,
                       verbose=True):
    """
    Parameters
    ----------
    results_store : {(delta, lam): [(results, label), ...]}
    infeas_store  : {(delta, lam): {(LB, n1): [fold별 dict, ...]}}
    folds         : fold 정의 리스트 (date_idx 재구성용)

    Returns
    -------
    dict : results_store와 같은 구조. carry-forward가 적용된 결과.
    """
    cf_store = {}
    for key, lst in results_store.items():
        infeas_map = infeas_store[key]
        out = []
        for res, label in lst:
            lb, n1 = parse_lb(label), parse_n1(label)
            logs   = infeas_map.get((lb, n1), infeas_map.get(lb, []))
            if verbose:
                print(f"  ▸ [delta={key[0]}, lam={key[1]}] {label}")
            res_idx = attach_date_idx(res, folds, lb, HORIZON, REBAL)
            out.append((carryforward_fallback(
                res_idx, logs, full_np, REBAL, lb,
                d=d, C=C, ridge=ridge, verbose=verbose), label))
        cf_store[key] = out

    if verbose:
        print(f"\n✓ carry-forward 적용 완료 — {len(cf_store)}개 (delta, lam)")
    return cf_store
