"""
save_weights.py
───────────────
각 리밸런싱 시점의 최적 포트폴리오 weight를 CSV로 저장 (Reviewer #11, #19, #22).

형식 (wide)
----------
  date_idx, date, <asset_1>, <asset_2>, ..., <asset_m>
  4528, 2018-01-02, 0.1234, 0.0000, ...

파일명
------
  {N_STOCKS}_inds_{model}[_lam{λ}][_LB{lb}][_n1{n1}].csv
  예) 10_inds_DFL-MDD_lam0.5_LB252_n10.1.csv
      10_inds_PTO-MVO_LB252.csv
      10_inds_EW.csv

사용법
------
  import importlib, save_weights
  importlib.reload(save_weights)
  from save_weights import save_all_weights

  save_all_weights(
      dfl_store=dfl_store_idx, pto_mdd=pto_mdd_idx, pto_mvo=mvo_idx,
      dfl_mvo_store=dfl_mvo_idx, bench_store=bench_store,
      full_dates=full_dates, stock_names=stock_names,
      folds=folds, HORIZON=HORIZON, REBAL=REBAL,
      N_STOCKS=N_STOCKS, save_dir="./weights")
"""

import os
import re
import numpy as np
import pandas as pd

__all__ = ["save_weights_csv", "save_all_weights", "summarize_concentration"]


# ──────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────

def _slug(label, lam=None):
    """
    label → 파일명 조각.
      'DFL-MDD (LB=252, n1=0.1)' + lam=0.5 → 'DFL-MDD_lam0.5_LB252_n10.1'
      'PTO-MVO (LB=252)'                    → 'PTO-MVO_LB252'
      'EW'                                  → 'EW'
    """
    model = label.split("(")[0].strip()
    parts = [model]
    if lam is not None:
        parts.append(f"lam{lam}")
    m_lb = re.search(r"LB=(\d+)", label)
    if m_lb:
        parts.append(f"LB{m_lb.group(1)}")
    m_n1 = re.search(r"n1=([0-9.]+)", label)
    if m_n1:
        parts.append(f"n1{m_n1.group(1)}")
    return "_".join(parts)


def _rebal_dates_from_folds(folds, LOOKBACK, HORIZON, REBAL):
    """date_idx가 없는 결과용: fold 구조에서 리밸런싱 인덱스 재구성."""
    idxs = []
    for fold in folds:
        i = fold["test_start_idx"]
        while i + HORIZON <= fold["test_end_idx"]:
            if i - LOOKBACK >= 0:
                idxs.append(i)
            i += REBAL
    return idxs


# ──────────────────────────────────────────────
# 단일 모델 저장
# ──────────────────────────────────────────────

def save_weights_csv(results, label, stock_names, save_dir,
                     N_STOCKS, full_dates=None, lam=None,
                     folds=None, HORIZON=None, REBAL=None, LOOKBACK=252,
                     verbose=True):
    """
    한 모델(config)의 weight 시계열을 wide CSV로 저장.

    Parameters
    ----------
    results     : backtest 결과 리스트 (각 dict에 'weights'; 'date_idx' 있으면 사용)
    label       : 'DFL-MDD (LB=252, n1=0.1)' 형식
    stock_names : 자산명 리스트 (열 이름)
    save_dir    : 저장 폴더
    N_STOCKS    : 파일명 구분용
    full_dates  : pd.DatetimeIndex — 있으면 실제 날짜 열 추가
    lam         : lambda 값 (DFL 계열이면 파일명에 포함)
    folds, HORIZON, REBAL, LOOKBACK
                : date_idx가 없을 때 리밸런싱 인덱스 재구성용

    Returns
    -------
    df : 저장된 DataFrame
    """
    os.makedirs(save_dir, exist_ok=True)

    # date_idx 확보
    date_idx = [r.get("date_idx") for r in results]
    if any(d is None for d in date_idx):
        if folds is not None and HORIZON is not None and REBAL is not None:
            recon = _rebal_dates_from_folds(folds, LOOKBACK, HORIZON, REBAL)
            if len(recon) == len(results):
                date_idx = recon
            else:
                if verbose:
                    print(f"  ⚠ {label}: date_idx 재구성 길이 불일치 "
                          f"({len(recon)} vs {len(results)}) — 순번으로 대체")
                date_idx = list(range(len(results)))
        else:
            date_idx = list(range(len(results)))

    W = np.vstack([np.asarray(r["weights"], dtype=float) for r in results])
    # solver 수치 잡음 정리 (long-only인데 -1e-13 같은 값이 남는 경우)
    W = np.where(np.abs(W) < 1e-8, 0.0, W)

    df = pd.DataFrame(W, columns=stock_names)
    df.insert(0, "date_idx", date_idx)

    if full_dates is not None:
        try:
            df.insert(1, "date", [str(full_dates[i])[:10] for i in date_idx])
        except Exception:
            pass

    fname = f"{N_STOCKS}_inds_{_slug(label, lam)}.csv"
    fpath = os.path.join(save_dir, fname)
    df.to_csv(fpath, index=False, float_format="%.6f")
    if verbose:
        print(f"  ✓ weights 저장: {fpath}  ({len(df)} rebalances)")
    return df


# ──────────────────────────────────────────────
# 전체 모델 일괄 저장
# ──────────────────────────────────────────────

def save_all_weights(dfl_store=None, pto_mdd=None, pto_mvo=None,
                     dfl_mvo_store=None, bench_store=None,
                     full_dates=None, stock_names=None,
                     folds=None, HORIZON=None, REBAL=None,
                     N_STOCKS="", save_dir="./weights", verbose=True):
    """
    모든 모델의 weight CSV를 한 번에 저장.

    Parameters
    ----------
    dfl_store     : dict {(delta, lam): [(results, label), ...]}   DFL-MDD
    pto_mdd       : list [(results, label), ...]
    pto_mvo       : list [(results, label), ...]
    dfl_mvo_store : dict {(delta, lam): [(results, label), ...]}   DFL-MVO
    bench_store   : dict {label: results}                          EW/GMV/hist-MVO
    나머지        : save_weights_csv 참고

    Returns
    -------
    n_saved : int
    """
    os.makedirs(save_dir, exist_ok=True)
    n = 0

    def _lb_of(label, default=252):
        m = re.search(r"LB=(\d+)", label)
        return int(m.group(1)) if m else default

    # DFL-MDD (lambda별)
    if dfl_store:
        for (delta_val, lam_val), results_list in dfl_store.items():
            for results, label in results_list:
                save_weights_csv(results, label, stock_names, save_dir,
                                 N_STOCKS, full_dates=full_dates, lam=lam_val,
                                 folds=folds, HORIZON=HORIZON, REBAL=REBAL,
                                 LOOKBACK=_lb_of(label), verbose=verbose)
                n += 1

    # DFL-MVO (lambda별)
    if dfl_mvo_store:
        for (delta_val, lam_val), results_list in dfl_mvo_store.items():
            for results, label in results_list:
                save_weights_csv(results, label, stock_names, save_dir,
                                 N_STOCKS, full_dates=full_dates, lam=lam_val,
                                 folds=folds, HORIZON=HORIZON, REBAL=REBAL,
                                 LOOKBACK=_lb_of(label), verbose=verbose)
                n += 1

    # PTO-MDD / PTO-MVO (lambda 무관)
    for lst in (pto_mdd, pto_mvo):
        if lst:
            for results, label in lst:
                save_weights_csv(results, label, stock_names, save_dir,
                                 N_STOCKS, full_dates=full_dates, lam=None,
                                 folds=folds, HORIZON=HORIZON, REBAL=REBAL,
                                 LOOKBACK=_lb_of(label), verbose=verbose)
                n += 1

    # 정적 벤치마크 (EW / GMV / hist-MVO)
    if bench_store:
        for label, results in bench_store.items():
            save_weights_csv(results, label, stock_names, save_dir,
                             N_STOCKS, full_dates=full_dates, lam=None,
                             folds=folds, HORIZON=HORIZON, REBAL=REBAL,
                             LOOKBACK=_lb_of(label), verbose=verbose)
            n += 1

    print(f"\n  ✓ 총 {n}개 weight CSV 저장 완료 → {save_dir}")
    return n


# ──────────────────────────────────────────────
# 집중도 요약 (Reviewer #22)
# ──────────────────────────────────────────────

def summarize_concentration(results, label=""):
    """
    포트폴리오 집중도 요약.

    Returns
    -------
    dict : label, HHI, EffN, MaxWeight, AvgMaxWeight, AvgNActive
      - EffN         : 1/HHI  (effective number of assets)
      - MaxWeight    : 전체 기간 최대 단일 비중
      - AvgMaxWeight : 리밸런싱별 최대비중의 평균
      - AvgNActive   : 비중 1% 초과 자산 수의 평균
    """
    W = np.vstack([np.asarray(r["weights"], dtype=float) for r in results])
    W = np.where(np.abs(W) < 1e-8, 0.0, W)   # solver 잡음 정리
    hhi_t = (W ** 2).sum(axis=1)
    hhi   = float(hhi_t.mean())
    return {
        "label"        : label,
        "HHI"          : hhi,
        "EffN"         : float(1.0 / hhi) if hhi > 0 else float("nan"),
        "MaxWeight"    : float(W.max()),
        "AvgMaxWeight" : float(W.max(axis=1).mean()),
        "AvgNActive"   : float((W > 0.01).sum(axis=1).mean()),
    }
