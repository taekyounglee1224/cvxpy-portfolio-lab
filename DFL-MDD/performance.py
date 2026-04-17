"""
performance.py
──────────────
백테스트 결과(results 리스트)로부터 포트폴리오 성과 지표를 계산·출력하는 모듈.

지표
----
  Ann.Ret   : 연환산 수익률
  Sharpe    : 연환산 샤프 지수  (rf=0 기준)
  CVaR(5%)  : 5% 수준의 Conditional VaR  (일별 수익률 기준, 손실 크기로 표시)
  MDD       : Maximum Drawdown  (전체 백테스트 기간 기준)
  HHI       : 평균 Herfindahl-Hirschman Index  (포트폴리오 집중도)

사용법
------
  from performance import compute_performance, print_performance_table

  # all_results : [(results_list, label), ...]
  print_performance_table(all_results)
"""

import numpy as np
import pandas as pd

__all__ = [
    "build_equity_curve",
    "compute_performance",
    "print_performance_table",
]


# ──────────────────────────────────────────────
# 내부 유틸
# ──────────────────────────────────────────────

def build_equity_curve(results):
    """
    results : backtest_* 함수가 반환한 dict 리스트
              각 dict에 'w_real' (ndarray, shape=(rebal,)) 키가 있어야 함.

    Returns
    -------
    equity : np.ndarray, shape=(T+1,)  — 시작값 1.0 기준 누적 포트폴리오 가치
    """
    cum_pv = [1.0]
    for res in results:
        base = cum_pv[-1]
        cum_pv.extend((base * (1.0 + res["w_real"])).tolist())
    return np.array(cum_pv)


def _annualized_return(equity):
    n_days = len(equity) - 1
    if n_days <= 0:
        return float("nan")
    return float((equity[-1] / equity[0]) ** (252.0 / n_days) - 1.0)


def _sharpe(equity, rf=0.0):
    rets = np.diff(equity) / (equity[:-1] + 1e-10)
    excess = rets - rf / 252.0
    std = excess.std()
    if std < 1e-12:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(252.0))


def _cvar(equity, alpha=0.05):
    """
    CVaR (Expected Shortfall) at alpha level.
    일별 수익률의 하위 alpha 분위 이하 평균값.
    손실 크기(양수)로 반환.
    """
    rets = np.diff(equity) / (equity[:-1] + 1e-10)
    cutoff = np.quantile(rets, alpha)
    tail = rets[rets <= cutoff]
    if len(tail) == 0:
        return float("nan")
    return float(-tail.mean())   # 손실 크기이므로 부호 반전


def _mdd(equity):
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / (peak + 1e-10)
    return float(np.max(drawdown))


def _hhi(results):
    """
    각 리밸런싱 윈도우의 HHI = sum(w_i^2) 를 평균.
    완전 분산 시 1/m, 완전 집중 시 1.
    """
    values = []
    for res in results:
        w = np.array(res["weights"])
        values.append(float(np.sum(w ** 2)))
    return float(np.mean(values)) if values else float("nan")


# ──────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────

def compute_performance(results, label=""):
    """
    results : backtest_* 반환 리스트
    label   : 전략 이름 (출력용)

    Returns
    -------
    dict : { "label", "Ann.Ret", "Sharpe", "CVaR(5%)", "MDD", "HHI" }
    """
    equity = build_equity_curve(results)
    return {
        "label"    : label,
        "Ann.Ret"  : _annualized_return(equity),
        "Sharpe"   : _sharpe(equity),
        "CVaR(5%)" : _cvar(equity),
        "MDD"      : _mdd(equity),
        "HHI"      : _hhi(results),
    }


def print_performance_table(all_results, title=None):
    """
    Parameters
    ----------
    all_results : list of (results_list, label) tuples
    title       : 출력 상단에 표시할 제목 (optional)

    Returns
    -------
    df : pd.DataFrame  (포맷 적용 전 수치값)
    """
    rows = []
    for results, label in all_results:
        rows.append(compute_performance(results, label))

    df_raw = pd.DataFrame(rows).set_index("label")

    # 포맷 적용 (표시용 복사본)
    df_fmt = df_raw.copy()
    df_fmt["Ann.Ret"]   = df_raw["Ann.Ret"].map("{:+.2%}".format)
    df_fmt["Sharpe"]    = df_raw["Sharpe"].map("{:.3f}".format)
    df_fmt["CVaR(5%)"]  = df_raw["CVaR(5%)"].map("{:.2%}".format)
    df_fmt["MDD"]       = df_raw["MDD"].map("{:.2%}".format)
    df_fmt["HHI"]       = df_raw["HHI"].map("{:.4f}".format)

    if title:
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")
    print(df_fmt.to_string())
    print()

    return df_raw   # 수치값 반환 (추가 분석용)
