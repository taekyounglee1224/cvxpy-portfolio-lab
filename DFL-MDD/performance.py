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

import os
import numpy as np
import pandas as pd

__all__ = [
    "build_equity_curve",
    "apply_tc",
    "compute_performance",
    "print_performance_table",
    "print_tc_performance_table",
    "build_metrics_dataframe",
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


def _calmar(equity):
    cum_ret = float(equity[-1] / equity[0] - 1.0)
    mdd     = _mdd(equity)
    return float(cum_ret / (mdd + 1e-10))


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
# Transaction Cost 유틸
# ──────────────────────────────────────────────

def apply_tc(results, tc_rate=0.0):
    """
    Transaction cost 사후(post-hoc) 반영.

    리밸런싱 시점마다 이전 weights 대비 turnover를 계산하고
    tc_rate × turnover 만큼 포트폴리오 가치를 차감.

    Parameters
    ----------
    results : backtest_* 반환 리스트  (각 dict에 'weights', 'w_real' 필요)
    tc_rate : float  (예: 0.001 = 0.1%,  0.10 = 10%)

    Returns
    -------
    adjusted : 동일 구조의 리스트  (w_real만 TC 반영하여 교체)
    """
    if tc_rate == 0.0:
        return results

    adjusted = []
    prev_w   = None

    for res in results:
        w      = np.array(res["weights"])
        w_real = np.array(res["w_real"], dtype=float)

        if prev_w is not None:
            turnover = float(np.sum(np.abs(w - prev_w)))
            tc       = tc_rate * turnover
            # base를 (1 - tc) 배로 줄인 효과:
            # (base*(1-tc)) * (1 + w_real) = base * (1 + (1-tc)*(1+w_real) - 1)
            w_real = (1.0 - tc) * (1.0 + w_real) - 1.0

        adjusted.append({**res, "w_real": w_real})
        prev_w = w

    return adjusted


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
        "Calmar"   : _calmar(equity),
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


def print_tc_performance_table(all_results,
                                tc_rates=(0.0, 0.10, 0.20, 0.40),
                                title=None):
    """
    TC rate별 성과 비교표를 한 번에 출력.

    Parameters
    ----------
    all_results : list of (results_list, label) tuples
    tc_rates    : iterable of float  (예: (0.0, 0.10, 0.20, 0.40))
    title       : 출력 상단 제목 (optional)

    Returns
    -------
    dict : { tc_rate: pd.DataFrame }
    """
    if title:
        print(f"\n{'═'*70}")
        print(f"  {title}  —  Transaction Cost 민감도 분석")
        print(f"{'═'*70}")

    dfs = {}
    for tc_rate in tc_rates:
        tc_label = f"TC={int(round(tc_rate*10000))}bps"
        rows = []
        for results, label in all_results:
            adj = apply_tc(results, tc_rate)
            rows.append(compute_performance(adj, label))

        df_raw = pd.DataFrame(rows).set_index("label")
        df_fmt = df_raw.copy()
        df_fmt["Ann.Ret"]   = df_raw["Ann.Ret"].map("{:+.2%}".format)
        df_fmt["Sharpe"]    = df_raw["Sharpe"].map("{:.3f}".format)
        df_fmt["CVaR(5%)"]  = df_raw["CVaR(5%)"].map("{:.2%}".format)
        df_fmt["MDD"]       = df_raw["MDD"].map("{:.2%}".format)
        df_fmt["HHI"]       = df_raw["HHI"].map("{:.4f}".format)

        print(f"\n  ── {tc_label} ──")
        print(df_fmt.to_string())

        dfs[tc_rate] = df_raw

    print()
    return dfs


def build_metrics_dataframe(dfl_results_store,
                             all_results_pto_mdd,
                             all_results_mvo,
                             tc_rate=0.0,
                             save_dir=None,
                             N_STOCKS=""):
    """
    lambda별로 DataFrame을 만들고 CSV 저장.
    각 CSV = 해당 lambda의 DFL-MDD 결과 + 전체 벤치마크 (PTO-MDD, PTO-MVO).

    Parameters
    ----------
    dfl_results_store   : dict  {(delta, lam): [(results, label), ...]}
    all_results_pto_mdd : list of (results, label)
    all_results_mvo     : list of (results, label)
    tc_rate             : float  transaction cost rate (default 0.0)
    save_dir            : str or None  저장 폴더 (예: "./csv")
    N_STOCKS            : int or str  파일명 구분용 (예: 10, 30)

    Returns
    -------
    dfs : dict  { lam_val: pd.DataFrame }
    """

    # ── 벤치마크 rows 먼저 계산 (모든 lambda CSV에 공통 포함) ──
    benchmark_rows = []

    for results, label in all_results_pto_mdd:
        try:
            lb = int(label.split("LB=")[1].split(",")[0].strip())
            n1 = float(label.split("n1=")[1].split(")")[0].strip())
        except Exception:
            lb, n1 = None, None
        perf = compute_performance(apply_tc(results, tc_rate))
        benchmark_rows.append({
            "Model"    : "PTO-MDD",
            "lam"      : None,
            "Lookback" : lb,
            "n1"       : n1,
            "Ann.Ret"  : perf["Ann.Ret"],
            "Sharpe"   : perf["Sharpe"],
            "CVaR(5%)" : perf["CVaR(5%)"],
            "MDD"      : perf["MDD"],
            "Calmar"   : perf["Calmar"],
            "HHI"      : perf["HHI"],
        })

    for results, label in all_results_mvo:
        try:
            lb = int(label.split("LB=")[1].split(")")[0].strip())
        except Exception:
            lb = None
        perf = compute_performance(apply_tc(results, tc_rate))
        benchmark_rows.append({
            "Model"    : "PTO-MVO",
            "lam"      : None,
            "Lookback" : lb,
            "n1"       : None,
            "Ann.Ret"  : perf["Ann.Ret"],
            "Sharpe"   : perf["Sharpe"],
            "CVaR(5%)" : perf["CVaR(5%)"],
            "MDD"      : perf["MDD"],
            "Calmar"   : perf["Calmar"],
            "HHI"      : perf["HHI"],
        })

    # ── lambda별 DFL-MDD + 벤치마크 합쳐서 저장 ──
    dfs = {}

    for (delta_val, lam_val), results_list in dfl_results_store.items():
        dfl_rows = []
        for results, label in results_list:
            try:
                lb = int(label.split("LB=")[1].split(",")[0].strip())
                n1 = float(label.split("n1=")[1].split(")")[0].strip())
            except Exception:
                lb, n1 = None, None
            perf = compute_performance(apply_tc(results, tc_rate))
            dfl_rows.append({
                "Model"    : "DFL-MDD",
                "lam"      : lam_val,
                "Lookback" : lb,
                "n1"       : n1,
                "Ann.Ret"  : perf["Ann.Ret"],
                "Sharpe"   : perf["Sharpe"],
                "CVaR(5%)" : perf["CVaR(5%)"],
                "MDD"      : perf["MDD"],
                "Calmar"   : perf["Calmar"],
                "HHI"      : perf["HHI"],
            })

        df = pd.DataFrame(dfl_rows + benchmark_rows)
        dfs[lam_val] = df

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            tc_suffix = f"_tc{int(round(tc_rate*10000))}bps" if tc_rate > 0 else ""
            fname = f"{N_STOCKS}_inds_lam{lam_val}{tc_suffix}.csv"
            fpath = os.path.join(save_dir, fname)
            df.to_csv(fpath, index=False)
            print(f"  ✓ CSV 저장: {fpath}")

    return dfs
