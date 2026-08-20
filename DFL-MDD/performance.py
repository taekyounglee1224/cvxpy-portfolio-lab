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
    # 표준 Calmar: 연환산 수익률(Ann.Ret) / MDD
    ann_ret = _annualized_return(equity)
    mdd     = _mdd(equity)
    return float(ann_ret / (mdd + 1e-10))


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


def _mdd_uncompounded(results):
    """
    Uncompounded(가법) 누적수익 경로 기준 절대 최대낙폭 (Reviewer #1).

    전체 테스트 기간의 일별 포트폴리오 수익을 이어붙여 산술 누적:
        cum_t = Σ r_p,s  (s<=t)
    낙폭은 상대비율이 아닌 절대 차이:
        MDD_abs = max_t ( max_{s<=t} cum_s − cum_t )

    최적화 모델(제약식)이 사용하는 drawdown 정의와 동일한 스케일.
    """
    daily = []
    for res in results:
        w_real = np.asarray(res["w_real"], dtype=float)   # 윈도우 내 누적경로
        # 윈도우 누적경로 → 일별 수익으로 환원 후 이어붙임
        daily.append(np.diff(np.concatenate([[0.0], w_real])))
    if not daily:
        return float("nan")
    r = np.concatenate(daily)
    cum      = np.concatenate([[0.0], np.cumsum(r)])
    run_max  = np.maximum.accumulate(cum)
    return float(np.max(run_max - cum))


def _turnover(results, full_np=None, REBAL=None, one_way=True):
    """
    평균 turnover (Reviewer #21 / 교수 코멘트).

    full_np·REBAL 제공 시 drift 반영:
      리밸런싱 직전 실현수익으로 표류한 실제 보유 weight 기준.
        gross_j   = Π(1+r) over 직전 보유기간
        w_drift_j = w_prev_j gross_j / Σ_l w_prev_l gross_l
        turnover  = Σ_j |w_curr_j − w_drift_j|      (one-way)
    미제공 시 타깃 weight 차분(drift 미반영).
    """
    use_drift = (full_np is not None) and (REBAL is not None)
    tos, prev_w, prev_idx = [], None, None

    for res in results:
        w = np.asarray(res["weights"], dtype=float)
        if prev_w is not None:
            if use_drift and (prev_idx is not None):
                R_hold  = full_np[prev_idx : prev_idx + REBAL]
                gross   = np.prod(1.0 + R_hold, axis=0)
                drifted = prev_w * gross
                s       = drifted.sum()
                w_drift = drifted / s if s > 0 else prev_w
                to = float(np.sum(np.abs(w - w_drift)))
            else:
                to = float(np.sum(np.abs(w - prev_w)))
            tos.append(to if one_way else 0.5 * to)
        prev_w   = w
        prev_idx = res.get("date_idx")

    return float(np.mean(tos)) if tos else float("nan")


# ──────────────────────────────────────────────
# Transaction Cost 유틸
# ──────────────────────────────────────────────

def apply_tc(results, tc_rate=0.0, full_np=None, REBAL=None):
    """
    Transaction cost 사후(post-hoc) 반영. (one-way)

    turnover 정의 (Reviewer #21 / 교수 코멘트):
      full_np·REBAL 제공 시 → drift 반영: 리밸런싱 직전 실현수익으로 표류한
        실제 보유 weight(w_drift) 기준으로 turnover 계산.
          gross_j   = Π (1 + r) over 직전 보유기간          (자산별 총수익)
          w_drift_j = w_prev_j gross_j / Σ_l w_prev_l gross_l
          turnover  = Σ_j |w_curr_j − w_drift_j|            (one-way)
      미제공 시 → 기존 방식(타깃 weight 차분, drift 미반영) — 하위 호환.

    Parameters
    ----------
    results : backtest_* 반환 리스트  (각 dict에 'weights', 'w_real'; drift 시 'date_idx' 권장)
    tc_rate : float
    full_np : (T, m) 일별 수익률.  drift 반영에 필요.
    REBAL   : int   보유일수.       drift 반영에 필요.

    Returns
    -------
    adjusted : 동일 구조 리스트 (w_real만 TC 반영)
    """
    if tc_rate == 0.0:
        return results

    use_drift = (full_np is not None) and (REBAL is not None)

    adjusted = []
    prev_w   = None
    prev_idx = None

    for res in results:
        w      = np.array(res["weights"], dtype=float)
        w_real = np.array(res["w_real"], dtype=float)

        if prev_w is not None:
            if use_drift and (prev_idx is not None):
                # 직전 리밸런싱 이후 보유기간 동안 표류
                R_hold  = full_np[prev_idx : prev_idx + REBAL]
                gross   = np.prod(1.0 + R_hold, axis=0)
                drifted = prev_w * gross
                s       = drifted.sum()
                w_drift = drifted / s if s > 0 else prev_w
                turnover = float(np.sum(np.abs(w - w_drift)))
            else:
                turnover = float(np.sum(np.abs(w - prev_w)))   # 기존(미반영)
            tc = tc_rate * turnover
            w_real = (1.0 - tc) * (1.0 + w_real) - 1.0

        adjusted.append({**res, "w_real": w_real})
        prev_w   = w
        prev_idx = res.get("date_idx")   # 벤치마크엔 있음; DFL엔 없을 수 있음

    return adjusted


# ──────────────────────────────────────────────
# 공개 API
# ──────────────────────────────────────────────

def compute_performance(results, label="", full_np=None, REBAL=None):
    """
    results : backtest_* 반환 리스트
    label   : 전략 이름 (출력용)
    full_np, REBAL : drift 반영 turnover 계산용 (Reviewer #21). 미제공 시 근사.

    Returns
    -------
    dict : { label, Ann.Ret, Sharpe, CVaR(5%), MDD, MDD_abs, Calmar, HHI, Turnover }
      - MDD      : compounded wealth 기준 상대 낙폭
      - MDD_abs  : uncompounded 누적수익 기준 절대 낙폭 (최적화 모델과 동일 정의)
      - Turnover : 평균 one-way turnover (drift 반영 시 실제 거래량)
    """
    equity = build_equity_curve(results)
    return {
        "label"    : label,
        "Ann.Ret"  : _annualized_return(equity),
        "Sharpe"   : _sharpe(equity),
        "CVaR(5%)" : _cvar(equity),
        "MDD"      : _mdd(equity),
        "MDD_abs"  : _mdd_uncompounded(results),
        "Calmar"   : _calmar(equity),
        "HHI"      : _hhi(results),
        "Turnover" : _turnover(results, full_np=full_np, REBAL=REBAL),
    }


def print_performance_table(all_results, title=None, full_np=None, REBAL=None):
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
        rows.append(compute_performance(results, label,
                                        full_np=full_np, REBAL=REBAL))

    df_raw = pd.DataFrame(rows).set_index("label")

    # 포맷 적용 (표시용 복사본)
    df_fmt = df_raw.copy()
    df_fmt["Ann.Ret"]   = df_raw["Ann.Ret"].map("{:+.2%}".format)
    df_fmt["Sharpe"]    = df_raw["Sharpe"].map("{:.3f}".format)
    df_fmt["CVaR(5%)"]  = df_raw["CVaR(5%)"].map("{:.2%}".format)
    df_fmt["MDD"]       = df_raw["MDD"].map("{:.2%}".format)
    df_fmt["MDD_abs"]   = df_raw["MDD_abs"].map("{:.4f}".format)
    df_fmt["Turnover"]  = df_raw["Turnover"].map("{:.4f}".format)
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
                                title=None, full_np=None, REBAL=None):
    """
    TC rate별 성과 비교표를 한 번에 출력.

    Parameters
    ----------
    all_results : list of (results_list, label) tuples
                  drift TC를 쓰려면 각 results dict에 'date_idx' 필요
                  (benchmarks.attach_date_idx로 DFL/PTO에 부착).
    tc_rates    : iterable of float
    title       : 출력 상단 제목 (optional)
    full_np, REBAL : drift 반영 TC용 (Reviewer #21). 미제공 시 기존 방식.

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
            adj = apply_tc(results, tc_rate, full_np=full_np, REBAL=REBAL)
            rows.append(compute_performance(adj, label, full_np=full_np, REBAL=REBAL))

        df_raw = pd.DataFrame(rows).set_index("label")
        df_fmt = df_raw.copy()
        df_fmt["Ann.Ret"]   = df_raw["Ann.Ret"].map("{:+.2%}".format)
        df_fmt["Sharpe"]    = df_raw["Sharpe"].map("{:.3f}".format)
        df_fmt["CVaR(5%)"]  = df_raw["CVaR(5%)"].map("{:.2%}".format)
        df_fmt["MDD"]       = df_raw["MDD"].map("{:.2%}".format)
        df_fmt["MDD_abs"]   = df_raw["MDD_abs"].map("{:.4f}".format)
        df_fmt["Turnover"]  = df_raw["Turnover"].map("{:.4f}".format)
        df_fmt["HHI"]       = df_raw["HHI"].map("{:.4f}".format)

        print(f"\n  ── {tc_label} ──")
        print(df_fmt.to_string())

        dfs[tc_rate] = df_raw

    print()
    return dfs


def _parse_lb_n1(label):
    """label에서 (Lookback, n1) 추출. 없으면 None."""
    lb, n1 = None, None
    if "LB=" in label:
        try:
            lb = int(label.split("LB=")[1].split(",")[0].split(")")[0].strip())
        except Exception:
            lb = None
    if "n1=" in label:
        try:
            n1 = float(label.split("n1=")[1].split(")")[0].strip())
        except Exception:
            n1 = None
    return lb, n1


def build_metrics_dataframe(dfl_results_store,
                             all_results_pto_mdd,
                             all_results_mvo,
                             tc_rate=0.0,
                             save_dir=None,
                             N_STOCKS="",
                             bench_store=None,
                             full_np=None,
                             REBAL=None):
    """
    lambda별로 DataFrame을 만들고 CSV 저장.
    각 CSV = 해당 lambda의 DFL-MDD 결과 + 전체 벤치마크.

    Parameters
    ----------
    dfl_results_store   : dict  {(delta, lam): [(results, label), ...]}
    all_results_pto_mdd : list of (results, label)
    all_results_mvo     : list of (results, label)
    tc_rate             : float  transaction cost rate (default 0.0)
    save_dir            : str or None  저장 폴더 (예: "./csv")
    N_STOCKS            : int or str  파일명 구분용 (예: 10, 30)
    bench_store         : dict or None  {label: results}
                          EW/GMV/hist-MVO 등. label에서 모델명·LB 자동 파싱.

    Returns
    -------
    dfs : dict  { lam_val: pd.DataFrame }
    """

    # ── 벤치마크 rows 먼저 계산 (모든 lambda CSV에 공통 포함) ──
    benchmark_rows = []

    # EW / GMV / hist-MVO
    if bench_store is not None:
        for blabel, bres in bench_store.items():
            lb, _ = _parse_lb_n1(blabel)
            model = blabel.split("(")[0].strip()   # "GMV (LB=252)" → "GMV"
            perf = compute_performance(apply_tc(bres, tc_rate, full_np=full_np, REBAL=REBAL),
                                        full_np=full_np, REBAL=REBAL)
            benchmark_rows.append({
                "Model"    : model,
                "lam"      : None,
                "Lookback" : lb,
                "n1"       : None,
                "Ann.Ret"  : perf["Ann.Ret"],
                "Sharpe"   : perf["Sharpe"],
                "CVaR(5%)" : perf["CVaR(5%)"],
                "MDD"      : perf["MDD"],
                "Calmar"   : perf["Calmar"],
                "MDD_abs"  : perf["MDD_abs"],
                "Turnover" : perf["Turnover"],
                "HHI"      : perf["HHI"],
            })

    for results, label in all_results_pto_mdd:
        try:
            lb = int(label.split("LB=")[1].split(",")[0].strip())
            n1 = float(label.split("n1=")[1].split(")")[0].strip())
        except Exception:
            lb, n1 = None, None
        perf = compute_performance(apply_tc(results, tc_rate, full_np=full_np, REBAL=REBAL),
                                   full_np=full_np, REBAL=REBAL)
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
            "MDD_abs"  : perf["MDD_abs"],
            "Turnover" : perf["Turnover"],
            "HHI"      : perf["HHI"],
        })

    for results, label in all_results_mvo:
        try:
            lb = int(label.split("LB=")[1].split(")")[0].strip())
        except Exception:
            lb = None
        perf = compute_performance(apply_tc(results, tc_rate, full_np=full_np, REBAL=REBAL),
                                   full_np=full_np, REBAL=REBAL)
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
            "MDD_abs"  : perf["MDD_abs"],
            "Turnover" : perf["Turnover"],
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
            perf = compute_performance(apply_tc(results, tc_rate, full_np=full_np, REBAL=REBAL),
                                   full_np=full_np, REBAL=REBAL)
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
                "MDD_abs"  : perf["MDD_abs"],
                "Turnover" : perf["Turnover"],
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
