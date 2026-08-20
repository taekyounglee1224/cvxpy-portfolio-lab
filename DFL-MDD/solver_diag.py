"""
solver_diag.py
──────────────
교수님 요청 진단 (2·3·4번).

  2. solver 비교 (ECOS / CLARABEL / SCS)   → solve_with_solver, compare_solvers
  3. DFL-MVO delta 스케일 실험             → delta_sweep_mvo
  4. feasible 구간만 성과 평가             → split_by_feasibility, perf_feasible_only

사용법
------
  import importlib, solver_diag
  importlib.reload(solver_diag)
  from solver_diag import compare_solvers, delta_sweep_mvo, split_by_feasibility
"""

import numpy as np
import torch
import warnings

__all__ = [
    "solve_with_solver",
    "compare_solvers",
    "delta_sweep_mvo",
    "split_by_feasibility",
    "perf_feasible_only",
]


# ══════════════════════════════════════════════
# 2. Solver 비교
# ══════════════════════════════════════════════

def solve_with_solver(y_hat, opt_layer, n1, C, x_min, x_max,
                      Sigma_list=None, solve_method="ECOS", verbose_first=True):
    """
    지정 solver로 solve. (weights, n_fail, messages) 반환.
    solve_portfolio와 동일하되 solver를 파라미터로 받고 실패 메시지를 보관.
    """
    batch, N, m = y_hat.shape
    n1C   = torch.tensor(n1 * C, dtype=torch.float64)
    xmn   = torch.tensor(x_min,  dtype=torch.float64)
    xmx   = torch.tensor(x_max,  dtype=torch.float64)

    xs, msgs = [], []
    for b in range(batch):
        try:
            if Sigma_list is not None:
                L_b = torch.linalg.cholesky(Sigma_list[b].double())
                out = opt_layer(y_hat[b].double(), n1C, xmn, xmx, L_b,
                                solver_args={"solve_method": solve_method})
            else:
                out = opt_layer(y_hat[b].double(), n1C, xmn, xmx,
                                solver_args={"solve_method": solve_method})
            x_star_b = out[0]
        except Exception as e:
            msgs.append(str(e)[:150])
            if verbose_first and len(msgs) == 1:
                print(f"    [{solve_method}] 첫 실패 메시지: {str(e)[:150]}")
            x_raw = torch.softmax(y_hat[b, -1, :], dim=0)
            x_star_b = (torch.clamp(x_raw, min=x_min, max=x_max)).double()
            x_star_b = x_star_b / x_star_b.sum()
        xs.append(x_star_b.float())

    return torch.stack(xs, dim=0), len(msgs), msgs


def compare_solvers(pred_model, opt_layer, rebal_samples,
                    n1, C, x_min, x_max, delta, is_mean, is_std,
                    solvers=(None, "ECOS", "CLARABEL", "SCS")):
    """solvers에 None을 넣으면 cvxpylayers 기본 설정(solver_args 미지정)으로 실행."""
    """
    동일 백테스트 구간에 대해 여러 solver의 실패율·해 차이를 비교.

    Returns
    -------
    dict : {solver: {"n_fail", "n_total", "rate", "weights", "msgs"}}
    """
    m  = rebal_samples[0][1].shape[1]
    lb = rebal_samples[0][0].shape[0] // m

    # 예측·공분산은 solver와 무관하므로 한 번만 계산
    Y, SIG = [], []
    pred_model.eval()
    for z_np, _ in rebal_samples:
        z = torch.tensor(z_np[None], dtype=torch.float32)
        with torch.no_grad():
            r_hat = pred_model(z)
        Y.append(torch.cumsum(r_hat, dim=1))
        if is_mean is not None and is_std is not None:
            z_raw = z_np.reshape(lb, m) * is_std + is_mean
            S = np.cov(z_raw.T) + 1e-4 * np.eye(m)
            SIG.append([torch.tensor(S, dtype=torch.float64)])
        else:
            SIG.append(None)

    out = {}
    for sv in solvers:
        print(f"\n  ▶ solver = {sv}")
        n_fail, W, all_msgs = 0, [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for y_hat, sig in zip(Y, SIG):
                w, nf, msgs = solve_with_solver(
                    y_hat.detach(), opt_layer, n1, C, x_min, x_max,
                    sig if delta > 0 else None, solve_method=sv)
                n_fail += nf
                all_msgs += msgs
                W.append(w[0].numpy())
        W = np.vstack(W)
        n_tot = len(rebal_samples)
        out[sv] = {"n_fail": n_fail, "n_total": n_tot,
                   "rate": n_fail / n_tot, "weights": W, "msgs": all_msgs}
        print(f"    실패 {n_fail}/{n_tot} = {n_fail/n_tot:.1%} | "
              f"HHI {(W**2).sum(1).mean():.3f} | "
              f"0인 비율 {(W < 1e-6).mean():.1%}")
    return out


# ══════════════════════════════════════════════
# 3. DFL-MVO delta sweep (재학습 없이 layer만 바꿔 solve)
# ══════════════════════════════════════════════

def delta_sweep_mvo(pred_model, rebal_samples, build_layer_fn, N, m,
                    gamma, x_min, x_max, is_mean, is_std,
                    deltas=(20, 200, 2000, 20000),
                    solve_method="ECOS"):
    """
    학습된 DFL-MVO 예측모델을 고정한 채 delta만 바꿔 재-최적화.
    delta 증가 → 위험 항 강화 → 집중도 완화 여부 확인.

    build_layer_fn : dfl_mvo.build_mvo_layer
    Returns : dict {delta: {"HHI","MaxW","nActive","weights","fail_rate"}}
    """
    mm = rebal_samples[0][1].shape[1]
    lb = rebal_samples[0][0].shape[0] // mm

    Y, SIG = [], []
    pred_model.eval()
    for z_np, _ in rebal_samples:
        z = torch.tensor(z_np[None], dtype=torch.float32)
        with torch.no_grad():
            r_hat = pred_model(z)
        Y.append(torch.cumsum(r_hat, dim=1))
        z_raw = z_np.reshape(lb, mm) * is_std + is_mean
        S = np.cov(z_raw.T) + 1e-4 * np.eye(mm)
        SIG.append(torch.tensor(S, dtype=torch.float64))

    res = {}
    print(f"{'delta':>8} | {'HHI':>6} {'MaxW':>7} {'n>1%':>6} {'실패율':>7}")
    print("-" * 45)
    for d in deltas:
        layer = build_layer_fn(N, m, gamma, delta=d)
        W, n_fail = [], 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for y_hat, S in zip(Y, SIG):
                try:
                    L = torch.linalg.cholesky(S.double())
                    out = layer(y_hat[0].double(),
                                torch.tensor(x_min, dtype=torch.float64),
                                torch.tensor(x_max, dtype=torch.float64), L,
                                solver_args={"solve_method": solve_method})
                    w = out[0].detach().numpy()
                except Exception:
                    n_fail += 1
                    w = torch.softmax(y_hat[0, -1, :], dim=0).detach().numpy()
                w = np.clip(w, 0, None); w = w / w.sum()
                W.append(w)
        W = np.vstack(W)
        hhi  = float((W**2).sum(1).mean())
        maxw = float(W.max(1).mean())
        nact = float((W > 0.01).sum(1).mean())
        rate = n_fail / len(rebal_samples)
        res[d] = {"HHI": hhi, "MaxW": maxw, "nActive": nact,
                  "weights": W, "fail_rate": rate}
        print(f"{d:>8} | {hhi:>6.3f} {maxw:>7.3f} {nact:>6.2f} {rate:>7.1%}")
    return res


# ══════════════════════════════════════════════
# 4. feasible 구간만 성과 평가
# ══════════════════════════════════════════════

def split_by_feasibility(results, tol=1e-6):
    """
    weight 특성으로 feasible(최적화 성공) / infeasible(fallback) 분리.

    판별: softmax fallback은 0인 weight를 만들 수 없으므로,
          '정확히 0인 자산이 하나도 없는' 윈도우를 fallback으로 간주.

    Returns
    -------
    (feasible_results, fallback_results, mask)  mask: True=feasible
    """
    mask = []
    for r in results:
        w = np.asarray(r["weights"], dtype=float)
        mask.append(bool((np.abs(w) < tol).any()))
    mask = np.array(mask)
    feas = [r for r, m in zip(results, mask) if m]
    fall = [r for r, m in zip(results, mask) if not m]
    return feas, fall, mask


def perf_feasible_only(results, label="", full_np=None, REBAL=None, tol=1e-6):
    """
    전체 / feasible-only / fallback-only 성과를 나란히 계산.

    주의: feasible-only는 기간이 불연속이므로 equity curve를 이어붙인
          '가상 성과'임. 해석 시 이 점을 명시할 것.
    """
    from performance import compute_performance

    feas, fall, mask = split_by_feasibility(results, tol)
    rows = []
    for tag, sub in (("All", results), ("Feasible-only", feas), ("Fallback-only", fall)):
        if not sub:
            continue
        p = compute_performance(sub, f"{label} [{tag}]",
                                full_np=full_np, REBAL=REBAL)
        p["n_windows"] = len(sub)
        p["share"]     = len(sub) / len(results)
        rows.append(p)
    return rows, mask
