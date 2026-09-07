import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cvxpy as cp

__all__ = ["train_pto_mvo", "backtest_pto_mvo"]

def train_pto_mvo(pred_model, train_samples, val_samples=None,
                  epochs=50, batch_size=16, lr=1e-4, patience=10):
    optimizer = optim.Adam(pred_model.parameters(), lr=lr, weight_decay=1e-4)
    zs      = torch.tensor(np.array([s[0] for s in train_samples]), dtype=torch.float32)
    r_reals = torch.tensor(np.array([s[1] for s in train_samples]), dtype=torch.float32)

    if val_samples is not None:
        zs_val = torch.tensor(np.array([s[0] for s in val_samples]), dtype=torch.float32)
        rs_val = torch.tensor(np.array([s[1] for s in val_samples]), dtype=torch.float32)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    print("\n── PTO-MVO Training (MSE + Val Early Stopping) ──")
    pred_model.train()
    for epoch in range(epochs):
        perm    = torch.randperm(len(train_samples))
        ep_loss = []
        for i in range(0, len(train_samples), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            r_hat = pred_model(zs[idx])
            loss  = F.mse_loss(r_hat, r_reals[idx])
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())

        tr_loss = np.mean(ep_loss)

        if val_samples is not None:
            pred_model.eval()
            with torch.no_grad():
                val_loss = F.mse_loss(pred_model(zs_val), rs_val).item()
            pred_model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in pred_model.state_dict().items()}
                no_improve    = 0
                marker = "*"
            else:
                no_improve += 1
                marker = f"({no_improve}/{patience})"

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  train={tr_loss:.6f}  val={val_loss:.6f}  {marker}")

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}  (best val={best_val_loss:.6f})")
                break
        else:
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  mse = {tr_loss:.6f}")

    if best_state is not None:
        pred_model.load_state_dict(best_state)

    return pred_model


_SOLVER_MAP = {
    "CLARABEL": cp.CLARABEL,
    "ECOS":     cp.ECOS,
    "SCS":      cp.SCS,
    "OSQP":     cp.OSQP,
}


def _report_infeasible(infeas_log, n_windows):
    """solver 실패 건수와 사유(prob.status)를 출력."""
    n = len(infeas_log)
    if n == 0:
        print("\n  \u2713 Fallback \uc5c6\uc74c (0%)")
        return
    rate = n / n_windows if n_windows else float("nan")
    print(f"\n  \u26a0 Fallback \ubc1c\uc0dd: {n}/{n_windows} ({rate:.2%})")
    for status, cnt in Counter(infeas_log).most_common():
        print(f"      {status}: {cnt}")


def _cp_solver(name):
    """문자열 solver 이름 → CVXPY solver 상수."""
    if not isinstance(name, str):
        return name
    return _SOLVER_MAP.get(name.upper(), cp.CLARABEL)


def _solve_mvo(mu, Sigma, delta, x_min, x_max, gamma=0.05,
               solve_method="CLARABEL", infeas_counter=None):
    m = len(mu)
    x = cp.Variable(m)
    objective   = cp.Maximize(mu @ x - (delta / 2) * cp.quad_form(x, Sigma)
                               - gamma * cp.sum_squares(x))
    constraints = [cp.sum(x) == 1, x >= x_min, x <= x_max]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=_cp_solver(solve_method), verbose=False)
    if x.value is None:
        if infeas_counter is not None:
            infeas_counter.append(prob.status or "solve_failed")
        # Fallback: x_max 클램핑 후 재정규화 (pto_mdd fallback과 동일 방식)
        w = np.clip(np.ones(m) / m, x_min, x_max)
        return w / w.sum()
    return x.value

def backtest_pto_mvo(pred_model, rebal_samples, N, d, C,
                     delta=1.0, x_min=0.0, x_max=0.30, gamma=0.05,
                     is_mean=None, is_std=None,
                     stock_names=None, rebal=None, solve_method="CLARABEL"):
    m        = rebal_samples[0][1].shape[1]
    lookback = rebal_samples[0][0].shape[0] // m
    names    = stock_names if stock_names else [f"S{j+1}" for j in range(m)]
    results  = []
    cum_pv   = [1.0]   # 누적 portfolio value (이미지 MDD와 동일 기준)

    print("\n── Backtest : PTO-MVO ──")
    print(f"{'Win':>4}  {'R_real':>8}  {'MDD(%)':>8}  {'Top-3 weights'}")
    print("-" * 65)

    infeas_log = []   # solver 실패 시 prob.status 기록

    pred_model.eval()
    for i, (z_np, r_np) in enumerate(rebal_samples):
        z = torch.tensor(z_np[None], dtype=torch.float32)
        with torch.no_grad():
            r_hat = pred_model(z)[0].numpy()

        mu = r_hat.mean(axis=0)

        # Sigma: lookback 실제 수익률로 추정 (표준 MVO)
        # is_mean/is_std가 제공되면 역정규화 후 sample covariance 사용
        if is_mean is not None and is_std is not None:
            z_raw = z_np.reshape(lookback, m) * is_std + is_mean  # 역정규화
            Sigma = np.cov(z_raw.T) + 1e-4 * np.eye(m)
        else:
            Sigma = np.cov(r_hat.T) + 1e-4 * np.eye(m)

        w     = _solve_mvo(mu, Sigma, delta, x_min, x_max, gamma,
                           solve_method=solve_method,
                           infeas_counter=infeas_log)

        y_real = np.cumsum(r_np, axis=0)
        w_real = y_real @ w

        # HORIZON > REBAL인 경우 실제 보유 기간만큼 잘라서 사용
        if rebal is not None:
            w_real = w_real[:rebal]

        R_real = w_real[-1] / (d * C)

        base    = cum_pv[-1]
        cum_pv.extend((base * (1 + w_real)).tolist())

        # 로그용: 해당 윈도우 내 per-window MDD
        pv_w    = 1 + w_real
        rmax_w  = np.maximum.accumulate(pv_w)
        M_real  = np.max((rmax_w - pv_w) / (rmax_w + 1e-10))

        n_active = int(np.sum(w > 0.01))
        top3     = {names[j]: round(w[j], 3) for j in np.argsort(w)[-3:][::-1]}
        results.append({
            "window" : i + 1,
            "weights": w,
            "w_real" : w_real,
            "R_real" : R_real,
            "M_real" : M_real,
        })
        print(f"  {i+1:3d}  {R_real:8.4f}  {M_real:8.4%}  n={n_active:2d}  {top3}")

    _report_infeasible(infeas_log, len(results))
    return results
