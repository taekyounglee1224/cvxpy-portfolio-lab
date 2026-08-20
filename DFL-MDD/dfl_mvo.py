"""
dfl_mvo.py
──────────
DFL-MVO: end-to-end mean-variance 포트폴리오 (drawdown 제약 없음).

DFL-MDD와의 유일한 차이:
  - optimization layer에서 drawdown 제약(보조변수 u, n1) 전부 제거
    → 순수 MVO:  max ŷ_N·x − (δ/2)‖Lᵀx‖²   s.t.  Σx=1,  x_min≤x≤x_max
  - task loss는 DFL-MDD와 동일:  L = λ(−Sharpe) + (1−λ)·MDD_real
    (하드 제약만 제거 → Reviewer #9 ablation: "DFL without the hard drawdown constraint")

나머지(예측모델·loss·학습루프·지표)는 dfl_mdd에서 재사용.

사용법
------
  import importlib, dfl_mvo
  importlib.reload(dfl_mvo)
  from dfl_mvo import build_mvo_layer, train_dfl_mvo, backtest_dfl_mvo
"""

import numpy as np
import cvxpy as cp
import torch
import torch.optim as optim
import warnings
from tqdm.auto import tqdm
from cvxpylayers.torch import CvxpyLayer

# 공유 컴포넌트 재사용
from dfl_mdd import (
    PredictionModel,
    compute_cumulative_path,
    compute_realized_path,
    compute_return,
    compute_max_drawdown,
    compute_sharpe,
    dfl_loss,
)

__all__ = [
    "build_mvo_layer",
    "solve_portfolio_mvo",
    "forward_pass_mvo",
    "train_dfl_mvo",
    "backtest_dfl_mvo",
]


# =============================================================================
# MVO Optimization Layer  (drawdown 제약 없음)
# =============================================================================
def build_mvo_layer(N, m, gamma=0.0, delta=0.0):
    """
    순수 mean-variance layer (long-only, full-investment).
      max  ŷ_N·x − (δ/2)‖Lᵀx‖² − γ‖x‖²
      s.t. Σx = 1,  x_min ≤ x ≤ x_max
    보조변수 u·drawdown 제약 없음.
    """
    x     = cp.Variable(m, name="x")
    Y_hat = cp.Parameter((N, m), name="Y_hat")
    x_min = cp.Parameter(name="x_min")
    x_max = cp.Parameter(name="x_max")

    if delta > 0:
        L_p       = cp.Parameter((m, m), name="L")   # lower-triangular Cholesky
        risk_term = (delta / 2) * cp.sum_squares(L_p.T @ x)
        objective = cp.Maximize(Y_hat[N - 1] @ x - risk_term
                                - gamma * cp.sum_squares(x))
        params    = [Y_hat, x_min, x_max, L_p]
    else:
        objective = cp.Maximize(Y_hat[N - 1] @ x - gamma * cp.sum_squares(x))
        params    = [Y_hat, x_min, x_max]

    constraints = [x >= x_min, x <= x_max, cp.sum(x) == 1]
    problem = cp.Problem(objective, constraints)
    assert problem.is_dcp(), "MVO problem is not DCP!"
    return CvxpyLayer(problem, parameters=params, variables=[x])


# =============================================================================
# Solve  (n1 없음)
# =============================================================================
def solve_portfolio_mvo(y_hat, opt_layer, x_min, x_max,
                        Sigma_list=None, infeas_counter=None):
    batch, N, m = y_hat.shape
    x_min_val = torch.tensor(x_min, dtype=torch.float64)
    x_max_val = torch.tensor(x_max, dtype=torch.float64)

    x_stars = []
    for b in range(batch):
        try:
            if Sigma_list is not None:
                L_b = torch.linalg.cholesky(Sigma_list[b].double())
                (x_star_b,) = opt_layer(
                    y_hat[b].double(), x_min_val, x_max_val, L_b,
                    solver_args={"solve_method": "ECOS"},
                )
            else:
                (x_star_b,) = opt_layer(
                    y_hat[b].double(), x_min_val, x_max_val,
                    solver_args={"solve_method": "ECOS"},
                )
        except Exception as e:
            if infeas_counter is not None:
                infeas_counter.append(str(e)[:120] or "solve_failed")
            x_raw     = torch.softmax(y_hat[b, -1, :], dim=0)
            x_clamped = torch.clamp(x_raw, min=x_min, max=x_max)
            x_star_b  = (x_clamped / x_clamped.sum()).double()
        x_stars.append(x_star_b.float())

    return torch.stack(x_stars, dim=0)


# =============================================================================
# Forward pass  (DFL-MDD와 동일, solve만 MVO)
# =============================================================================
def forward_pass_mvo(z, r_real, pred_model, opt_layer, C, d, x_min, x_max, lam,
                     is_mean=None, is_std=None, delta=0.0):
    r_hat = pred_model(z)
    y_hat = compute_cumulative_path(r_hat)

    Sigma_list = None
    if is_mean is not None and is_std is not None:
        batch     = z.shape[0]
        m_dim     = r_hat.shape[2]
        lb        = z.shape[1] // m_dim
        is_mean_t = torch.tensor(is_mean, dtype=torch.float32)
        is_std_t  = torch.tensor(is_std,  dtype=torch.float32)
        z_raw     = z.reshape(batch, lb, m_dim) * is_std_t + is_mean_t
        Sigma_list = []
        for b in range(batch):
            z_b = z_raw[b].detach().numpy()
            S   = np.cov(z_b.T) + 1e-4 * np.eye(m_dim)
            Sigma_list.append(torch.tensor(S, dtype=torch.float64))

    x_star = solve_portfolio_mvo(y_hat, opt_layer, x_min, x_max,
                                 Sigma_list if delta > 0 else None)
    y_real = compute_cumulative_path(r_real)
    w_real = compute_realized_path(x_star, y_real)
    R_real = compute_return(w_real, d, C)
    M_real = compute_max_drawdown(w_real)
    Sharpe = compute_sharpe(x_star, r_real, Sigma_list)
    loss   = dfl_loss(Sharpe, M_real, lam)
    return {"r_hat": r_hat, "y_hat": y_hat, "x_star": x_star,
            "y_real": y_real, "w_real": w_real,
            "R_real": R_real, "M_real": M_real, "Sharpe": Sharpe, "loss": loss}


# =============================================================================
# Train  (train_dfl_mdd과 동일 구조, forward만 MVO / n1 없음)
# =============================================================================
def train_dfl_mvo(pred_model, opt_layer, train_samples, val_samples=None,
                  epochs=50, batch_size=16, lr=1e-4,
                  C=1.0, d=1.0, x_min=0.0, x_max=1.0, lam=0.3,
                  is_mean=None, is_std=None, delta=0.0,
                  patience=10, lr_patience=10, lr_factor=0.5,
                  train_dates=None):
    optimizer = optim.Adam(pred_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )

    zs_tr = torch.tensor(np.array([s[0] for s in train_samples]), dtype=torch.float32)
    rs_tr = torch.tensor(np.array([s[1] for s in train_samples]), dtype=torch.float32)
    if val_samples is not None:
        zs_val = torch.tensor(np.array([s[0] for s in val_samples]), dtype=torch.float32)
        rs_val = torch.tensor(np.array([s[1] for s in val_samples]), dtype=torch.float32)

    best_val_loss  = float("inf")
    best_state     = None
    no_improve     = 0
    inaccurate_log = []

    print("\n── DFL-MVO Training (Val Early Stopping + LR Scheduler) ──")

    for epoch in range(epochs):
        pred_model.train()
        perm    = torch.randperm(len(train_samples))
        ep_loss = []
        for i in range(0, len(train_samples), batch_size):
            idx = perm[i : i + batch_size]
            z_b, r_b = zs_tr[idx], rs_tr[idx]
            optimizer.zero_grad()

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = forward_pass_mvo(
                    z_b, r_b, pred_model, opt_layer,
                    C, d, x_min, x_max, lam,
                    is_mean=is_mean, is_std=is_std, delta=delta,
                )
                n_inaccurate = sum(
                    1 for warning in w if "Inaccurate" in str(warning.message)
                )

            if n_inaccurate > 0:
                entry = {"epoch": epoch + 1, "batch": i // batch_size,
                         "n_inaccurate": n_inaccurate}
                if train_dates is not None:
                    batch_dates = [train_dates[j.item()] for j in idx]
                    entry["date_start"] = min(d0[0] for d0 in batch_dates)
                    entry["date_end"]   = max(d0[1] for d0 in batch_dates)
                inaccurate_log.append(entry)
                continue

            result["loss"].backward()
            optimizer.step()
            ep_loss.append(result["loss"].item())

        tr_loss = np.mean(ep_loss)

        if val_samples is not None:
            pred_model.eval()
            val_losses = []
            for j in range(0, len(val_samples), batch_size):
                z_v = zs_val[j : j + batch_size]
                r_v = rs_val[j : j + batch_size]
                with torch.no_grad():
                    res = forward_pass_mvo(
                        z_v, r_v, pred_model, opt_layer,
                        C, d, x_min, x_max, lam,
                        is_mean=is_mean, is_std=is_std, delta=delta,
                    )
                val_losses.append(res["loss"].item())

            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in pred_model.state_dict().items()}
                no_improve    = 0
                marker = "*"
            else:
                no_improve += 1
                marker = f"({no_improve}/{patience})"

            if (epoch + 1) % 5 == 0 or epoch == 0:
                inacc_str = f"  [inaccurate={len(inaccurate_log)}]" if inaccurate_log else ""
                print(f"  Epoch {epoch+1:3d}/{epochs}  train={tr_loss:.6f}  "
                      f"val={val_loss:.6f}  lr={current_lr:.2e}  {marker}{inacc_str}")

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}  (best val={best_val_loss:.6f})")
                break
        else:
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  loss = {tr_loss:.6f}")

    if best_state is not None:
        pred_model.load_state_dict(best_state)

    if inaccurate_log:
        print(f"\n  ⚠ Inaccurate 발생: 총 {len(inaccurate_log)}회")
    else:
        print("\n  ✓ Inaccurate 없음")

    return pred_model, inaccurate_log


# =============================================================================
# Backtest  (backtest_dfl_mdd과 동일, solve만 MVO / 3개 반환)
# =============================================================================
def backtest_dfl_mvo(pred_model, opt_layer, rebal_samples, N, d, C,
                     x_min=0.0, x_max=1.0,
                     delta=0.0, is_mean=None, is_std=None,
                     stock_names=None, rebal=None):
    m        = rebal_samples[0][1].shape[1]
    lookback = rebal_samples[0][0].shape[0] // m
    names    = stock_names if stock_names else [f"S{j+1}" for j in range(m)]
    results  = []

    print("\n── Backtest : DFL-MVO ──")
    print(f"{'Win':>4}  {'R_real':>8}  {'Sharpe':>8}  {'MDD(%)':>8}  {'Top-3 weights'}")
    print("-" * 75)

    pred_model.eval()
    bt_inaccurate_log = []
    infeas_log        = []

    for i, (z_np, r_np) in enumerate(tqdm(rebal_samples, desc="Backtesting")):
        z      = torch.tensor(z_np[None], dtype=torch.float32)
        r_real = torch.tensor(r_np[None], dtype=torch.float32)

        with torch.no_grad():
            r_hat = pred_model(z)

        Sigma_list = None
        if is_mean is not None and is_std is not None:
            z_raw      = z_np.reshape(lookback, m) * is_std + is_mean
            S          = np.cov(z_raw.T) + 1e-4 * np.eye(m)
            Sigma_list = [torch.tensor(S, dtype=torch.float64)]

        y_hat = compute_cumulative_path(r_hat)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x_star = solve_portfolio_mvo(y_hat.detach(), opt_layer, x_min, x_max,
                                         Sigma_list if delta > 0 else None,
                                         infeas_counter=infeas_log)
            n_inaccurate = sum(
                1 for warning in w if "Inaccurate" in str(warning.message)
            )
        if n_inaccurate > 0:
            bt_inaccurate_log.append({"window": i + 1, "n_inaccurate": n_inaccurate})

        y_real = compute_cumulative_path(r_real)
        w_real = compute_realized_path(x_star, y_real)[0].numpy()

        if rebal is not None:
            w_real = w_real[:rebal]
            r_real = r_real[:, :rebal, :]

        R_real = w_real[-1] / (d * C)

        pv_w    = 1 + w_real
        rmax_w  = np.maximum.accumulate(pv_w)
        M_real  = np.max((rmax_w - pv_w) / (rmax_w + 1e-10))
        sharpe_val = compute_sharpe(x_star, r_real, Sigma_list).item()

        w        = x_star[0].numpy()
        n_active = int(np.sum(w > 0.01))
        top3     = {names[j]: round(w[j], 3) for j in np.argsort(w)[-3:][::-1]}
        results.append({
            "window" : i + 1,
            "weights": w,
            "w_real" : w_real,
            "R_real" : R_real,
            "M_real" : M_real,
            "Sharpe" : sharpe_val,
        })
        print(f"  {i+1:3d}  {R_real:8.4f}  {sharpe_val:8.4f}  {M_real:8.4%}  n={n_active:2d}  {top3}")

    if bt_inaccurate_log:
        print(f"\n  ⚠ Backtest Inaccurate: 총 {len(bt_inaccurate_log)}회")
    else:
        print("\n  ✓ Backtest Inaccurate 없음")

    n_win = len(results)
    n_inf = len(infeas_log)
    infeas_summary = {
        "n_infeasible": n_inf,
        "n_windows":    n_win,
        "rate":         (n_inf / n_win) if n_win else float("nan"),
    }
    if n_inf > 0:
        print(f"\n  ⚠ Infeasible fallback: {n_inf}/{n_win} ({infeas_summary['rate']:.2%})")
    else:
        print("\n  ✓ Infeasible 없음 (0%)")

    return results, bt_inaccurate_log, infeas_summary
