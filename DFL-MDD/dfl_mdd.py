
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import random
from dataclasses import dataclass
from typing import Dict, Tuple
from tqdm import tqdm
import itertools
import torch.nn as nn
import torch.optim as optim


__all__ = [
    "PredictionModel",
    "build_optimization_layer",
    "solve_portfolio",
    "compute_cumulative_path",
    "compute_realized_path",
    "compute_return",
    "compute_max_drawdown",
    "compute_sharpe",
    "dfl_loss",
    "forward_pass",
    "train_dfl_mdd",
    "backtest_dfl_mdd",
    "plot_pnl",
]

# =============================================================================
# Step 1. Prediction Model
# =============================================================================
class PredictionModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, N: int, m: int):
        super().__init__()
        self.N = N
        self.m = m
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N * m),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch = z.shape[0]
        out   = self.net(z)
        return out.view(batch, self.N, self.m)



# =============================================================================
# Step 2. Cumulative Return Path
# =============================================================================
def compute_cumulative_path(r: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(r, dim=1)



# =============================================================================
# Step 3. Optimization Layer
# =============================================================================
def build_optimization_layer(N: int, m: int, gamma: float = 0.01,
                             delta: float = 0.0) -> CvxpyLayer:
    x     = cp.Variable(m,     name="x")
    u     = cp.Variable(N + 1, name="u")
    Y_hat = cp.Parameter((N, m), name="Y_hat")
    n1C   = cp.Parameter(nonneg=True, name="n1C")
    x_min = cp.Parameter(name="x_min")
    x_max = cp.Parameter(name="x_max")

    if delta > 0:
        # risk term: -(delta/2) * ||L^T x||^2  (Cholesky: Sigma = L L^T)
        # cp.sum_squares(L_p.T @ x) 는 DPP-compliant (파라미터가 선형으로 1회 등장)
        L_p       = cp.Parameter((m, m), name="L")   # lower-triangular Cholesky factor
        risk_term = (delta / 2) * cp.sum_squares(L_p.T @ x)
        objective = cp.Maximize(Y_hat[N - 1] @ x - risk_term
                                - gamma * cp.sum_squares(x))
        params    = [Y_hat, n1C, x_min, x_max, L_p]
    else:
        objective = cp.Maximize(Y_hat[N - 1] @ x - gamma * cp.sum_squares(x))
        params    = [Y_hat, n1C, x_min, x_max]

    constraints = [u[0] == 0]
    for k in range(1, N + 1):
        y_k = Y_hat[k - 1]
        constraints.append(u[k] - y_k @ x <= n1C)
        constraints.append(u[k] >= y_k @ x)
        constraints.append(u[k] >= u[k - 1])
    constraints += [x >= x_min, x <= x_max, cp.sum(x) == 1]

    problem = cp.Problem(objective, constraints)
    assert problem.is_dcp(), "Problem is not DCP!"
    return CvxpyLayer(problem, parameters=params, variables=[x, u])



def solve_portfolio(
    y_hat: torch.Tensor,
    opt_layer: CvxpyLayer,
    n1: float,
    C: float,
    x_min: float,
    x_max: float,
    Sigma_list=None,          # list of (m,m) torch.Tensor, delta>0일 때만 사용
) -> torch.Tensor:
    batch, N, m = y_hat.shape
    n1C_val   = torch.tensor(n1 * C, dtype=torch.float64)
    x_min_val = torch.tensor(x_min,  dtype=torch.float64)
    x_max_val = torch.tensor(x_max,  dtype=torch.float64)

    x_stars = []
    for b in range(batch):
        try:
            if Sigma_list is not None:
                # Cholesky 분해: Sigma = L L^T  (L: lower-triangular)
                L_b = torch.linalg.cholesky(Sigma_list[b].double())
                x_star_b, _ = opt_layer(
                    y_hat[b].double(), n1C_val, x_min_val, x_max_val,
                    L_b,
                    solver_args={"solve_method": "ECOS"},
                )
            else:
                x_star_b, _ = opt_layer(
                    y_hat[b].double(), n1C_val, x_min_val, x_max_val,
                    solver_args={"solve_method": "ECOS"},
                )
        except Exception:
            x_raw     = torch.softmax(y_hat[b, -1, :], dim=0)
            x_clamped = torch.clamp(x_raw, min=x_min, max=x_max)
            x_star_b  = (x_clamped / x_clamped.sum()).double()
        x_stars.append(x_star_b.float())

    return torch.stack(x_stars, dim=0)



# =============================================================================
# Step 4. Realized Portfolio Path
# =============================================================================
def compute_realized_path(x_star: torch.Tensor, y_real: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bj, btj -> bt", x_star, y_real)



# =============================================================================
# Step 5. Performance Metrics
# =============================================================================
def compute_return(w_real: torch.Tensor, d: float, C: float) -> torch.Tensor:
    return w_real[:, -1] / (d * C)

def compute_max_drawdown(w_real: torch.Tensor) -> torch.Tensor:
    running_max, _ = torch.cummax(w_real, dim=1)
    drawdown       = running_max - w_real
    return torch.max(drawdown, dim=1).values

def compute_sharpe(
    x_star: torch.Tensor,
    r_real: torch.Tensor,
    Sigma_list=None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-sample Sharpe ratio.

    x_star     : (batch, m)       — portfolio weights
    r_real     : (batch, N, m)    — per-period asset returns
    Sigma_list : list of (m, m) float64 tensors estimated from lookback window.
                 When provided, portfolio variance = x^T Σ x.
                 When None, falls back to sample std of realised portfolio returns.
    """
    # Per-period portfolio returns: shape (batch, N)
    p_real = torch.einsum("bj, btj -> bt", x_star, r_real)
    mu_p   = p_real.mean(dim=1)   # (batch,)

    sharpes = []
    for b in range(x_star.shape[0]):
        if Sigma_list is not None:  
            x_b   = x_star[b].double()
            S_b   = Sigma_list[b]                           # (m, m) float64
            var_p = x_b @ S_b @ x_b                        # scalar
            sig_p = torch.sqrt(var_p.clamp(min=0).float() + eps)
        else:
            sig_p = p_real[b].std(unbiased=False) + eps
        sharpes.append(mu_p[b] / sig_p)

    return torch.stack(sharpes)   # (batch,)



# =============================================================================
# Step 6. DFL Loss  (Sharpe-MDD)
# =============================================================================
def dfl_loss(Sharpe: torch.Tensor, M_real: torch.Tensor, lam: float) -> torch.Tensor:
    """
    lam * (-Sharpe)  +  (1 - lam) * MDD
    Maximise Sharpe while penalising max drawdown.
    """
    return (lam * (-Sharpe) + (1 - lam) * M_real).mean()




# =============================================================================
# Full Pipeline
# =============================================================================
def forward_pass(z, r_real, pred_model, opt_layer, n1, C, d, x_min, x_max, lam,
                 is_mean=None, is_std=None, delta=0.0):
    r_hat  = pred_model(z)
    y_hat  = compute_cumulative_path(r_hat)

    # Sigma 추정: is_mean/is_std 제공 시 항상 추정 (Sharpe 및 delta 공통 사용)
    Sigma_list = None
    if is_mean is not None and is_std is not None:
        batch     = z.shape[0]
        m_dim     = r_hat.shape[2]
        lb        = z.shape[1] // m_dim
        is_mean_t = torch.tensor(is_mean, dtype=torch.float32)
        is_std_t  = torch.tensor(is_std,  dtype=torch.float32)
        z_raw     = z.reshape(batch, lb, m_dim) * is_std_t + is_mean_t   # 역정규화
        Sigma_list = []
        for b in range(batch):
            z_b = z_raw[b].detach().numpy()
            S   = np.cov(z_b.T) + 1e-4 * np.eye(m_dim)
            Sigma_list.append(torch.tensor(S, dtype=torch.float64))

    # solve_portfolio에는 delta>0일 때만 Sigma 전달 (최적화 목적함수용)
    x_star = solve_portfolio(y_hat, opt_layer, n1, C, x_min, x_max,
                             Sigma_list if delta > 0 else None)
    y_real = compute_cumulative_path(r_real)
    w_real = compute_realized_path(x_star, y_real)
    R_real = compute_return(w_real, d, C)
    M_real = compute_max_drawdown(w_real)
    Sharpe = compute_sharpe(x_star, r_real, Sigma_list)   # lookback Σ 사용
    loss   = dfl_loss(Sharpe, M_real, lam)
    return {"r_hat": r_hat, "y_hat": y_hat, "x_star": x_star,
            "y_real": y_real, "w_real": w_real,
            "R_real": R_real, "M_real": M_real, "Sharpe": Sharpe, "loss": loss}

# =============================================================================
# Train (DFL-MDD) — Val Early Stopping
# =============================================================================
def train_dfl_mdd(pred_model, opt_layer, train_samples, val_samples=None,
                  epochs=50, batch_size=16, lr=1e-4,
                  n1=0.10, C=1.0, d=1.0, x_min=0.0, x_max=0.30, lam=0.3,
                  is_mean=None, is_std=None, delta=0.0,
                  patience=10):
    """
    DFL-MDD 학습 함수.
    val_samples가 주어지면 매 epoch val loss를 계산하여 early stopping 수행.
    val_samples는 리밸런싱 간격으로 서브샘플링된 것을 권장 (속도).
    """
    optimizer = optim.Adam(pred_model.parameters(), lr=lr, weight_decay=1e-4)

    zs_tr = torch.tensor(np.array([s[0] for s in train_samples]), dtype=torch.float32)
    rs_tr = torch.tensor(np.array([s[1] for s in train_samples]), dtype=torch.float32)

    if val_samples is not None:
        zs_val = torch.tensor(np.array([s[0] for s in val_samples]), dtype=torch.float32)
        rs_val = torch.tensor(np.array([s[1] for s in val_samples]), dtype=torch.float32)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    print("\n── DFL-MDD Training (with Val Early Stopping) ──")

    for epoch in range(epochs):
        pred_model.train()
        perm    = torch.randperm(len(train_samples))
        ep_loss = []
        for i in range(0, len(train_samples), batch_size):
            idx = perm[i : i + batch_size]
            z_b, r_b = zs_tr[idx], rs_tr[idx]
            optimizer.zero_grad()
            result = forward_pass(
                z_b, r_b, pred_model, opt_layer,
                n1, C, d, x_min, x_max, lam,
                is_mean=is_mean, is_std=is_std, delta=delta,
            )
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
                    res = forward_pass(
                        z_v, r_v, pred_model, opt_layer,
                        n1, C, d, x_min, x_max, lam,
                        is_mean=is_mean, is_std=is_std, delta=delta,
                    )
                val_losses.append(res["loss"].item())

            val_loss = np.mean(val_losses)

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
                print(f"  Epoch {epoch+1:3d}/{epochs}  loss = {tr_loss:.6f}")

    if best_state is not None:
        pred_model.load_state_dict(best_state)

    return pred_model


# =============================================================================
# Backtest
# =============================================================================
def backtest_dfl_mdd(pred_model, opt_layer, rebal_samples, N, d, C,
                     n1=0.10, x_min=0.0, x_max=0.30,
                     delta=0.0, is_mean=None, is_std=None,
                     stock_names=None):
    m        = rebal_samples[0][1].shape[1]
    lookback = rebal_samples[0][0].shape[0] // m
    names    = stock_names if stock_names else [f"S{j+1}" for j in range(m)]
    results  = []
    cum_pv   = [1.0]   # 누적 portfolio value (이미지 MDD와 동일 기준)

    print("\n── Backtest : DFL-MDD ──")
    print(f"{'Win':>4}  {'R_real':>8}  {'Sharpe':>8}  {'MDD(%)':>8}  {'Top-3 weights'}")
    print("-" * 75)

    pred_model.eval()

    for i, (z_np, r_np) in enumerate(tqdm(rebal_samples, desc="Backtesting")):
        z      = torch.tensor(z_np[None], dtype=torch.float32)
        r_real = torch.tensor(r_np[None], dtype=torch.float32)

        with torch.no_grad():
            r_hat = pred_model(z)

        # Sigma 추정: is_mean/is_std 제공 시 항상 추정
        Sigma_list = None
        if is_mean is not None and is_std is not None:
            z_raw      = z_np.reshape(lookback, m) * is_std + is_mean   # 역정규화
            S          = np.cov(z_raw.T) + 1e-4 * np.eye(m)
            Sigma_list = [torch.tensor(S, dtype=torch.float64)]

        y_hat  = compute_cumulative_path(r_hat)
        x_star = solve_portfolio(y_hat.detach(), opt_layer, n1, C, x_min, x_max,
                                 Sigma_list if delta > 0 else None)
        y_real = compute_cumulative_path(r_real)
        w_real = compute_realized_path(x_star, y_real)[0].numpy()

        R_real = w_real[-1] / (d * C)

        base    = cum_pv[-1]
        cum_pv.extend((base * (1 + w_real)).tolist())
        pv_arr  = np.array(cum_pv)
        run_max = np.maximum.accumulate(pv_arr)
        M_real  = np.max((run_max - pv_arr) / (run_max + 1e-10))

        # Sharpe: lookback Σ 기반 포트폴리오 분산
        sharpe_val = compute_sharpe(x_star, r_real, Sigma_list).item()

        w    = x_star[0].numpy()
        top3 = {names[j]: round(w[j], 3) for j in np.argsort(w)[-3:][::-1]}
        results.append({
            "window" : i + 1,
            "weights": w,
            "w_real" : w_real,
            "R_real" : R_real,
            "M_real" : M_real,
            "Sharpe" : sharpe_val,
        })
        print(f"  {i+1:3d}  {R_real:8.4f}  {sharpe_val:8.4f}  {M_real:8.4%}  {top3}")

    return results

# =============================================================================
# PnL Plot
# =============================================================================
def plot_pnl(bt_results: list, horizon: int, label: str = "Portfolio", figsize=(12, 6)):
    pv            = [1.0]
    rebal_indices = [0]

    for res in bt_results:
        w    = res["w_real"]
        base = pv[-1]
        pv.extend((base * (1 + w)).tolist())
        rebal_indices.append(len(pv) - 1)

    pv          = np.array(pv)
    x           = np.arange(len(pv))
    running_max = np.maximum.accumulate(pv)
    drawdown    = (running_max - pv) / (running_max + 1e-10)
    total_ret   = pv[-1] - 1.0
    max_dd      = drawdown.max()
    calmar      = total_ret / (max_dd + 1e-10)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax1.plot(x, pv, color="steelblue", linewidth=1.8, label=label)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for idx in rebal_indices[1:-1]:
        ax1.axvline(idx, color="orange", linestyle=":", linewidth=1.0, alpha=0.7,
                    label="Rebalance" if idx == rebal_indices[1] else "")
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
    ax1.set_ylabel("Portfolio Value")
    ax1.set_title(
        f"Cumulative PnL  |  Return: {total_ret:.2%}  "
        f"Max DD: {max_dd:.2%}  Calmar: {calmar:.2f}"
    )
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.25)

    ax2.fill_between(x, -drawdown * 100, 0, color="crimson", alpha=0.45, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trading Days (BT Period)")
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()

    print(f"\n── PnL Summary ({label}) ──")
    print(f"  Final Value  : {pv[-1]:.4f}")
    print(f"  Total Return : {total_ret:.4%}")
    print(f"  Max Drawdown : {max_dd:.4%}")
    print(f"  Calmar Ratio : {calmar:.4f}")