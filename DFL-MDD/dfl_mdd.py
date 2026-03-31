
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
    "dfl_loss",
    "forward_pass",
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
def build_optimization_layer(N: int, m: int, gamma: float = 0.01) -> CvxpyLayer:
    x     = cp.Variable(m,     name="x")
    u     = cp.Variable(N + 1, name="u")
    Y_hat = cp.Parameter((N, m), name="Y_hat")
    n1C   = cp.Parameter(nonneg=True, name="n1C")
    x_min = cp.Parameter(name="x_min")
    x_max = cp.Parameter(name="x_max")

    # L2 regularization: -gamma * ||x||^2 makes the problem strictly concave,
    # ensuring an interior solution and non-zero KKT gradients for backprop.
    objective   = cp.Maximize(Y_hat[N - 1] @ x - gamma * cp.sum_squares(x))
    constraints = [u[0] == 0]
    for k in range(1, N + 1):
        y_k = Y_hat[k - 1]
        constraints.append(u[k] - y_k @ x <= n1C)
        constraints.append(u[k] >= y_k @ x)
        constraints.append(u[k] >= u[k - 1])
    constraints += [x >= x_min, x <= x_max, cp.sum(x) == 1]

    problem = cp.Problem(objective, constraints)
    assert problem.is_dcp(), "Problem is not DCP!"
    return CvxpyLayer(problem, parameters=[Y_hat, n1C, x_min, x_max], variables=[x, u])



def solve_portfolio(
    y_hat: torch.Tensor,
    opt_layer: CvxpyLayer,
    n1: float,
    C: float,
    x_min: float,
    x_max: float,
) -> torch.Tensor:
    batch, N, m = y_hat.shape
    n1C_val   = torch.tensor(n1 * C, dtype=torch.float64)
    x_min_val = torch.tensor(x_min,  dtype=torch.float64)
    x_max_val = torch.tensor(x_max,  dtype=torch.float64)

    x_stars = []
    for b in range(batch):
        try:
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



# =============================================================================
# Step 6. DFL Loss
# =============================================================================
def dfl_loss(R_real: torch.Tensor, M_real: torch.Tensor, lam: float) -> torch.Tensor:
    return (lam * (-R_real) + (1 - lam) * M_real).mean()




# =============================================================================
# Full Pipeline
# =============================================================================
def forward_pass(z, r_real, pred_model, opt_layer, n1, C, d, x_min, x_max, lam):
    r_hat  = pred_model(z)
    y_hat  = compute_cumulative_path(r_hat)
    x_star = solve_portfolio(y_hat, opt_layer, n1, C, x_min, x_max)
    y_real = compute_cumulative_path(r_real)
    w_real = compute_realized_path(x_star, y_real)
    R_real = compute_return(w_real, d, C)
    M_real = compute_max_drawdown(w_real)
    loss   = dfl_loss(R_real, M_real, lam)
    return {"r_hat": r_hat, "y_hat": y_hat, "x_star": x_star,
            "y_real": y_real, "w_real": w_real,
            "R_real": R_real, "M_real": M_real, "loss": loss}

# =============================================================================
# Backtest
# =============================================================================
def backtest_dfl_mdd(pred_model, opt_layer, rebal_samples, N, d, C,
                     n1=0.10, x_min=0.0, x_max=0.30, stock_names=None):
    m     = rebal_samples[0][1].shape[1]
    names = stock_names if stock_names else [f"S{j+1}" for j in range(m)]
    results = []

    print("\n── Backtest : DFL-MDD ──")
    print(f"{'Win':>4}  {'R_real':>8}  {'MDD(%)':>8}  {'Top-3 weights'}")
    print("-" * 65)

    pred_model.eval()

    # ── tqdm: 실제 데이터 143회 리밸런싱 대응 ──
    for i, (z_np, r_np) in enumerate(tqdm(rebal_samples, desc="Backtesting")):
        z      = torch.tensor(z_np[None], dtype=torch.float32)
        r_real = torch.tensor(r_np[None], dtype=torch.float32)

        with torch.no_grad():
            r_hat = pred_model(z)

        y_hat  = compute_cumulative_path(r_hat)
        x_star = solve_portfolio(y_hat.detach(), opt_layer, n1, C, x_min, x_max)
        y_real = compute_cumulative_path(r_real)
        w_real = compute_realized_path(x_star, y_real)[0].numpy()

        R_real = w_real[-1] / (d * C)

        window_pv   = 1.0 * (1 + w_real)
        running_max = np.maximum.accumulate(window_pv)
        M_real      = np.max((running_max - window_pv) / (running_max + 1e-10))

        w    = x_star[0].numpy()
        top3 = {names[j]: round(w[j], 3) for j in np.argsort(w)[-3:][::-1]}
        results.append({
            "window" : i + 1,
            "weights": w,
            "w_real" : w_real,
            "R_real" : R_real,
            "M_real" : M_real,
        })
        print(f"  {i+1:3d}  {R_real:8.4f}  {M_real:8.4%}  {top3}")

    R_list = [r["R_real"] for r in results]
    M_list = [r["M_real"] for r in results]

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