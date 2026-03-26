
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import yfinance as yf
import random
from dataclasses import dataclass
from typing import Dict, Tuple
from tqdm import tqdm
import itertools
import torch.nn as nn
import torch.optim as optim

# Step 1. Prediction Model: z -> r_hat
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
        out = self.net(z)               
        r_hat = out.view(batch, self.N, self.m)
        return r_hat



# Step 2. Cumulative Return Path: r_hat -> y_hat
def compute_cumulative_path(r: torch.Tensor) -> torch.Tensor:
  
    y = torch.cumsum(r, dim=1)  
    return y


# Step 3. Optimization Layer: y_hat -> x_star  
def build_optimization_layer(N: int, m: int) -> CvxpyLayer:
   
    # --- cvxpy 변수 ---
    x = cp.Variable(m, name="x")           
    u = cp.Variable(N + 1, name="u")       

    # --- cvxpy 파라미터 ---
    Y_hat = cp.Parameter((N, m), name="Y_hat")   
    n1C   = cp.Parameter(nonneg=True, name="n1C") 
    x_min = cp.Parameter(name="x_min")
    x_max = cp.Parameter(name="x_max")

    # --- 목적함수: y_hat(N)^T x 최대화 ---
    # Y_hat[N-1] == y_hat(t=N) (0-indexed)
    objective = cp.Maximize(Y_hat[N - 1] @ x)

    # --- 제약식 ---
    constraints = []
    constraints.append(u[0] == 0)

    for k in range(1, N + 1):
        y_k = Y_hat[k - 1]        

        constraints.append(u[k] - y_k @ x <= n1C)
        constraints.append(u[k] >= y_k @ x)
        constraints.append(u[k] >= u[k - 1])

    constraints.append(x >= x_min)
    constraints.append(x <= x_max)
    constraints.append(cp.sum(x) == 1)

    # --- 문제 정의 ---
    problem = cp.Problem(objective, constraints)
    assert problem.is_dcp(), "Problem is not DCP!"

    # --- CvxpyLayer 생성 ---
    layer = CvxpyLayer(
        problem,
        parameters=[Y_hat, n1C, x_min, x_max],
        variables=[x, u],
    )
    return layer


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
    x_min_val = torch.tensor(x_min, dtype=torch.float64)
    x_max_val = torch.tensor(x_max, dtype=torch.float64)

    x_stars = []
    for b in range(batch):
        Y_hat_b = y_hat[b].double()  # (N, m)
        x_star_b, _ = opt_layer(
            Y_hat_b,
            n1C_val,
            x_min_val,
            x_max_val,
            solver_args={"solve_method": "ECOS"},
        )
        x_stars.append(x_star_b.float())

    x_star = torch.stack(x_stars, dim=0)  # (batch, m)
    return x_star


# Step 4. Realized Portfolio Path: (x_star, y_real) -> w_real(t)
def compute_realized_path(
    x_star: torch.Tensor,
    y_real: torch.Tensor,
) -> torch.Tensor:

    w_real = torch.einsum("bj, btj -> bt", x_star, y_real)  # (batch, N)
    return w_real


# Step 5. Performance Metrics: w_real -> R_real, M_real
def compute_return(
    w_real: torch.Tensor,
    d: float,
    C: float,
) -> torch.Tensor:
    
    R_real = w_real[:, -1] / (d * C)
    return R_real


def compute_max_drawdown(w_real: torch.Tensor) -> torch.Tensor:
    
    running_max, _ = torch.cummax(w_real, dim=1)  
    drawdown = running_max - w_real               

    M_real = torch.max(drawdown, dim=1).values 
    # M_real = torch.logsumexp(beta * drawdown, dim=1) / beta   # (Gradient가 smooth 하지 않을 때 사용)
    return M_real



# def compute_max_drawdown_smoothing(w_real: torch.Tensor, beta: float = 100.0) -> torch.Tensor:
#     """
#     실현 MaxDD 계산 (differentiable 근사)

#     D_real(t) = max_{tau <= t} w_real(tau) - w_real(t)
#     M_real    = max_{1 <= t <= N} D_real(t)

#     Note:
#         torch.cummax는 미분 가능하지만 max는 argmax가 겹치면
#         gradient가 불안정할 수 있음.
#         DFL loss의 M_real 항은 gradient를 통해 theta를 업데이트하는 데 사용.

#     Args:
#         w_real : (batch, N)
#     Returns:
#         M_real : (batch,)
#     """
#     # running maximum up to each time step
#     running_max, _ = torch.cummax(w_real, dim=1)  # (batch, N)

#     # drawdown at each time step
#     drawdown = running_max - w_real                # (batch, N)

#     # maximum drawdown
#     # M_real = torch.max(drawdown, dim=1).values 
#     M_real = torch.logsumexp(beta * drawdown, dim=1) / beta   (Gradient가 smooth 하지 않을 때 사용)
#     return M_real



# DFL Loss Function
def dfl_loss(
    R_real: torch.Tensor,
    M_real: torch.Tensor,
    lam: float,
) -> torch.Tensor:
   
    loss = (-R_real + lam * M_real).mean()
    return loss


# Full Pipeline (End-to-End)
def forward_pass(
    z: torch.Tensor,
    r_real: torch.Tensor,
    pred_model: PredictionModel,
    opt_layer: CvxpyLayer,
    n1: float,
    C: float,
    d: float,
    x_min: float,
    x_max: float,
    lam: float,
) -> dict:

    # Step 1: z -> r_hat
    r_hat = pred_model(z)                           

    # Step 2: r_hat -> y_hat
    y_hat = compute_cumulative_path(r_hat)          

    # Step 3: y_hat -> x_star  (LP)
    x_star = solve_portfolio(
        y_hat, opt_layer, n1, C, x_min, x_max
    )                                              

    # Step 4: x_star + y_real -> w_real
    y_real = compute_cumulative_path(r_real)        
    w_real = compute_realized_path(x_star, y_real)  

    # Step 5: w_real -> R_real, M_real
    R_real = compute_return(w_real, d, C)          
    M_real = compute_max_drawdown(w_real)           

    # Step 6: Loss
    loss = dfl_loss(R_real, M_real, lam)            

    return {
        "r_hat" : r_hat,
        "y_hat" : y_hat,
        "x_star": x_star,
        "y_real": y_real,
        "w_real": w_real,
        "R_real": R_real,
        "M_real": M_real,
        "loss"  : loss,
    }


def plot_pnl(bt_results: list, horizon: int, figsize=(12, 6)):
 
    pv = [1.0]  
    rebal_indices = [0]

    for res in bt_results:
        w = res["w_real"]         
        base = pv[-1]
        pv.extend((base * (1 + w)).tolist())
        rebal_indices.append(len(pv) - 1)

    pv = np.array(pv)
    x  = np.arange(len(pv))

 
    running_max = np.maximum.accumulate(pv)
    drawdown    = (running_max - pv) / (running_max + 1e-10)

    total_ret = pv[-1] - 1.0
    max_dd    = drawdown.max()
    calmar    = total_ret / (max_dd + 1e-10)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True
    )

    # -- PnL --
    ax1.plot(x, pv, color="steelblue", linewidth=1.8, label="DFL Portfolio")
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    for idx in rebal_indices[1:-1]:   # skip start & end
        ax1.axvline(idx, color="orange", linestyle=":", linewidth=1.0,
                    alpha=0.7, label="Rebalance" if idx == rebal_indices[1] else "")
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.3f"))
    ax1.set_ylabel("Portfolio Value")
    ax1.set_title(
        f"Cumulative PnL  |  Return: {total_ret:.2%}  "
        f"Max DD: {max_dd:.2%}  Calmar: {calmar:.2f}"
    )
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.25)

    # -- Drawdown --
    ax2.fill_between(x, -drawdown * 100, 0, color="crimson", alpha=0.45, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trading Days (BT Period)")
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f%%"))
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.show()

    print(f"\n── PnL Summary ──")
    print(f"  Final Value  : {pv[-1]:.4f}")
    print(f"  Total Return : {total_ret:.4%}")
    print(f"  Max Drawdown : {max_dd:.4%}")
    print(f"  Calmar Ratio : {calmar:.4f}")
