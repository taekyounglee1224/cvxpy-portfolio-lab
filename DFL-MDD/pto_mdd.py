
# pto_mdd.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


__all__ = ["train_pto_mdd", "backtest_pto_mdd"]

# =============================================================================
# Train : MSE Loss
# =============================================================================
def train_pto_mdd(pred_model, is_samples, epochs, batch_size, lr):

    optimizer = optim.Adam(pred_model.parameters(), lr=lr)
    zs      = torch.tensor(np.array([s[0] for s in is_samples]), dtype=torch.float32)
    r_reals = torch.tensor(np.array([s[1] for s in is_samples]), dtype=torch.float32)

    print("\n── PTO-MDD Training (MSE) ──")
    pred_model.train()

    for epoch in range(epochs):
        perm    = torch.randperm(len(is_samples))
        ep_loss = []

        for i in range(0, len(is_samples), batch_size):
            idx = perm[i : i + batch_size]
            z_b = zs[idx]
            r_b = r_reals[idx]

            optimizer.zero_grad()
            r_hat = pred_model(z_b)
            loss  = F.mse_loss(r_hat, r_b)
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  mse = {np.mean(ep_loss):.6f}")

    return pred_model

# =============================================================================
# MDD LP Solver
# =============================================================================
def _solve_mdd_lp(Y_hat, N, m, n1, C, x_min, x_max):

    x = cp.Variable(m)
    u = cp.Variable(N + 1)

    objective   = cp.Maximize(Y_hat[N - 1] @ x)
    constraints = [u[0] == 0]

    for k in range(1, N + 1):
        y_k = Y_hat[k - 1]
        constraints.append(u[k] - y_k @ x <= n1 * C)
        constraints.append(u[k] >= y_k @ x)
        constraints.append(u[k] >= u[k - 1])

    constraints += [x >= x_min, x <= x_max, cp.sum(x) == 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)

    if x.value is None:
        return np.ones(m) / m
    return x.value

# =============================================================================
# Backtest
# =============================================================================
def backtest_pto_mdd(pred_model, rebal_samples, N, d, C,
                     n1=0.10, x_min=0.0, x_max=0.30, stock_names=None):

    m     = rebal_samples[0][1].shape[1]
    names = stock_names if stock_names else [f"S{j+1}" for j in range(m)]
    results = []

    print("\n── Backtest : PTO-MDD ──")
    print(f"{'Win':>4}  {'R_real':>8}  {'MDD':>8}  {'Top-3 weights'}")
    print("-" * 55)

    pred_model.eval()

    for i, (z_np, r_np) in enumerate(rebal_samples):
        z = torch.tensor(z_np[None], dtype=torch.float32)

        with torch.no_grad():
            r_hat = pred_model(z)[0].numpy()

        Y_hat  = np.cumsum(r_hat, axis=0)
        w      = _solve_mdd_lp(Y_hat, N, m, n1, C, x_min, x_max)

        y_real = np.cumsum(r_np, axis=0)
        w_real = y_real @ w

        R_real      = w_real[-1] / (d * C)
        running_max = np.maximum.accumulate(w_real)
        M_real      = np.max(running_max - w_real)

        top3 = {names[j]: round(w[j], 3) for j in np.argsort(w)[-3:][::-1]}
        results.append({
            "window" : i + 1,
            "weights": w,
            "w_real" : w_real,
            "R_real" : R_real,
            "M_real" : M_real,
        })
        print(f"  {i+1:2d}  {R_real:8.4f}  {M_real:8.4f}  {top3}")

    R_list = [r["R_real"] for r in results]
    M_list = [r["M_real"] for r in results]
    calmar = sum(R_list) / (max(M_list) + 1e-10)

    print(f"\n── PTO-MDD Summary ──")
    print(f"  Avg Return   : {np.mean(R_list):.4f}")
    print(f"  Total Return : {sum(R_list):.4f}")
    print(f"  Avg MDD      : {np.mean(M_list):.4f}")
    print(f"  Max MDD      : {max(M_list):.4f}")
    print(f"  Calmar Ratio : {calmar:.4f}")

    return results


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
    ax1.plot(x, pv, color="steelblue", linewidth=1.8, label="PTO-MDD Portfolio")
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

