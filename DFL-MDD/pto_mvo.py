import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cvxpy as cp

__all__ = ["train_pto_mvo", "backtest_pto_mvo"]

def train_pto_mvo(pred_model, is_samples, epochs, batch_size, lr):
    optimizer = optim.Adam(pred_model.parameters(), lr=lr)
    zs      = torch.tensor(np.array([s[0] for s in is_samples]), dtype=torch.float32)
    r_reals = torch.tensor(np.array([s[1] for s in is_samples]), dtype=torch.float32)
    print("\n── PTO-MVO Training (MSE) ──")
    pred_model.train()
    for epoch in range(epochs):
        perm    = torch.randperm(len(is_samples))
        ep_loss = []
        for i in range(0, len(is_samples), batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            r_hat = pred_model(zs[idx])
            loss  = F.mse_loss(r_hat, r_reals[idx])
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  mse = {np.mean(ep_loss):.6f}")
    return pred_model

def _solve_mvo(mu, Sigma, lam_mvo, x_min, x_max):
    m = len(mu)
    x = cp.Variable(m)
    objective   = cp.Maximize(mu @ x - (lam_mvo / 2) * cp.quad_form(x, Sigma))
    constraints = [cp.sum(x) == 1, x >= x_min, x <= x_max]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS, verbose=False)
    if x.value is None:
        return np.ones(m) / m
    return x.value

def backtest_pto_mvo(pred_model, rebal_samples, N, d, C,
                     lam_mvo=1.0, x_min=0.0, x_max=0.30, stock_names=None):
    m     = rebal_samples[0][1].shape[1]
    names = stock_names if stock_names else [f"S{j+1}" for j in range(m)]
    results = []

    print("\n── Backtest : PTO-MVO ──")
    print(f"{'Win':>4}  {'R_real':>8}  {'MDD(%)':>8}  {'Top-3 weights'}")
    print("-" * 55)

    pred_model.eval()
    for i, (z_np, r_np) in enumerate(rebal_samples):
        z = torch.tensor(z_np[None], dtype=torch.float32)
        with torch.no_grad():
            r_hat = pred_model(z)[0].numpy()

        mu    = r_hat.mean(axis=0)
        Sigma = np.cov(r_hat.T) + 1e-6 * np.eye(m)
        w     = _solve_mvo(mu, Sigma, lam_mvo, x_min, x_max)

        y_real = np.cumsum(r_np, axis=0)
        w_real = y_real @ w

        R_real = w_real[-1] / (d * C)

        # ── plot_pnl 방식과 동일: percentage MDD on portfolio value ──
        window_pv   = 1.0 * (1 + w_real)
        running_max = np.maximum.accumulate(window_pv)
        M_real      = np.max((running_max - window_pv) / (running_max + 1e-10))

        top3 = {names[j]: round(w[j], 3) for j in np.argsort(w)[-3:][::-1]}
        results.append({
            "window" : i + 1,
            "weights": w,
            "w_real" : w_real,
            "R_real" : R_real,
            "M_real" : M_real,
        })
        print(f"  {i+1:2d}  {R_real:8.4f}  {M_real:8.4%}  {top3}")

    R_list = [r["R_real"] for r in results]
    M_list = [r["M_real"] for r in results]

    print(f"\n── PTO-MVO Summary ──")
    print(f"  Avg Daily Return   : {np.mean(R_list):.4f}")
    print(f"  Avg MDD      : {np.mean(M_list):.4%}")
    print(f"  Max MDD      : {max(M_list):.4%}")

    return results
