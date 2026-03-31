import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cvxpy as cp

__all__ = ["train_pto_mdd", "backtest_pto_mdd"]


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
            optimizer.zero_grad()
            r_hat = pred_model(zs[idx])
            loss  = F.mse_loss(r_hat, r_reals[idx])
            loss.backward()
            optimizer.step()
            ep_loss.append(loss.item())
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  mse = {np.mean(ep_loss):.6f}")
    return pred_model


def _solve_mdd_lp(Y_hat, N, m, n1, C, x_min, x_max, gamma=0.05):
    x = cp.Variable(m)
    u = cp.Variable(N + 1)
    # L2 regularization: DFL-MDD와 동일하게 분산 투자 유도
    objective   = cp.Maximize(Y_hat[N - 1] @ x - gamma * cp.sum_squares(x))
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
        # Fallback: x_max 클램핑 후 재정규화 (DFL-MDD fallback과 동일 방식)
        w = np.clip(np.ones(m) / m, x_min, x_max)
        return w / w.sum()
    return x.value


def backtest_pto_mdd(pred_model, rebal_samples, N, d, C,
                     n1=0.50, x_min=0.0, x_max=0.30, gamma=0.05,
                     stock_names=None):
    m     = rebal_samples[0][1].shape[1]
    names = stock_names if stock_names else [f"S{j+1}" for j in range(m)]
    results = []

    print("\n── Backtest : PTO-MDD ──")
    print(f"{'Win':>4}  {'R_real':>8}  {'MDD(%)':>8}  {'Top-3 weights'}")
    print("-" * 65)

    pred_model.eval()
    for i, (z_np, r_np) in enumerate(rebal_samples):
        z = torch.tensor(z_np[None], dtype=torch.float32)
        with torch.no_grad():
            r_hat = pred_model(z)[0].numpy()

        Y_hat  = np.cumsum(r_hat, axis=0)
        w      = _solve_mdd_lp(Y_hat, N, m, n1, C, x_min, x_max, gamma)
        y_real = np.cumsum(r_np, axis=0)
        w_real = y_real @ w                             # (N,)

        R_real = w_real[-1] / (d * C)

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
        print(f"  {i+1:3d}  {R_real:8.4f}  {M_real:8.4%}  {top3}")

    R_list = [r["R_real"] for r in results]
    M_list = [r["M_real"] for r in results]

    print(f"\n── PTO-MDD Summary ──")
    print(f"  Avg Daily Return : {np.mean(R_list):.4f}")
    print(f"  Avg MDD          : {np.mean(M_list):.4%}")
    print(f"  Max MDD          : {max(M_list):.4%}")

    return results
