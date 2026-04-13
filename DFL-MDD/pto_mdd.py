import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cvxpy as cp

__all__ = ["train_pto_mdd", "backtest_pto_mdd"]


def train_pto_mdd(pred_model, train_samples, val_samples=None,
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

    print("\n── PTO-MDD Training (MSE + Val Early Stopping) ──")
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


def _solve_mdd_lp(Y_hat, N, m, n1, C, x_min, x_max, gamma=0.0,
                  Sigma=None, delta=0.0):
    x = cp.Variable(m)
    u = cp.Variable(N + 1)

    # 목적함수: 예측 수익률 - risk term - L2 정규화
    risk_term = (delta / 2) * cp.quad_form(x, Sigma) if (Sigma is not None and delta > 0) else 0
    objective = cp.Maximize(Y_hat[N - 1] @ x - risk_term - gamma * cp.sum_squares(x))

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
        # Fallback: x_max 클램핑 후 재정규화
        w = np.clip(np.ones(m) / m, x_min, x_max)
        return w / w.sum()
    return x.value


def backtest_pto_mdd(pred_model, rebal_samples, N, d, C,
                     n1=0.50, x_min=0.0, x_max=1.0, gamma=0.0,
                     delta=0.0, is_mean=None, is_std=None,
                     stock_names=None):
    m        = rebal_samples[0][1].shape[1]
    lookback = rebal_samples[0][0].shape[0] // m
    names    = stock_names if stock_names else [f"S{j+1}" for j in range(m)]
    results  = []
    cum_pv   = [1.0]   # 누적 portfolio value (이미지 MDD와 동일 기준)

    print("\n── Backtest : PTO-MDD ──")
    print(f"{'Win':>4}  {'R_real':>8}  {'MDD(%)':>8}  {'Top-3 weights'}")
    print("-" * 65)

    pred_model.eval()
    for i, (z_np, r_np) in enumerate(rebal_samples):
        z = torch.tensor(z_np[None], dtype=torch.float32)
        with torch.no_grad():
            r_hat = pred_model(z)[0].numpy()

        # Sigma: lookback 실제 수익률 기반 (delta > 0일 때만 사용)
        Sigma = None
        if delta > 0 and is_mean is not None and is_std is not None:
            z_raw = z_np.reshape(lookback, m) * is_std + is_mean  # 역정규화
            Sigma = np.cov(z_raw.T) + 1e-4 * np.eye(m)

        Y_hat  = np.cumsum(r_hat, axis=0)
        w      = _solve_mdd_lp(Y_hat, N, m, n1, C, x_min, x_max, gamma,
                               Sigma=Sigma, delta=delta)
        y_real = np.cumsum(r_np, axis=0)
        w_real = y_real @ w                             # (N,)

        R_real = w_real[-1] / (d * C)

        base    = cum_pv[-1]
        cum_pv.extend((base * (1 + w_real)).tolist())
        pv_arr  = np.array(cum_pv)
        run_max = np.maximum.accumulate(pv_arr)
        M_real  = np.max((run_max - pv_arr) / (run_max + 1e-10))

        top3 = {names[j]: round(w[j], 3) for j in np.argsort(w)[-3:][::-1]}
        results.append({
            "window" : i + 1,
            "weights": w,
            "w_real" : w_real,
            "R_real" : R_real,
            "M_real" : M_real,
        })
        print(f"  {i+1:3d}  {R_real:8.4f}  {M_real:8.4%}  {top3}")

    return results
