
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import random
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple
from tqdm.auto import tqdm
import itertools
import torch.nn as nn
import torch.optim as optim


__all__ = [
    "PredictionModel",
    "build_optimization_layer",
    "solve_portfolio",
    "min_achievable_dd",
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
        # cp.sum_squares(L_p.T @ x) is DPP-compliant: the parameter appears linearly, once
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
    Sigma_list=None,          # list of (m,m) torch.Tensor, only used when delta>0
    infeas_counter=None,      # mutable list, appended to on solver failure (reviewer comment #4)
    solve_method="ECOS",      # diffcp forward solver: "ECOS" | "SCS" | "CLARABEL"
) -> torch.Tensor:
    batch, N, m = y_hat.shape
    n1C_val   = torch.tensor(n1 * C, dtype=torch.float64)
    x_min_val = torch.tensor(x_min,  dtype=torch.float64)
    x_max_val = torch.tensor(x_max,  dtype=torch.float64)

    x_stars = []
    for b in range(batch):
        try:
            if Sigma_list is not None:
                # Cholesky factorisation: Sigma = L L^T, L lower-triangular
                L_b = torch.linalg.cholesky(Sigma_list[b].double())
                x_star_b, _ = opt_layer(
                    y_hat[b].double(), n1C_val, x_min_val, x_max_val,
                    L_b,
                    solver_args={"solve_method": solve_method},
                )
            else:
                x_star_b, _ = opt_layer(
                    y_hat[b].double(), n1C_val, x_min_val, x_max_val,
                    solver_args={"solve_method": solve_method},
                )
            # Catch a solution that is invalid even though no exception was raised.
            # On numerical failure CLARABEL returns x=0, violating sum(x)==1,
            # rather than raising.
            s = float(x_star_b.detach().sum())
            fail_reason = None if abs(s - 1.0) <= 1e-4 else f"invalid_sum={s:.2e}"
        except Exception as e:
            fail_reason = str(e)[:120] or "solve_failed"

        if fail_reason is not None:
            if infeas_counter is not None:
                infeas_counter.append(fail_reason)
            x_raw     = torch.softmax(y_hat[b, -1, :], dim=0)
            x_clamped = torch.clamp(x_raw, min=x_min, max=x_max)
            x_star_b  = (x_clamped / x_clamped.sum()).double()
        x_stars.append(x_star_b.float())

    return torch.stack(x_stars, dim=0)


def min_achievable_dd(y_hat_np, m, C, x_min, x_max):
    """
    Compute the smallest achievable drawdown n1* for a predicted cumulative path.

        min  n1   s.t.  u_0 = 0
                        u_k - y_k'x <= n1*C,  u_k >= y_k'x,  u_k >= u_{k-1}
                        x_min <= x <= x_max,  sum(x) = 1

    This is a pure feasibility problem with the return and risk objective removed,
    so n1* greater than the configured n1 means the window is genuinely infeasible
    rather than a numerical failure.

    Returns
    -------
    (n1_star, status) : n1_star is nan on failure
    """
    N = y_hat_np.shape[0]
    n1v = cp.Variable(nonneg=True)
    x   = cp.Variable(m)
    u   = cp.Variable(N + 1)
    cons = [u[0] == 0]
    for k in range(1, N + 1):
        yk = y_hat_np[k - 1]
        cons += [u[k] - yk @ x <= n1v * C,
                 u[k] >= yk @ x,
                 u[k] >= u[k - 1]]
    cons += [x >= x_min, x <= x_max, cp.sum(x) == 1]
    prob = cp.Problem(cp.Minimize(n1v), cons)
    try:
        prob.solve(solver=cp.CLARABEL)
        val = float(n1v.value) if n1v.value is not None else float("nan")
        return val, prob.status
    except Exception as e:
        return float("nan"), f"err:{str(e)[:40]}"




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

    x_star     : (batch, m)       -- portfolio weights
    r_real     : (batch, N, m)    -- per-period asset returns
    Sigma_list : list of (m, m) float64 tensors estimated from lookback window.
                 When provided, portfolio variance = x^T sum x.
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
                 is_mean=None, is_std=None, delta=0.0, solve_method="ECOS"):
    r_hat  = pred_model(z)
    y_hat  = compute_cumulative_path(r_hat)

    # Sigma is always estimated when is_mean/is_std are given; it feeds both the
    # Sharpe ratio and the delta risk term
    Sigma_list = None
    if is_mean is not None and is_std is not None:
        batch     = z.shape[0]
        m_dim     = r_hat.shape[2]
        lb        = z.shape[1] // m_dim
        is_mean_t = torch.tensor(is_mean, dtype=torch.float32)
        is_std_t  = torch.tensor(is_std,  dtype=torch.float32)
        z_raw     = z.reshape(batch, lb, m_dim) * is_std_t + is_mean_t   # undo standardisation
        Sigma_list = []
        for b in range(batch):
            z_b = z_raw[b].detach().numpy()
            S   = np.cov(z_b.T) + 1e-4 * np.eye(m_dim)
            Sigma_list.append(torch.tensor(S, dtype=torch.float64))

    # Sigma is only passed to solve_portfolio when delta>0, for the objective
    x_star = solve_portfolio(y_hat, opt_layer, n1, C, x_min, x_max,
                             Sigma_list if delta > 0 else None,
                             solve_method=solve_method)
    y_real = compute_cumulative_path(r_real)
    w_real = compute_realized_path(x_star, y_real)
    R_real = compute_return(w_real, d, C)
    M_real = compute_max_drawdown(w_real)
    Sharpe = compute_sharpe(x_star, r_real, Sigma_list)   # uses the lookback covariance
    loss   = dfl_loss(Sharpe, M_real, lam)
    return {"r_hat": r_hat, "y_hat": y_hat, "x_star": x_star,
            "y_real": y_real, "w_real": w_real,
            "R_real": R_real, "M_real": M_real, "Sharpe": Sharpe, "loss": loss}

def _stack_f32(samples, field):
    """Stack a sample set into an (n, ...) float32 tensor.

    When the samples hold contiguous float32 arrays (.Z / .R), as run_dfl_mdd.WindowSet
    does, they are wrapped with torch.from_numpy without copying, which cuts memory
    use to about a third. Anything else (a plain list, say) is stacked and cast as
    before. Both paths produce the same values.

    from_numpy shares memory, so the returned tensor must not be modified in place.
    The training loop only uses advanced indexing such as zs_tr[idx], which copies.
    """
    arr = getattr(samples, ("Z", "R")[field], None)
    if arr is not None and arr.dtype == np.float32:
        return torch.from_numpy(arr)
    return torch.tensor(np.array([s[field] for s in samples]), dtype=torch.float32)


# =============================================================================
# Train (DFL-MDD) -- Val Early Stopping
# =============================================================================
def train_dfl_mdd(pred_model, opt_layer, train_samples, val_samples=None,
                  epochs=50, batch_size=16, lr=1e-4,
                  n1=0.10, C=1.0, d=1.0, x_min=0.0, x_max=0.30, lam=0.3,
                  is_mean=None, is_std=None, delta=0.0,
                  patience=10, lr_patience=10, lr_factor=0.5,
                  train_dates=None, solve_method="ECOS"):
    """
    Train DFL-MDD.

    When val_samples is given, the validation loss is computed every epoch and used
    for early stopping. Subsampling val_samples at the rebalancing interval is
    recommended for speed.

    lr_patience : ReduceLROnPlateau patience; the learning rate drops when the
                  validation loss stops improving
    lr_factor   : multiplicative factor for that drop (default 0.5, i.e. halved)
    """
    optimizer = optim.Adam(pred_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience
    )

    zs_tr = _stack_f32(train_samples, 0)
    rs_tr = _stack_f32(train_samples, 1)

    if val_samples is not None:
        zs_val = _stack_f32(val_samples, 0)
        rs_val = _stack_f32(val_samples, 1)

    best_val_loss    = float("inf")
    best_state       = None
    no_improve       = 0
    inaccurate_log   = []   # records {"epoch", "batch", "n_inaccurate"}

    print("\n-- DFL-MDD Training (with Val Early Stopping + LR Scheduler) --")

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
                result = forward_pass(
                    z_b, r_b, pred_model, opt_layer,
                    n1, C, d, x_min, x_max, lam,
                    is_mean=is_mean, is_std=is_std, delta=delta,
                    solve_method=solve_method,
                )
                n_inaccurate = sum(
                    1 for warning in w if "Inaccurate" in str(warning.message)
                )

            if n_inaccurate > 0:
                entry = {
                    "epoch"       : epoch + 1,
                    "batch"       : i // batch_size,
                    "n_inaccurate": n_inaccurate,
                }
                if train_dates is not None:
                    batch_dates   = [train_dates[j.item()] for j in idx]
                    entry["date_start"] = min(d[0] for d in batch_dates)
                    entry["date_end"]   = max(d[1] for d in batch_dates)
                inaccurate_log.append(entry)
                continue   # gradient update skip

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
                        solve_method=solve_method,
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
                inaccurate_str = f"  [inaccurate={len(inaccurate_log)}]" if inaccurate_log else ""
                print(f"  Epoch {epoch+1:3d}/{epochs}  train={tr_loss:.6f}  val={val_loss:.6f}  lr={current_lr:.2e}  {marker}{inaccurate_str}")

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}  (best val={best_val_loss:.6f})")
                break
        else:
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs}  loss = {tr_loss:.6f}")

    if best_state is not None:
        pred_model.load_state_dict(best_state)

    if inaccurate_log:
        print(f"\n  inaccurate solves: {len(inaccurate_log)} in total")
        for ev in inaccurate_log:
            date_str = f"  [{ev['date_start']} ~ {ev['date_end']}]" if "date_start" in ev else ""
            print(f"    epoch={ev['epoch']:3d}, batch={ev['batch']:3d}, count={ev['n_inaccurate']}{date_str}")
    else:
        print("\n  no inaccurate solves")

    return pred_model, inaccurate_log


# =============================================================================
# Backtest
# =============================================================================
def backtest_dfl_mdd(pred_model, opt_layer, rebal_samples, N, d, C,
                     n1=0.10, x_min=0.0, x_max=0.30,
                     delta=0.0, is_mean=None, is_std=None,
                     stock_names=None, rebal=None, solve_method="ECOS"):
    m        = rebal_samples[0][1].shape[1]
    lookback = rebal_samples[0][0].shape[0] // m
    names    = stock_names if stock_names else [f"S{j+1}" for j in range(m)]
    results  = []
    cum_pv   = [1.0]   # cumulative portfolio value (same basis as the reported MDD)

    print("\n-- Backtest : DFL-MDD --")
    print(f"{'Win':>4}  {'R_real':>8}  {'Sharpe':>8}  {'MDD(%)':>8}  {'Top-3 weights'}")
    print("-" * 75)

    pred_model.eval()
    bt_inaccurate_log = []   # {"window", "n_inaccurate"}
    infeas_log        = []   # records solver failures, i.e. infeasible fallbacks (reviewer comment #4)
    failed_windows    = []   # indices of failed windows (0-based)
    min_n1_log        = []   # smallest achievable drawdown per failed window

    for i, (z_np, r_np) in enumerate(tqdm(rebal_samples, desc="Backtesting")):
        z      = torch.tensor(z_np[None], dtype=torch.float32)
        r_real = torch.tensor(r_np[None], dtype=torch.float32)

        with torch.no_grad():
            r_hat = pred_model(z)

        # Sigma is always estimated when is_mean/is_std are given
        Sigma_list = None
        if is_mean is not None and is_std is not None:
            z_raw      = z_np.reshape(lookback, m) * is_std + is_mean   # undo standardisation
            S          = np.cov(z_raw.T) + 1e-4 * np.eye(m)
            Sigma_list = [torch.tensor(S, dtype=torch.float64)]

        y_hat = compute_cumulative_path(r_hat)
        infeas_before = len(infeas_log)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            x_star = solve_portfolio(y_hat.detach(), opt_layer, n1, C, x_min, x_max,
                                     Sigma_list if delta > 0 else None,
                                     infeas_counter=infeas_log,
                                     solve_method=solve_method)
            n_inaccurate = sum(
                1 for warning in w if "Inaccurate" in str(warning.message)
            )
        window_failed = len(infeas_log) > infeas_before
        if window_failed:
            failed_windows.append(i)      # record the failed window index

        # Smallest achievable drawdown n1*, computed for every window.
        #   failed windows    : tells a numerical failure from a true infeasibility
        #   successful windows: quantifies the slack against the nominal n1
        mdd_min, st = min_achievable_dd(
            y_hat[0].detach().double().numpy(), m, C, x_min, x_max)
        min_n1_log.append({
            "window": i, "min_n1": mdd_min, "status": st,
            "failed": window_failed,
            "true_infeasible": ((mdd_min > n1) if mdd_min == mdd_min else None)
                               if window_failed else False,
        })
        if n_inaccurate > 0:
            bt_inaccurate_log.append({"window": i + 1, "n_inaccurate": n_inaccurate})
        y_real = compute_cumulative_path(r_real)
        w_real = compute_realized_path(x_star, y_real)[0].numpy()

        # when HORIZON > REBAL, truncate to the actual holding period
        if rebal is not None:
            w_real  = w_real[:rebal]
            r_real  = r_real[:, :rebal, :]

        R_real = w_real[-1] / (d * C)

        base    = cum_pv[-1]
        cum_pv.extend((base * (1 + w_real)).tolist())

        # for logging: per-window MDD inside this window
        pv_w    = 1 + w_real
        rmax_w  = np.maximum.accumulate(pv_w)
        M_real  = np.max((rmax_w - pv_w) / (rmax_w + 1e-10))

        # Sharpe: portfolio variance from the lookback covariance
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
        print(f"\n  inaccurate solves during backtest: {len(bt_inaccurate_log)} in total")
        for ev in bt_inaccurate_log:
            print(f"    window={ev['window']:3d}, count={ev['n_inaccurate']}")
    else:
        print("\n  no inaccurate solves during backtest")

    # -- infeasibility rate (Reviewer #4) --
    n_win  = len(results)
    n_inf  = len(infeas_log)
    infeas_summary = {
        "n_infeasible":   n_inf,
        "n_windows":      n_win,
        "rate":           (n_inf / n_win) if n_win else float("nan"),
        "failed_windows": list(failed_windows),   # indices of failed windows
        "min_n1_log":     list(min_n1_log),       # breakdown of the failure cause
        "n_true_infeas":  sum(1 for e in min_n1_log
                              if e["failed"] and e["true_infeasible"] is True),
        "n_numerical":    sum(1 for e in min_n1_log
                              if e["failed"] and e["true_infeasible"] is False),
        "solve_method":   solve_method,
    }
    if n_inf > 0:
        print(f"\n  fallbacks: {n_inf}/{n_win} ({infeas_summary['rate']:.2%})")
        print(f"      truly infeasible {infeas_summary['n_true_infeas']} / "
              f"numerical failures {infeas_summary['n_numerical']}")
        need = [e["min_n1"] for e in min_n1_log
                if e["failed"] and e["min_n1"] == e["min_n1"]]
        if need:
            print(f"      smallest feasible n1: mean {np.mean(need):.3f}, "
                  f"max {np.max(need):.3f}  (configured {n1})")
    else:
        print("\n  no fallbacks (0%)")

    return results, bt_inaccurate_log, infeas_summary

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

    print(f"\n-- PnL Summary ({label}) --")
    print(f"  Final Value  : {pv[-1]:.4f}")
    print(f"  Total Return : {total_ret:.4%}")
    print(f"  Max Drawdown : {max_dd:.4%}")
    print(f"  Calmar Ratio : {calmar:.4f}")