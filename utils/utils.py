import math
import numpy as np

import scipy
from scipy.stats import qmc
from smt.sampling_methods import LHS

from sklearn.metrics import mean_squared_error

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import os

def train_joint(
    model, x_train, y_train,
    max_outer_iter=5000,
    max_inner_iter=10,
    tol_loss_change_pct=1e-8,
    verbose=False,
):
    """
    Train a JointModel using L-BFGS with outer loop and early stopping.

    Args:
        model: JointModel instance
        x_train: (N, D)
        y_train: (N, 1)
        max_outer_iter: number of outer optimization steps
        max_inner_iter: L-BFGS inner max_iter
        tol_loss_change_pct: relative loss change to early stop
        verbose: whether to print log

    Returns:
        loss_history: list of total loss per outer iteration
    """
    loss_history = []

    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.001,
        history_size=50,
        max_iter=max_inner_iter,
        line_search_fn="strong_wolfe"
    )

    for outer_epoch in range(max_outer_iter):
        def closure():
            optimizer.zero_grad()
            total_loss, ae_loss, gp_nll = model.loss(x_train, y_train)
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("  total_loss is NaN/Inf")
            if torch.isnan(ae_loss) or torch.isinf(ae_loss):
                print("  ae_loss is NaN/Inf")
            if torch.isnan(gp_nll) or torch.isinf(gp_nll):
                print("  gp_nll is NaN/Inf")
            total_loss.backward()
            if verbose:
                print(f"[Epoch {outer_epoch:03d}] Total: {total_loss.item():.5f} | AE: {ae_loss.item():.5f} | GP-NLL: {gp_nll.item():.5f}")
            return total_loss

        optimizer.step(closure)

        # Compute current loss for stopping check
        with torch.no_grad():
            current_loss, _, _ = model.loss(x_train, y_train)
            loss_history.append(current_loss.item())

        # Early stopping check
        if len(loss_history) >= 2:
            numer = abs(loss_history[-1] - loss_history[-2])
            denom = abs(loss_history[-1] - loss_history[0]) + 1e-8
            rel_change = numer / denom
            if rel_change < tol_loss_change_pct:
                if verbose:
                    print(f"[Stopping] Relative loss change {rel_change:.2e} < tol {tol_loss_change_pct}")
                break
    return loss_history

def alc_acquisition(model, x_train, y_train, x_cand, x_ref, k_variance=50):
    """
    Two-stage ALC acquisition:
    1. From all candidates, pick top-k_variance with highest posterior variance.
    2. Among these, compute exact ALC (Cohn style) and select best one.
    """
    model.gp.eval()
    model.ae.eval()

    N, Nc, Nr = x_train.shape[0], x_cand.shape[0], x_ref.shape[0]
    device = x_train.device

    with torch.no_grad():
        # Encode inputs
        z_train = model.ae.encode(x_train)
        z_cand = model.ae.encode(x_cand)
        z_ref = model.ae.encode(x_ref)

        # GP kernel and posterior variance
        K = model.gp.compute_K_eta(z_train)  # (N, N)
        L = torch.linalg.cholesky(K + 1e-8 * torch.eye(N, device=device))
        K_s = model.gp.kernel(z_train, z_cand)  # (N, Nc)
        v = torch.cholesky_solve(K_s, L)
        var_cand = model.gp.compute_K_eta(z_cand).diagonal(dim1=0, dim2=1) - (K_s * v).sum(dim=0)

        topk_var_idx = torch.topk(var_cand, k=min(k_variance, Nc)).indices

        # Compute tau² (MLE of variance)
        alpha = torch.cholesky_solve(y_train, L)
        tau2_hat = (y_train.T @ alpha / N).squeeze()

        # Base posterior covariance at z_ref
        K_s_ref = model.gp.kernel(z_train, z_ref)
        K_ss_ref = model.gp.kernel(z_ref,z_ref)
        v_ref = torch.cholesky_solve(K_s_ref, L)
        cov_orig = K_ss_ref - K_s_ref.T @ v_ref
        trace_orig = cov_orig.diag().sum()

        # Evaluate ALC among top candidates
        best_score = -float('inf')
        best_idx = -1

        for idx in topk_var_idx:
            x_aug = torch.vstack([x_train, x_cand[idx:idx+1]])
            z_aug = model.ae.encode(x_aug)

            K_aug = model.gp.compute_K_eta(z_aug)
            L_aug = torch.linalg.cholesky(K_aug + 1e-8 * torch.eye(K_aug.size(0), device=device))

            K_s_aug = model.gp.kernel(z_aug, z_ref)
            K_ss_aug = model.gp.kernel(z_ref,z_ref)
            v_aug = torch.cholesky_solve(K_s_aug, L_aug)
            cov_new = K_ss_aug - K_s_aug.T @ v_aug

            trace_new = cov_new.diag().sum()
            alc_score = tau2_hat * (trace_orig - trace_new)

            if alc_score > best_score:
                best_score = alc_score
                best_idx = idx.item()

    return best_idx, best_score

def run_active_learning(
    model, x_train_init, y_train_init,
    x_cand, y_cand,
    x_ref, x_test, y_test,
    T=100,
    save_plot_path=None,
    randomAC = False,
    topk=100,
    epsilon=0.02,
    delta=1e-4,
    patience=3
):
    x_train = x_train_init.clone()
    y_train = y_train_init.clone()

    rmse_list = []
    rrmse_list = []
    rmspesd_list = []
    T = min(T, len(y_cand))
    slow_count = 0

    for t in range(T):
        print(f"\n--- Round {t} ---")

        # Retrain model
        model.train()
        train_joint(model, x_train, y_train, max_outer_iter=10000, max_inner_iter=100, verbose=False)
        print(model.gp.get_hyperparams())

        # Evaluate on test
        model.eval()
        with torch.no_grad():
            z_train = model.ae.encode(x_train)
            z_test = model.ae.encode(x_test)
            y_pred, y_var = model.gp.predict(z_train, y_train, z_test)
            rmse = torch.sqrt(torch.mean((y_pred - y_test) ** 2)).item()
            rmspesd = rmspe_sd(y_pred, y_test)

            y_mean = y_test.mean()
            rrmse = torch.sqrt(torch.sum((y_pred - y_test) ** 2) / torch.sum((y_test - y_mean) ** 2)).item()

            print(f"Test RMSE: {rmse:.4f}")
            print(f"Test RRMSE: {rrmse:.4f}")
            print(f"Test rmsepd_sd: {rmspesd:.4f}")

            rmse_list.append(rmse)
            rrmse_list.append(rrmse)
            rmspesd_list.append(rmspesd)

        # Select new point
        if randomAC:
            best_idx = np.random.choice(len(y_cand), size=1, replace=False)
        else:
            best_idx, _ = alc_acquisition(model, x_train, y_train, x_cand, x_ref, k_variance=topk)
        if isinstance(best_idx, int):
            best_idx = [best_idx]

        x_new = x_cand[best_idx]
        y_new = y_cand[best_idx]
        x_train = torch.cat([x_train, x_new], dim=0)
        y_train = torch.cat([y_train, y_new], dim=0)

        mask = torch.ones(len(x_cand), dtype=torch.bool, device=x_cand.device)
        mask[best_idx] = False
        x_cand = x_cand[mask]
        y_cand = y_cand[mask]

    return {
        "rmse": np.array(rmse_list),
        "rrmse": np.array(rrmse_list),
        "rmspesd": np.array(rmspesd_list),
        "final_x_train": x_train,
        "final_y_train": y_train,
        "x_test": x_test
    }

def rmspe_sd(pred_y,true_y):
    N_test = true_y.shape[0]
    pred_y = pred_y.view((N_test,1))
    SSPE = torch.sum( (pred_y-true_y)**2)
    MSPE = SSPE/N_test
    RMSPE = torch.sqrt(MSPE)
    STD = torch.std(true_y)
    return (RMSPE/STD).item()