from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .config import (
    ExperimentConfig,
    LoggingConfig,
    MetricsConfig,
    NetworkConfig,
    PenaltyConfig,
    SolverConfig,
)
from .metrics import calculate_metrics
from .networks import PolicyNet, ValueNet
from .terminalPenalties import TerminalPenaltyBundle, build_terminal_penalty
from .train_utils import train_value_global

DTYPE = np.float64


# -------------------------------
# L1-sparse control
# -------------------------------
def soft_threshold_box(x: np.ndarray, kappa: float, u_max: float) -> np.ndarray:
    """Soft-thresholding with box constraint, coordinate-wise."""
    y = np.sign(x) * np.maximum(np.abs(x) - kappa, 0.0)
    return np.clip(y, -u_max, u_max)


def optimal_u_and_driver(
    Z: np.ndarray,
    eps: float,
    lam: float,
    u_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given Z = Σ^T ∇V with Σ = sqrt(ε) I, solve the ℓ1-regularized optimal control.
    """
    arg = -math.sqrt(eps) * Z
    u_star = soft_threshold_box(arg, eps * lam, u_max)

    quad = (0.5 / eps) * np.sum(u_star * u_star, axis=1)
    l1 = lam * np.sum(np.abs(u_star), axis=1)
    cross = (1.0 / math.sqrt(eps)) * np.sum(Z * u_star, axis=1)
    h = quad + l1 + cross

    u_star = np.nan_to_num(u_star, nan=0.0, posinf=u_max, neginf=-u_max)
    h = np.nan_to_num(h, nan=1e6, posinf=1e6, neginf=1e6)
    return u_star, h


# -------------------------------
# Helpers: evaluate V and grad_x V via autodiff
# -------------------------------
@torch.no_grad()
def eval_value(
    v_net: nn.Module,
    t_scalar: float,
    x_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Evaluate V_phi(t, x) on a batch of numpy states, return np.array (N,).
    """
    x = torch.from_numpy(x_np.astype(np.float32)).to(device)
    t = torch.tensor(t_scalar, dtype=torch.float32, device=device)
    v = v_net(t, x)
    return v.cpu().numpy()


def eval_grad(
    v_net: nn.Module,
    t_scalar: float,
    x_np: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Evaluate ∇_x V_phi(t, x) on a batch, but *without* multiplying by sqrt(ε).
    """
    x = torch.from_numpy(x_np.astype(np.float32)).to(device)
    x.requires_grad_(True)
    t = torch.tensor(t_scalar, dtype=torch.float32, device=device)
    v = v_net(t, x)  # (N,)
    ones = torch.ones_like(v)
    (g_x,) = torch.autograd.grad(v, x, grad_outputs=ones, create_graph=False)
    return g_x.detach().cpu().numpy()


class RegularizedSBSolver:
    """
    Organized implementation of toyExample_dGauss.ipynb (cells 1-295) with config support.
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(cfg.solver.seed)
        self.dt = 1.0 / cfg.solver.T
        self.target_mean = self._default_target_mean()
        self.penalty_bundle = self._build_penalty_bundle()
        self.target_mean = self.penalty_bundle.metadata.get("target_mean", self.target_mean)
        self.value_net = self._instantiate_value_net(cfg.value_net)
        self.policy_net = self._instantiate_policy_net(cfg.policy_net)
        self.v_opt = torch.optim.Adam(self.value_net.parameters(), lr=cfg.value_net.lr)
        self.u_opt = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.policy_net.lr)
        self.X0 = self._sample_source()

    # ---- setup helpers ----
    def _default_target_mean(self) -> np.ndarray:
        s = self.cfg.solver
        if s.target_type == "full":
            target_mean = np.full(s.d, s.target_shift, dtype=DTYPE)
        elif s.target_type == "sparse":
            target_mean = np.zeros(s.d, dtype=DTYPE)
            target_mean[0] = s.target_shift
        else:
            raise ValueError(f"Unknown target_type '{s.target_type}'")
        return target_mean

    def _build_penalty_bundle(self) -> TerminalPenaltyBundle:
        penalty_cfg: PenaltyConfig = self.cfg.penalty
        return build_terminal_penalty(
            penalty_cfg.name,
            target_mean=self.target_mean,
            params=penalty_cfg.params,
        )

    def _instantiate_value_net(self, net_cfg: NetworkConfig) -> nn.Module:
        name = net_cfg.name.lower()
        d = self.cfg.solver.d
        if name in ("valuenet", "value"):
            return ValueNet(d=d, hidden=net_cfg.hidden).to(self.device)
        raise ValueError(f"Unknown value network '{net_cfg.name}'")

    def _instantiate_policy_net(self, net_cfg: NetworkConfig) -> nn.Module:
        name = net_cfg.name.lower()
        d = self.cfg.solver.d
        if name in ("policynet", "policy"):
            return PolicyNet(d=d, hidden=net_cfg.hidden).to(self.device)
        raise ValueError(f"Unknown policy network '{net_cfg.name}'")

    def _sample_source(self) -> np.ndarray:
        s = self.cfg.solver
        return self.rng.normal(loc=0.0, scale=1.0, size=(s.n_particles, s.d)).astype(DTYPE)

    # ---- main entry point ----
    def run(self) -> Dict[str, Any]:
        s = self.cfg.solver
        dt = self.dt

        for it in range(s.outer_loops):
            X = np.zeros((s.T + 1, s.n_particles, s.d), dtype=DTYPE)
            X[0] = self.X0

            self.policy_net.eval()
            with torch.no_grad():
                for i in range(s.T):
                    t_i = i / float(s.T)
                    x_i = torch.from_numpy(X[i].astype(np.float32)).to(self.device)
                    t_t = torch.tensor(t_i, dtype=torch.float32, device=self.device)
                    u_nom = self.policy_net(t_t, x_i).cpu().numpy()
                    u_nom = np.clip(u_nom, -s.u_max, s.u_max)
                    dW = self.rng.normal(size=(s.n_particles, s.d)).astype(DTYPE) * math.sqrt(dt)
                    X[i + 1] = X[i] + u_nom * dt + math.sqrt(s.eps) * dW

            XT = X[s.T]
            g_vec, grad_g = self.penalty_bundle.evaluate(XT)

            Y = np.zeros((s.T + 1, s.n_particles), dtype=DTYPE)
            Z = np.zeros((s.T + 1, s.n_particles, s.d), dtype=DTYPE)
            Y[s.T] = (s.beta_eff * g_vec).astype(DTYPE)
            Z[s.T] = math.sqrt(s.eps) * (s.beta_eff * grad_g)
            Z[s.T] = np.clip(np.nan_to_num(Z[s.T], nan=0.0, posinf=1e3, neginf=-1e3), -1e3, 1e3)

            self._fit_terminal_boundary(XT, Y[s.T])

            tx_list = []
            Ty_list = []
            self.value_net.train()

            for i in reversed(range(s.T)):
                t_ip1 = (i + 1) / float(s.T)
                if i == s.T - 1:
                    Y_ip1 = Y[s.T]
                    Z_ip1 = Z[s.T]
                else:
                    X_ip1 = X[i + 1].astype(np.float32)
                    Y_ip1 = eval_value(self.value_net, t_ip1, X_ip1, self.device)
                    grad_ip1 = eval_grad(self.value_net, t_ip1, X_ip1, self.device)
                    Z_ip1 = math.sqrt(s.eps) * grad_ip1
                    Z_ip1 = np.clip(np.nan_to_num(Z_ip1, nan=0.0, posinf=1e3, neginf=-1e3), -1e3, 1e3)

                u_star_ip1, h_ip1 = optimal_u_and_driver(Z_ip1, s.eps, s.lam, s.u_max)

                X_ip1 = X[i + 1].astype(np.float32)
                with torch.no_grad():
                    x_ip1_torch = torch.from_numpy(X_ip1).to(self.device)
                    t_ip1_torch = torch.full(
                        (s.n_particles,),
                        fill_value=t_ip1,
                        dtype=torch.float32,
                        device=self.device,
                    )
                    u_nom_p1 = self.policy_net(t_ip1_torch, x_ip1_torch).cpu().numpy()
                u_nom_p1 = np.clip(u_nom_p1, -s.u_max, s.u_max)
                K_p1 = u_nom_p1 / math.sqrt(s.eps)
                h_tilde_ip1 = h_ip1 - np.sum(Z_ip1 * K_p1, axis=1)

                T_i = Y_ip1 + h_tilde_ip1 * dt
                t_i_norm = i / float(s.T)
                x_i_np = X[i].astype(np.float32)
                t_col = np.full((s.n_particles, 1), t_i_norm, dtype=np.float32)
                tx_i = np.concatenate([t_col, x_i_np], axis=1)
                tx_list.append(tx_i)
                Ty_list.append(T_i.astype(np.float32))

            TX = np.concatenate(tx_list, axis=0)
            TY = np.concatenate(Ty_list, axis=0)
            train_value_global(
                self.value_net,
                self.v_opt,
                TX,
                TY,
                device=self.device,
                epochs=s.value_epochs,
                batch_size=4096,
            )

            self.value_net.eval()
            for i in range(s.T + 1):
                t_i_norm = i / float(s.T)
                x_i_np = X[i].astype(np.float32)
                Y[i] = eval_value(self.value_net, t_i_norm, x_i_np, self.device).astype(DTYPE)
                if i < s.T:
                    grad_i = eval_grad(self.value_net, t_i_norm, x_i_np, self.device).astype(DTYPE)
                    Z[i] = math.sqrt(s.eps) * grad_i
                    Z[i] = np.clip(np.nan_to_num(Z[i], nan=0.0, posinf=1e3, neginf=-1e3), -1e3, 1e3)

            self._policy_update(X, Z)

            if (it + 1) % 20 == 0 or it == 0:
                XT_mean_vec = XT.mean(axis=0)
                XT_std_vec = XT.std(axis=0)
                print(
                    f"[outer {it+1}/{s.outer_loops}] "
                    f"X_T mean={XT_mean_vec} (target {self.target_mean}), "
                    f"std={XT_std_vec}, mean terminal g={g_vec.mean():.3f}"
                )

            if self.cfg.logging.save_every_outer and (it + 1) % self.cfg.logging.save_every_outer == 0:
                payload = {"trajectories": X.astype(np.float32)} if self.cfg.logging.save_samples else None
                self._save_checkpoint(f"outer_{it+1:04d}", payload)

        result = self._final_evaluation()
        if self.cfg.metrics.enabled:
            target_samples = self.penalty_bundle.metadata.get("target_samples")
            metrics_cfg: MetricsConfig = self.cfg.metrics
            metrics, _, _ = calculate_metrics(
                self.X0,
                target_samples,
                self.policy_net,
                s.T,
                eps=s.eps,
                u_max=s.u_max,
                rng=self.rng,
                target_mean=self.target_mean,
                bandwidth=metrics_cfg.kernel_bandwidth,
            )
            result["metrics"] = metrics

        if self.cfg.logging.save_final:
            arrays = {"XT_eval": result["XT_eval"], "mean": result["mean"], "std": result["std"]}
            payload = arrays if self.cfg.logging.save_samples else None
            self._save_checkpoint("final", payload)

        return result

    # ---- inner steps from notebook ----
    def _fit_terminal_boundary(self, XT: np.ndarray, YT: np.ndarray) -> None:
        n_particles = XT.shape[0]
        boundary_epochs = 1
        batch_b = min(1024, n_particles)
        self.value_net.train()
        X_T_np = XT.astype(np.float32)
        Y_T_np = YT.astype(np.float32)
        t_T = torch.full((n_particles,), fill_value=1.0, dtype=torch.float32, device=self.device)
        x_T = torch.from_numpy(X_T_np).to(self.device)
        y_T = torch.from_numpy(Y_T_np).to(self.device)

        for _ in range(boundary_epochs):
            perm = torch.randperm(n_particles, device=self.device)
            for s in range(0, n_particles, batch_b):
                idx = perm[s : s + batch_b]
                t_b = t_T[idx]
                x_b = x_T[idx]
                y_b = y_T[idx]
                self.v_opt.zero_grad(set_to_none=True)
                pred_T = self.value_net(t_b, x_b)
                loss_T = nn.functional.mse_loss(pred_T, y_b)
                loss_T.backward()
                self.v_opt.step()

    def _policy_update(self, X: np.ndarray, Z: np.ndarray) -> None:
        s = self.cfg.solver
        t_list = []
        x_list = []
        u_list = []

        for i in range(s.T):
            t_i_norm = i / float(s.T)
            t_i_vec = np.full((s.n_particles, 1), t_i_norm, dtype=np.float32)
            t_list.append(t_i_vec)
            x_list.append(X[i].astype(np.float32))
            u_star_i, _ = optimal_u_and_driver(Z[i], s.eps, s.lam, s.u_max)
            u_list.append(u_star_i.astype(np.float32))

        T_policy = np.concatenate(t_list, axis=0)[:, 0]
        X_policy = np.concatenate(x_list, axis=0)
        U_policy = np.concatenate(u_list, axis=0)

        self.policy_net.train()
        t_pol = torch.from_numpy(T_policy).to(self.device)
        x_pol = torch.from_numpy(X_policy).to(self.device)
        u_pol = torch.from_numpy(U_policy).to(self.device)

        N_pol = t_pol.size(0)
        batch_p = 1024

        for _ in range(self.cfg.solver.policy_epochs):
            perm = torch.randperm(N_pol, device=self.device)
            for s_idx in range(0, N_pol, batch_p):
                idx = perm[s_idx : s_idx + batch_p]
                t_b = t_pol[idx]
                x_b = x_pol[idx]
                u_b = u_pol[idx]
                self.u_opt.zero_grad(set_to_none=True)
                u_pred = self.policy_net(t_b, x_b)
                loss_u = nn.functional.mse_loss(u_pred, u_b)
                loss_u.backward()
                self.u_opt.step()

        self.policy_net.eval()

    def _final_evaluation(self) -> Dict[str, Any]:
        s = self.cfg.solver
        dt = self.dt
        self.policy_net.eval()
        X_eval = np.zeros((s.T + 1, s.n_particles, s.d), dtype=DTYPE)
        X_eval[0] = self.X0
        with torch.no_grad():
            for i in range(s.T):
                t_i = i / float(s.T)
                x_i = torch.from_numpy(X_eval[i].astype(np.float32)).to(self.device)
                t_t = torch.tensor(t_i, dtype=torch.float32, device=self.device)
                u_nom = self.policy_net(t_t, x_i).cpu().numpy()
                u_nom = np.clip(u_nom, -s.u_max, s.u_max)
                dW = self.rng.normal(size=(s.n_particles, s.d)).astype(DTYPE) * math.sqrt(dt)
                X_eval[i + 1] = X_eval[i] + u_nom * dt + math.sqrt(s.eps) * dW

        XT_eval = X_eval[s.T]
        mean_vec = XT_eval.mean(axis=0)
        std_vec = XT_eval.std(axis=0)

        return {
            "XT_eval": XT_eval,
            "X0": self.X0,
            "target_mean": self.target_mean,
            "lam": self.cfg.solver.lam,
            "mean": mean_vec,
            "std": std_vec,
        }

    def _save_checkpoint(self, prefix: str, arrays: Optional[Dict[str, np.ndarray]]) -> None:
        run_dir = self.cfg.run_dir
        os.makedirs(run_dir, exist_ok=True)
        torch.save(self.value_net.state_dict(), os.path.join(run_dir, f"{prefix}_value_net.pt"))
        torch.save(self.policy_net.state_dict(), os.path.join(run_dir, f"{prefix}_policy_net.pt"))
        meta = {
            "solver": self.cfg.solver.__dict__,
            "penalty": {"name": self.cfg.penalty.name, "params": self.cfg.penalty.params},
            "target_mean": self.target_mean.tolist() if self.target_mean is not None else None,
        }
        with open(os.path.join(run_dir, f"{prefix}_config.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if arrays is not None:
            np.savez_compressed(os.path.join(run_dir, f"{prefix}_samples.npz"), **arrays)


def solve_from_config(cfg: ExperimentConfig) -> Dict[str, Any]:
    solver = RegularizedSBSolver(cfg)
    return solver.run()


def run_d_sb_demo(
    d: int = 2,
    target_shift: float = 2.0,
    n_particles: int = 4096,
    eps: float = 0.1,
    lam: float = 0.0,
    u_max: float = 5.0,
    T: int = 50,
    outer_loops: int = 100,
    beta_eff: float = 600.0,
    value_hidden: int = 64,
    policy_hidden: int = 64,
    value_epochs: int = 5,
    policy_epochs: int = 5,
    seed: int = 123,
    target_type: str = "sparse",
) -> Dict[str, Any]:
    """
    Convenience wrapper that reproduces toyExample_dGauss.ipynb (cells 1–295).
    """
    solver_cfg = SolverConfig(
        d=d,
        target_shift=target_shift,
        n_particles=n_particles,
        eps=eps,
        lam=lam,
        u_max=u_max,
        T=T,
        outer_loops=outer_loops,
        beta_eff=beta_eff,
        value_epochs=value_epochs,
        policy_epochs=policy_epochs,
        seed=seed,
        target_type=target_type,
    )
    value_cfg = NetworkConfig(name="ValueNet", hidden=value_hidden, lr=1e-3)
    policy_cfg = NetworkConfig(name="PolicyNet", hidden=policy_hidden, lr=1e-3)
    logging_cfg = LoggingConfig(output_dir="outputs", experiment_name="notebook_demo", save_final=False, save_samples=False)
    penalty_cfg = PenaltyConfig(name="quadratic", params={"target_type": target_type})
    metrics_cfg = MetricsConfig(enabled=False)
    exp_cfg = ExperimentConfig(
        solver=solver_cfg,
        value_net=value_cfg,
        policy_net=policy_cfg,
        logging=logging_cfg,
        penalty=penalty_cfg,
        metrics=metrics_cfg,
    )
    return solve_from_config(exp_cfg)