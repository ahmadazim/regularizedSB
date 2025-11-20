from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class NetworkConfig:
    name: str = "ValueNet"
    hidden: int = 128
    lr: float = 1e-3


@dataclass
class SolverConfig:
    d: int = 2
    target_shift: float = 2.0
    n_particles: int = 4096
    eps: float = 0.1
    lam: float = 0.0
    u_max: float = 5.0
    T: int = 50
    outer_loops: int = 100
    beta_eff: float = 600.0
    value_epochs: int = 5
    policy_epochs: int = 5
    seed: int = 123
    target_type: str = "sparse"


@dataclass
class LoggingConfig:
    output_dir: str = "outputs"
    experiment_name: str = "dGauss"
    save_every_outer: int = 0  # 0 disables intermediate saves
    save_final: bool = True
    save_samples: bool = True


@dataclass
class PenaltyConfig:
    name: str = "quadratic"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsConfig:
    enabled: bool = True
    target_samples_path: Optional[str] = None
    kernel_bandwidth: float = 1.0
    active_dim: int = 0
    lambdas: Optional[list[float]] = None


@dataclass
class ExperimentConfig:
    solver: SolverConfig
    value_net: NetworkConfig
    policy_net: NetworkConfig
    logging: LoggingConfig
    penalty: PenaltyConfig = field(default_factory=PenaltyConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def run_dir(self) -> str:
        run_dir = os.path.join(self.logging.output_dir, self.logging.experiment_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    solver = SolverConfig(**raw.get("solver", {}))
    value_net = NetworkConfig(**raw.get("value_net", {}))
    policy_net = NetworkConfig(**raw.get("policy_net", {"name": "PolicyNet"}))
    logging = LoggingConfig(**raw.get("logging", {}))
    penalty = PenaltyConfig(**raw.get("penalty", {}))
    metrics = MetricsConfig(**raw.get("metrics", {}))

    extra = {
        k: v
        for k, v in raw.items()
        if k not in {"solver", "value_net", "policy_net", "logging", "penalty", "metrics"}
    }
    return ExperimentConfig(
        solver=solver,
        value_net=value_net,
        policy_net=policy_net,
        logging=logging,
        penalty=penalty,
        metrics=metrics,
        extra=extra,
    )


