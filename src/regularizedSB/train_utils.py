import math
from typing import Optional

import numpy as np
import torch
from torch import nn


def train_value_global(
    v_net: nn.Module,
    v_opt: torch.optim.Optimizer,
    TX: np.ndarray,
    TY: np.ndarray,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 4096,
    shuffle: bool = True,
    verbose: bool = False,
) -> None:
    """
    Train the value network globally on the concatenated dataset (t, x) -> target Y.
      - TX: (N_total, 1 + d) ndarray where TX[:,0] = t \in [0,1], TX[:,1:] = x
      - TY: (N_total,) targets
    This mirrors the logic in toyExample_dGauss.ipynb section (1-295).
    """
    v_net.train()

    # Preload tensors on device to reduce host<->device traffic
    TX_t = torch.from_numpy(TX.astype(np.float32)).to(device)
    TY_t = torch.from_numpy(TY.astype(np.float32)).to(device)

    N = TX_t.size(0)
    B = min(batch_size, N)

    for _ in range(max(1, int(epochs))):
        if shuffle:
            perm = torch.randperm(N, device=device)
            TX_t = TX_t[perm]
            TY_t = TY_t[perm]

        for s in range(0, N, B):
            batch = TX_t[s : s + B]
            targets = TY_t[s : s + B]

            t_b = batch[:, 0]           # (B,)
            x_b = batch[:, 1:]          # (B,d)

            v_opt.zero_grad(set_to_none=True)
            pred = v_net(t_b, x_b).view_as(targets)
            loss = nn.functional.mse_loss(pred, targets)
            loss.backward()
            v_opt.step()

        if verbose:
            with torch.no_grad():
                pred_all = v_net(TX_t[:, 0], TX_t[:, 1:])
                mse = nn.functional.mse_loss(pred_all, TY_t).item()
            print(f"[train_value_global] epoch_mse={mse:.6f}")


