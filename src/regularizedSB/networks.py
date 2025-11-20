import torch
from torch import nn


class ValueNet(nn.Module):
    """
    Simple MLP for V_phi(t, x): input [t, x] -> scalar.
    """
    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        t: (N,) or scalar tensor in [0,1]
        x: (N, d)
        returns: (N,)
        """
        if t.ndim == 0:
            t = t.view(1).expand(x.size(0))
        elif t.ndim == 1 and t.size(0) == 1:
            t = t.expand(x.size(0))
        t = t.view(-1, 1)
        inp = torch.cat([t, x], dim=1)
        out = self.net(inp).squeeze(-1)
        return out


class PolicyNet(nn.Module):
    """
    Policy net for nominal drift \bar u_psi(t, x): input [t, x] -> R^d.
    """
    def __init__(self, d: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1 + d, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.view(1).expand(x.size(0))
        elif t.ndim == 1 and t.size(0) == 1:
            t = t.expand(x.size(0))
        t = t.view(-1, 1)
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)