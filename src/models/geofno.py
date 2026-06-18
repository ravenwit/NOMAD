import torch
import torch.nn as nn
import torch.nn.functional as F
from .fno import BaseFNO2d
from typing import Optional, Tuple

class DiffeomorphismNet(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 2, 1)    # outputs (du, dv) displacement
        )
        # identity initialisation
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, geom_features: torch.Tensor, base_grid: torch.Tensor) -> torch.Tensor:
        deformation = self.net(geom_features)            # (B, 2, H, W)
        deformation = deformation.permute(0, 2, 3, 1)    # (B, H, W, 2)
        latent_grid = base_grid + deformation
        return torch.clamp(latent_grid, -1.0, 1.0)

class GeoFNO(nn.Module):
    def __init__(self, modes: int = 16, width: int = 64, t_in: int = 5, t_out: int = 10, geom_channels: int = 3,
                 n_theta: int = 64, n_phi: int = 64):
        super().__init__()
        # input channels: pressure history (t_in) + source history (t_in) + geometry (geom_channels)
        in_channels = t_in + t_in + geom_channels
        self.fno = BaseFNO2d(modes, width, in_channels, t_out)
        self.geo_net = DiffeomorphismNet(in_channels=geom_channels)

        # base uniform computational grid (normalised [-1, 1])
        ty = torch.linspace(-1, 1, n_theta)
        tx = torch.linspace(-1, 1, n_phi)
        mesh_y, mesh_x = torch.meshgrid(ty, tx, indexing='ij')
        self.register_buffer('base_grid', torch.stack((mesh_x, mesh_y), dim=-1).unsqueeze(0))

    def forward(self, p_in: torch.Tensor, s_in: torch.Tensor, geom_features: torch.Tensor) -> torch.Tensor:
        # p_in, s_in: (B, T_in, H, W)
        # geom_features: (B, C_geom, H, W)
        B = p_in.shape[0]
        base_grid_b = self.base_grid.expand(B, -1, -1, -1)

        # 1. learn spatial deformation from physical to flat latent space
        latent_grid = self.geo_net(geom_features, base_grid_b)

        # 2. warp physical fields into latent space
        x_physical = torch.cat([p_in, s_in, geom_features], dim=1)
        x_latent = F.grid_sample(x_physical, latent_grid,
                                 mode='bilinear', padding_mode='border', align_corners=True)

        # 3. vectorised time stepping in latent space
        p_out_latent = self.fno(x_latent)   # (B, T_out, H, W)

        # 4. warp prediction back to physical torus (identity grid)
        p_out_physical = F.grid_sample(p_out_latent, base_grid_b,
                                       mode='bilinear', padding_mode='border', align_corners=True)

        return p_out_physical.float()      # ensure float32 for loss

