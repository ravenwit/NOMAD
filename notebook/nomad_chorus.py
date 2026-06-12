# Cleaned NOMAD CHORUS script

import torch
import numpy as np
import h5py
import os
import argparse
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import sys
import onnx
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr
from typing import Callable, Optional, Tuple
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from google.colab import drive
from torch.utils.data import DataLoader, Subset
from torch.utils.data import DataLoader, Subset
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxconverter_common import float16
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data import random_split, DataLoader
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader

class TorusGeometry():

    def __init__(self, R: float=1.0, r: float=0.3):
        '\n        Differential geometry of a 2D Torus.\n        Coordinates: (theta, phi)\n        theta \\in [0, 2pi) - Poloidal angle (around the tube)\n        phi \\in [0, 2pi)   - Toroidal angle (around the main axis)\n\n        Metric g:\n        ds^2 = r^2 d\\theta^2 + (R + r \\cos\\theta)^2 d\\phi^2\n        '
        self.R = R
        self.r = r

    def get_metric_elements(self, theta: torch.Tensor):
        '\n        Returns the non-zero components of the metric tensor: g_{\\theta\\theta} and g_{\\phi\\phi}\n        '
        g_tt = torch.full_like(theta, (self.r ** 2))
        g_pp = ((self.R + (self.r * torch.cos(theta))) ** 2)
        return (g_tt, g_pp)

    def get_inverse_metric_elements(self, theta: torch.Tensor):
        '\n        Returns g^{\\theta\\theta} and g^{\\phi\\phi}\n        '
        (g_tt, g_pp) = self.get_metric_elements(theta)
        return ((1.0 / g_tt), (1.0 / g_pp))

    def get_sqrt_det_g(self, theta: torch.Tensor):
        '\n        Returns \\sqrt{|g|} = r(R + r \\cos\\theta)\n        '
        return (self.r * (self.R + (self.r * torch.cos(theta))))

class TorusWaveSolverRK4():

    def __init__(self, R: float=1.0, r: float=0.3, c: float=1.0, N_theta: int=256, N_phi: int=256, CFL: float=0.5):
        '\n        High-fidelity 4th-Order Runge-Kutta acoustic wave solver on the Torus.\n        '
        self.geom = TorusGeometry(R, r)
        self.c = c
        self.N_theta = N_theta
        self.N_phi = N_phi
        self.dtheta = ((2 * np.pi) / N_theta)
        self.dphi = ((2 * np.pi) / N_phi)
        min_dx = min((r * self.dtheta), ((R - r) * self.dphi))
        self.dt = ((CFL * min_dx) / c)
        print(f'Initialized TorusSolver: Grid {N_theta}x{N_phi}. Required dt: {self.dt:.6f}')

    def generate_ricker_pulse(self, t: float, t0: float, sigma_t: float, theta0: float, phi0: float, sigma_s: float, amplitude: torch.Tensor, device: torch.device):
        '\n        A zero-mean Ricker Wavelet (Mexican Hat) source pulse.\n        '
        theta_1d = torch.linspace(0, (2 * np.pi), (self.N_theta + 1), device=device)[:(- 1)]
        phi_1d = torch.linspace(0, (2 * np.pi), (self.N_phi + 1), device=device)[:(- 1)]
        (theta_grid, phi_grid) = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
        dtheta_dist = ((((theta_grid - theta0) + np.pi) % (2 * np.pi)) - np.pi)
        dphi_dist = ((((phi_grid - phi0) + np.pi) % (2 * np.pi)) - np.pi)
        r_sq = (((self.geom.r * dtheta_dist) ** 2) + (((self.geom.R + (self.geom.r * np.cos(theta0))) * dphi_dist) ** 2))
        r_sq_over_sigma_sq = (r_sq / (sigma_s ** 2))
        spatial = ((2.0 - r_sq_over_sigma_sq) * torch.exp(((- r_sq) / (2 * (sigma_s ** 2)))))
        spatial = (spatial - spatial.mean())
        temporal = np.exp(((- ((t - t0) ** 2)) / (2 * (sigma_t ** 2))))
        S = (spatial * temporal)
        S = (S.unsqueeze((- 1)) * amplitude)
        S = S.unsqueeze(0).permute(0, 3, 1, 2)
        return S

    def _rk4_step(self, P: torch.Tensor, Q: torch.Tensor, S: torch.Tensor) -> Tuple[(torch.Tensor, torch.Tensor)]:
        '\n        Wave equation as first order system:\n        dP/dt = Q\n        dQ/dt = c^2 \\Delta_M P + S\n        '

        def dP_dt(q):
            return q

        def dQ_dt(p, s):
            LB = compute_laplace_beltrami(p, self.geom, self.dtheta, self.dphi)
            return (((self.c ** 2) * LB) + s)
        k1_P = dP_dt(Q)
        k1_Q = dQ_dt(P, S)
        P2 = (P + ((0.5 * self.dt) * k1_P))
        Q2 = (Q + ((0.5 * self.dt) * k1_Q))
        k2_P = dP_dt(Q2)
        k2_Q = dQ_dt(P2, S)
        P3 = (P + ((0.5 * self.dt) * k2_P))
        Q3 = (Q + ((0.5 * self.dt) * k2_Q))
        k3_P = dP_dt(Q3)
        k3_Q = dQ_dt(P3, S)
        P4 = (P + (self.dt * k3_P))
        Q4 = (Q + (self.dt * k3_Q))
        k4_P = dP_dt(Q4)
        k4_Q = dQ_dt(P4, S)
        P_new = (P + ((self.dt / 6.0) * (((k1_P + (2 * k2_P)) + (2 * k3_P)) + k4_P)))
        Q_new = (Q + ((self.dt / 6.0) * (((k1_Q + (2 * k2_Q)) + (2 * k3_Q)) + k4_Q)))
        return (P_new, Q_new)

    def simulate(self, num_steps: int, source_fn: Optional[Callable], device: torch.device, record_every: int=10, channels: int=1):
        P = torch.zeros((1, channels, self.N_theta, self.N_phi), device=device)
        Q = torch.zeros_like(P)
        history_P = []
        history_S = []
        t = 0.0
        for step in range(num_steps):
            if source_fn:
                S = source_fn(t, device)
            else:
                S = torch.zeros_like(P)
            (P, Q) = self._rk4_step(P, Q, S)
            t += self.dt
            if ((step % record_every) == 0):
                history_P.append(P.clone().cpu())
                history_S.append(S.clone().cpu())
                if ((step % (num_steps // 10)) == 0):
                    print(f'Simulating progress: {((100 * step) / num_steps):.1f}% (t={t:.4f}s)')
                    if ((not torch.isfinite(P).all()) or (P.abs().max() > 1000000.0)):
                        print('WARNING: Numerical instability detected!')
                        break
        return (torch.stack(history_P, dim=1), torch.stack(history_S, dim=1))

class TorusSpectralSolver():

    def __init__(self, R: float=3.0, r: float=1.0, c: float=343.0, N_theta: int=256, N_phi: int=256, CFL: float=0.1):
        '\n        Implementation of the Fourier Pseudospectral method for acoustic waves on a torus.\n        As described in acoustic-spectral.md\n        '
        self.R = R
        self.r = r
        self.c = c
        self.N_theta = N_theta
        self.N_phi = N_phi
        self.d_theta = ((2 * np.pi) / N_theta)
        self.d_phi = ((2 * np.pi) / N_phi)
        min_dx = min((r * self.d_theta), ((R - r) * self.d_phi))
        self.dt = ((CFL * min_dx) / c)
        self.k_theta = (torch.fft.fftfreq(N_theta).to(torch.float32) * N_theta)
        self.k_phi = (torch.fft.fftfreq(N_phi).to(torch.float32) * N_phi)
        (self.K_THETA, self.K_PHI) = torch.meshgrid(self.k_theta, self.k_phi, indexing='ij')
        theta_grid = torch.linspace(0, (2 * np.pi), (N_theta + 1))[:(- 1)]
        phi_grid = torch.linspace(0, (2 * np.pi), (N_phi + 1))[:(- 1)]
        (THETA, _) = torch.meshgrid(theta_grid, phi_grid, indexing='ij')
        self.THETA = THETA
        self.g_inv_tt = (1.0 / (r ** 2))
        self.g_inv_pp = (1.0 / ((R + (r * torch.cos(THETA))) ** 2))
        self.gamma_term = ((- torch.sin(THETA)) / (r * (R + (r * torch.cos(THETA)))))
        self.sqrt_g = (r * (R + (r * torch.cos(THETA))))

    def generate_ricker_pulse(self, t: float, t0: float, sigma_t: float, theta0: float, phi0: float, sigma_s: float, amplitude: torch.Tensor, device: torch.device):
        '\n        Generate a zero-mean Ricker Wavelet (Mexican Hat) source pulse in 2D.\n        '
        theta_1d = torch.linspace(0, (2 * np.pi), (self.N_theta + 1), device=device)[:(- 1)]
        phi_1d = torch.linspace(0, (2 * np.pi), (self.N_phi + 1), device=device)[:(- 1)]
        (theta_grid, phi_grid) = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
        dtheta_dist = ((((theta_grid - theta0) + np.pi) % (2 * np.pi)) - np.pi)
        dphi_dist = ((((phi_grid - phi0) + np.pi) % (2 * np.pi)) - np.pi)
        r_sq = (((self.r * dtheta_dist) ** 2) + (((self.R + (self.r * np.cos(theta0))) * dphi_dist) ** 2))
        r_sq_over_sigma_sq = (r_sq / (sigma_s ** 2))
        spatial = ((2.0 - r_sq_over_sigma_sq) * torch.exp(((- r_sq) / (2 * (sigma_s ** 2)))))
        spatial = (spatial - spatial.mean())
        temporal = np.exp(((- ((t - t0) ** 2)) / (2 * (sigma_t ** 2))))
        S_base = (spatial * temporal)
        S = (S_base.unsqueeze(0) * amplitude.view((- 1), 1, 1))
        return S

    def compute_laplace_beltrami(self, P: torch.Tensor) -> torch.Tensor:
        '\n        Computes the Laplacian in spectral space.\n        '
        device = P.device
        self.K_THETA = self.K_THETA.to(device)
        self.K_PHI = self.K_PHI.to(device)
        self.g_inv_pp = self.g_inv_pp.to(device)
        self.gamma_term = self.gamma_term.to(device)
        self.sqrt_g = self.sqrt_g.to(device)
        P_hat = torch.fft.fft2(P)
        dP_dtheta_hat = ((1j * self.K_THETA) * P_hat)
        dP_dtheta = torch.real(torch.fft.ifft2(dP_dtheta_hat))
        d2P_dtheta2_hat = ((- (self.K_THETA ** 2)) * P_hat)
        d2P_dtheta2 = torch.real(torch.fft.ifft2(d2P_dtheta2_hat))
        d2P_dphi2_hat = ((- (self.K_PHI ** 2)) * P_hat)
        d2P_dphi2 = torch.real(torch.fft.ifft2(d2P_dphi2_hat))
        laplace = (((self.g_inv_tt * d2P_dtheta2) + (self.gamma_term * dP_dtheta)) + (self.g_inv_pp * d2P_dphi2))
        return laplace

    def simulate(self, num_steps: int, source_fn: Optional[Callable], device: torch.device, record_every: int=10, channels: int=3):
        '\n        Explicit Leapfrog Time-Stepping as per acoustic-spectral.md\n        '
        P_curr = torch.zeros((channels, self.N_theta, self.N_phi), device=device)
        P_prev = torch.zeros_like(P_curr)
        history_P = []
        history_S = []
        t = 0.0
        for step in range(num_steps):
            S_curr = (source_fn(t, device) if source_fn else torch.zeros_like(P_curr))
            laplacian = self.compute_laplace_beltrami(P_curr)
            accel = ((self.c ** 2) * (laplacian + S_curr))
            P_next = (((2 * P_curr) - P_prev) + ((self.dt ** 2) * accel))
            P_prev = P_curr
            P_curr = P_next
            t += self.dt
            if ((step % record_every) == 0):
                history_P.append(P_curr.clone().cpu())
                history_S.append(S_curr.clone().cpu())
                if ((step % max(1, (num_steps // 10))) == 0):
                    print(f'Spectral Sim: {((100 * step) / num_steps):.1f}% (t={t:.4f}s)')
                    if (not torch.isfinite(P_curr).all()):
                        print('ERROR: Divergence in spectral solver!')
                        break
        P_stack = torch.stack(history_P, dim=0).unsqueeze(0)
        S_stack = torch.stack(history_S, dim=0).unsqueeze(0)
        return (P_stack, S_stack)

class TorusAcousticSimulator():

    def __init__(self, R=3.0, r=1.0, c=343.0, N_theta=128, N_phi=128, dt=None):
        self.solver = TorusSpectralSolver(R, r, c, N_theta, N_phi, CFL=0.1)
        self.dt = (dt if dt else self.solver.dt)
        self.solver.dt = self.dt

    def generate_gaussian_source(self, t, t0=0.05, sigma_t=0.01, theta0=np.pi, phi0=np.pi, sigma_s=0.5, amplitude=None, device='cpu'):
        'Deprecated: use generate_ricker_pulse instead.'
        return self.generate_ricker_pulse(t, t0, sigma_t, theta0, phi0, sigma_s, amplitude, device)

    def generate_ricker_pulse(self, t, t0=0.05, sigma_t=0.01, theta0=np.pi, phi0=np.pi, sigma_s=0.5, amplitude=None, device='cpu'):
        if (amplitude is None):
            amplitude = torch.tensor([1.0], device=device)
        return self.solver.generate_ricker_pulse(t, t0, sigma_t, theta0, phi0, sigma_s, amplitude, device)

    def generate_kicker_pulse(self, t, t0=0.05, sigma_t=0.01, theta0=np.pi, phi0=np.pi, sigma_s=0.5, amplitude=None, device='cpu'):
        'Alias for generate_ricker_pulse.'
        return self.generate_ricker_pulse(t, t0, sigma_t, theta0, phi0, sigma_s, amplitude, device)

    def simulate(self, num_steps=500, source_generator_fn=None, device='cpu', record_every=10):
        return self.solver.simulate(num_steps, source_generator_fn, device, record_every)

    def save_to_h5(self, P, S, filename):
        save_simulation_to_h5(P, S, filename, self.solver.R, self.solver.r, self.solver.dt, self.solver.N_theta, self.solver.N_phi)

class SpectralConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter((self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))
        self.weights2 = nn.Parameter((self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)))

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size((- 2)), ((x.size((- 1)) // 2) + 1), dtype=torch.cfloat, device=x.device)
        out_ft[(:, :, :self.modes1, :self.modes2)] = torch.einsum('bixy,ioxy->boxy', x_ft[(:, :, :self.modes1, :self.modes2)], self.weights1)
        out_ft[(:, :, (- self.modes1):, :self.modes2)] = torch.einsum('bixy,ioxy->boxy', x_ft[(:, :, (- self.modes1):, :self.modes2)], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size((- 2)), x.size((- 1))))
        return x

class FNO2d(nn.Module):

    def __init__(self, modes=12, width=32, in_channels=4, out_channels=1, n_layers=4):
        super(FNO2d, self).__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        self.fc0 = nn.Linear(in_channels, self.width)
        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for (conv, w) in zip(self.convs, self.ws):
            x = F.gelu((conv(x) + w(x)))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x

class DoubleConv(nn.Module):
    '(convolution => [BN] => ReLU) * 2'

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if (not mid_channels):
            mid_channels = out_channels
        self.double_conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, padding_mode='circular', bias=False), nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True), nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular', bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    'Downscaling with maxpool then double conv'

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    'Upscaling then double conv'

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, (in_channels // 2))
        else:
            self.up = nn.ConvTranspose2d(in_channels, (in_channels // 2), kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = (x2.size()[2] - x1.size()[2])
        diffX = (x2.size()[3] - x1.size()[3])
        x1 = F.pad(x1, [(diffX // 2), (diffX - (diffX // 2)), (diffY // 2), (diffY - (diffY // 2))], mode='circular')
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PeriodicUNet(nn.Module):
    "\n    Conventional UNet architecture strongly conditioned to Periodic Manifolds.\n    By setting padding_mode='circular', the convolution kernels inherently warp\n    across the left/right and top/bottom edges of the state tensor, natively simulating\n    the Torus topology.\n    "

    def __init__(self, n_channels, n_classes, bilinear=False):
        super(PeriodicUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = (2 if bilinear else 1)
        self.down4 = Down(512, (1024 // factor))
        self.up1 = Up(1024, (512 // factor), bilinear)
        self.up2 = Up(512, (256 // factor), bilinear)
        self.up3 = Up(256, (128 // factor), bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class BaseFNO2d(nn.Module):

    def __init__(self, modes=16, width=64, in_channels=12, out_channels=10, n_layers=4):
        super().__init__()
        self.width = width
        self.fc0 = nn.Linear(in_channels, width)
        self.convs = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        for (conv, w) in zip(self.convs, self.ws):
            x = F.gelu((conv(x) + w(x)))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x.float()

class DiffeomorphismNet(nn.Module):

    def __init__(self, in_channels=3, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, hidden_dim, 1), nn.GELU(), nn.Conv2d(hidden_dim, hidden_dim, 1), nn.GELU(), nn.Conv2d(hidden_dim, 2, 1))
        nn.init.zeros_(self.net[(- 1)].weight)
        nn.init.zeros_(self.net[(- 1)].bias)

    def forward(self, geom_features, base_grid):
        deformation = self.net(geom_features)
        deformation = deformation.permute(0, 2, 3, 1)
        latent_grid = (base_grid + deformation)
        return torch.clamp(latent_grid, (- 1.0), 1.0)

class GeoFNO(nn.Module):

    def __init__(self, modes=16, width=64, t_in=5, t_out=10, geom_channels=3, n_theta=64, n_phi=64):
        super().__init__()
        in_channels = ((t_in + t_in) + geom_channels)
        self.fno = BaseFNO2d(modes, width, in_channels, t_out)
        self.geo_net = DiffeomorphismNet(in_channels=geom_channels)
        ty = torch.linspace((- 1), 1, n_theta)
        tx = torch.linspace((- 1), 1, n_phi)
        (mesh_y, mesh_x) = torch.meshgrid(ty, tx, indexing='ij')
        self.register_buffer('base_grid', torch.stack((mesh_x, mesh_y), dim=(- 1)).unsqueeze(0))

    def forward(self, p_in, s_in, geom_features):
        B = p_in.shape[0]
        base_grid_b = self.base_grid.expand(B, (- 1), (- 1), (- 1))
        latent_grid = self.geo_net(geom_features, base_grid_b)
        x_physical = torch.cat([p_in, s_in, geom_features], dim=1)
        x_latent = F.grid_sample(x_physical, latent_grid, mode='bilinear', padding_mode='border', align_corners=True)
        p_out_latent = self.fno(x_latent)
        p_out_physical = F.grid_sample(p_out_latent, base_grid_b, mode='bilinear', padding_mode='border', align_corners=True)
        return p_out_physical.float()

class WaveletConv2d(nn.Module):

    def __init__(self, width):
        super().__init__()
        self.dwt = HaarWavelet2d(width)
        self.wavelet_filter = nn.Conv2d((width * 4), (width * 4), kernel_size=1)

    def forward(self, x):
        x_w = self.dwt(x)
        x_w_filtered = self.wavelet_filter(x_w)
        return self.dwt.inverse(x_w_filtered)

class TorusWaveDataset(Dataset):

    def __init__(self, h5_path, seq_len=4, transform=True):
        with h5py.File(h5_path, 'r') as f:
            raw_P = torch.from_numpy(f['pressure'][:])[(..., 0:1)]
            raw_S = torch.from_numpy(f['source'][:])[(..., 0:1)]
            self.P = raw_P.permute(0, 1, 4, 2, 3)
            self.S = raw_S.permute(0, 1, 4, 2, 3)
            (self.R, self.r) = (f.attrs['R'], f.attrs['r'])
            (self.N_theta, self.N_phi) = (f.attrs['N_theta'], f.attrs['N_phi'])
        self.seq_len = seq_len
        self.num_rollouts = self.P.shape[0]
        self.time_steps = self.P.shape[1]
        self.valid_starts_per_rollout = max(0, (self.time_steps - self.seq_len))
        self.transform = transform
        if self.transform:
            (self.p_mean, self.p_std) = (self.P.mean(), torch.clamp(self.P.std(), min=1e-08))
            (self.s_mean, self.s_std) = (self.S.mean(), torch.clamp(self.S.std(), min=1e-08))
            self.P = ((self.P - self.p_mean) / self.p_std)
            self.S = ((self.S - self.s_mean) / self.s_std)
        theta_grid = torch.linspace(0, (2 * np.pi), (self.N_theta + 1))[:(- 1)]
        phi_grid = torch.linspace(0, (2 * np.pi), (self.N_phi + 1))[:(- 1)]
        (THETA, _) = torch.meshgrid(theta_grid, phi_grid, indexing='ij')
        metric = (self.r * (self.R + (self.r * torch.cos(THETA))))
        (m_min, m_max) = ((self.r * (self.R - self.r)), (self.r * (self.R + self.r)))
        self.metric_embed = ((metric - m_min) / (m_max - m_min)).unsqueeze(0).unsqueeze(0).to(torch.float32)

    def __len__(self):
        return (self.num_rollouts * self.valid_starts_per_rollout)

    def __getitem__(self, idx):
        rollout_idx = (idx // self.valid_starts_per_rollout)
        t_start = (idx % self.valid_starts_per_rollout)
        return (self.S[(rollout_idx, t_start:(t_start + self.seq_len))].float(), self.P[(rollout_idx, t_start:(t_start + self.seq_len))].float(), self.metric_embed.expand(self.seq_len, (- 1), (- 1), (- 1)).float())

class DataDrivenTrainer():

    def __init__(self, model, lr=0.0001, weight_decay=1e-05):
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = SpectralLoss(lambda_fft=0.1)
        self.history = {'train': [], 'val': []}

    def train_epoch(self, dataloader, teacher_forcing_ratio=0.0):
        self.model.train()
        total_loss = 0.0
        unroll_steps = 4
        for (s_seq, p_seq, m_seq) in dataloader:
            (s_seq, p_seq, m_seq) = (s_seq.to(self.device), p_seq.to(self.device), m_seq.to(self.device))
            seq_len = p_seq.shape[1]
            (p_prev, p_curr) = (p_seq[(:, 0)], p_seq[(:, 1)])
            m_static = m_seq[(:, 0)]
            self.optimizer.zero_grad()
            batch_loss = 0.0
            for t in range(2, seq_len):
                s_curr = s_seq[(:, (t - 1))]
                x_in = torch.cat([p_curr, p_prev, s_curr, m_static], dim=1)
                p_next_pred = self.model(x_in)
                batch_loss += self.criterion(p_next_pred, p_seq[(:, t)])
                use_teacher = (torch.rand(1).item() < teacher_forcing_ratio)
                p_prev = p_curr
                p_curr = (p_seq[(:, t)] if use_teacher else p_next_pred)
                if ((((t - 1) % unroll_steps) == 0) or (t == (seq_len - 1))):
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    total_loss += batch_loss.item()
                    batch_loss = 0.0
                    p_prev = p_prev.detach()
                    p_curr = p_curr.detach()
        return (total_loss / len(dataloader))

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for (s_seq, p_seq, m_seq) in dataloader:
                (s_seq, p_seq, m_seq) = (s_seq.to(self.device), p_seq.to(self.device), m_seq.to(self.device))
                (p_prev, p_curr) = (p_seq[(:, 0)], p_seq[(:, 1)])
                m_static = m_seq[(:, 0)]
                batch_loss = 0.0
                for t in range(2, p_seq.shape[1]):
                    x_in = torch.cat([p_curr, p_prev, s_seq[(:, (t - 1))], m_static], dim=1)
                    p_next_pred = self.model(x_in)
                    batch_loss += self.criterion(p_next_pred, p_seq[(:, t)])
                    p_prev = p_curr
                    p_curr = p_next_pred
                total_loss += (batch_loss.item() / (p_seq.shape[1] - 2))
        return (total_loss / len(dataloader))

    def train_epochs(self, train_loader, val_loader=None, epochs=10, initial_teacher_forcing=0.5, save_every=500, checkpoint_base_path='./wno2d_operator_net.pt', dataset=None):
        for epoch in range(epochs):
            tf_ratio = (initial_teacher_forcing * (1.0 - (epoch / epochs)))
            train_loss = self.train_epoch(train_loader, teacher_forcing_ratio=tf_ratio)
            self.history['train'].append(train_loss)
            if val_loader:
                val_loss = self.evaluate(val_loader)
                self.history['val'].append(val_loss)
                if ((((epoch + 1) % max(1, (epochs // 20))) == 0) or (epoch == 0)):
                    print(f'Epoch {(epoch + 1):03d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | TF: {tf_ratio:.2f}')
            current_epoch = (epoch + 1)
            if (((current_epoch % save_every) == 0) and (current_epoch < epochs)):
                (base_dir, ext) = os.path.splitext(checkpoint_base_path)
                periodic_destination = f'{base_dir}_epoch_{current_epoch}{ext}'
                torch.save({'epoch': current_epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'history': self.history, 'p_mean': dataset.p_mean, 'p_std': dataset.p_std, 's_mean': dataset.s_mean, 's_std': dataset.s_std}, periodic_destination)
        return self.history

class ManifoldEvaluator():
    '\n    Executes unassisted autoregressive rollouts and computes error metrics\n    between the Neural Operator predictions and Spectral Ground Truth.\n    '

    def __init__(self, model, dataset, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.dataset = dataset

    @torch.no_grad()
    def generate_rollout(self, rollout_idx=0, max_steps=50):
        '\n        Runs the model strictly autoregressively for `max_steps`.\n        '
        S_full = self.dataset.S[rollout_idx].to(self.device)
        P_full = self.dataset.P[rollout_idx].to(self.device)
        max_steps = min(max_steps, P_full.shape[0])
        p_prev = P_full[0].unsqueeze(0)
        p_curr = P_full[1].unsqueeze(0)
        m_static = self.dataset.metric_embed.to(self.device)
        predictions = [p_prev.cpu(), p_curr.cpu()]
        for t in range(2, max_steps):
            s_curr = S_full[(t - 1)].unsqueeze(0)
            x_in = torch.cat([p_curr, p_prev, s_curr, m_static], dim=1)
            p_next = self.model(x_in)
            predictions.append(p_next.cpu())
            p_prev = p_curr
            p_curr = p_next
        P_pred_seq = torch.cat(predictions, dim=0)
        P_true_seq = P_full[:max_steps].cpu()
        return (P_true_seq, P_pred_seq)

    def compute_relative_l2_error(self, P_true, P_pred):
        '\n        Computes the relative L2 error norm over time: ||P_pred - P_true||_2 / ||P_true||_2\n        '
        true_flat = P_true.view(P_true.shape[0], (- 1))
        pred_flat = P_pred.view(P_pred.shape[0], (- 1))
        error_norm = torch.linalg.norm((pred_flat - true_flat), dim=1)
        true_norm = torch.linalg.norm(true_flat, dim=1)
        true_norm = torch.clamp(true_norm, min=1e-08)
        rel_error = (error_norm / true_norm)
        return rel_error.numpy()

class ScaledPhysicsLoss(nn.Module):

    def __init__(self, dataset, alpha=0.9, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        (self.dt, self.c) = (dataset.dt, dataset.c)
        (self.dtheta, self.dphi) = (((2 * np.pi) / dataset.N_theta), ((2 * np.pi) / dataset.N_phi))
        theta_grid = torch.linspace(0, (2 * np.pi), (dataset.N_theta + 1))[:(- 1)]
        phi_grid = torch.linspace(0, (2 * np.pi), (dataset.N_phi + 1))[:(- 1)]
        (THETA, _) = torch.meshgrid(theta_grid, phi_grid, indexing='ij')
        self.g_inv_theta = (1.0 / (dataset.r ** 2))
        self.g_inv_phi = (1.0 / ((dataset.R + (dataset.r * torch.cos(THETA))) ** 2))
        self.sqrt_g = (dataset.r * (dataset.R + (dataset.r * torch.cos(THETA))))
        self.manifold_volume = torch.sum(self.sqrt_g)

    def forward(self, p_next_pred, p_next_target, p_curr, p_prev, s_curr, p_scale, s_scale):
        device = p_curr.device
        if (self.g_inv_phi.device != device):
            self.g_inv_theta = torch.tensor(self.g_inv_theta, device=device)
            (self.g_inv_phi, self.sqrt_g) = (self.g_inv_phi.to(device), self.sqrt_g.to(device))
            self.manifold_volume = self.manifold_volume.to(device)
        mse_loss = self.mse(p_next_pred, p_next_target)
        dp_dtheta_pred = ((torch.roll(p_next_pred, shifts=(- 1), dims=2) - torch.roll(p_next_pred, shifts=1, dims=2)) / (2 * self.dtheta))
        dp_dphi_pred = ((torch.roll(p_next_pred, shifts=(- 1), dims=3) - torch.roll(p_next_pred, shifts=1, dims=3)) / (2 * self.dphi))
        dp_dtheta_true = ((torch.roll(p_next_target, shifts=(- 1), dims=2) - torch.roll(p_next_target, shifts=1, dims=2)) / (2 * self.dtheta))
        dp_dphi_true = ((torch.roll(p_next_target, shifts=(- 1), dims=3) - torch.roll(p_next_target, shifts=1, dims=3)) / (2 * self.dphi))
        grad_mse_loss = (self.mse(dp_dtheta_pred, dp_dtheta_true) + self.mse(dp_dphi_pred, dp_dphi_true))
        sobolev_data_loss = (mse_loss + (0.1 * grad_mse_loss))
        (p_scale, s_scale) = (p_scale.view((- 1), 1, 1, 1), s_scale.view((- 1), 1, 1, 1))
        P_next_raw = (p_next_pred * p_scale)
        P_curr_raw = (p_curr * p_scale)
        P_prev_raw = (p_prev * p_scale)
        S_curr_raw = (s_curr * s_scale)
        dp_dt_curr = ((P_curr_raw - P_prev_raw) / self.dt)
        dp_dtheta_curr = ((torch.roll(P_curr_raw, shifts=(- 1), dims=2) - torch.roll(P_curr_raw, shifts=1, dims=2)) / (2 * self.dtheta))
        dp_dphi_curr = ((torch.roll(P_curr_raw, shifts=(- 1), dims=3) - torch.roll(P_curr_raw, shifts=1, dims=3)) / (2 * self.dphi))
        grad_sq_curr = ((self.g_inv_theta * (dp_dtheta_curr ** 2)) + (self.g_inv_phi * (dp_dphi_curr ** 2)))
        Energy_curr = torch.sum((((0.5 * (dp_dt_curr ** 2)) + ((0.5 * (self.c ** 2)) * grad_sq_curr)) * self.sqrt_g), dim=[1, 2, 3])
        dp_dt_next = ((P_next_raw - P_curr_raw) / self.dt)
        dp_dtheta_next = ((torch.roll(P_next_raw, shifts=(- 1), dims=2) - torch.roll(P_next_raw, shifts=1, dims=2)) / (2 * self.dtheta))
        dp_dphi_next = ((torch.roll(P_next_raw, shifts=(- 1), dims=3) - torch.roll(P_next_raw, shifts=1, dims=3)) / (2 * self.dphi))
        grad_sq_next = ((self.g_inv_theta * (dp_dtheta_next ** 2)) + (self.g_inv_phi * (dp_dphi_next ** 2)))
        Energy_next = torch.sum((((0.5 * (dp_dt_next ** 2)) + ((0.5 * (self.c ** 2)) * grad_sq_next)) * self.sqrt_g), dim=[1, 2, 3])
        Work_done = (torch.sum(((S_curr_raw * dp_dt_curr) * self.sqrt_g), dim=[1, 2, 3]) * self.dt)
        E_char = ((0.5 * self.manifold_volume) * ((p_scale.squeeze() / self.dt) ** 2))
        energy_residual = (Energy_next - (Energy_curr + Work_done))
        physics_loss = torch.mean(((energy_residual / E_char) ** 2))
        return ((self.alpha * sobolev_data_loss) + (self.beta * physics_loss))

class FNOEvaluator():
    '\n    Executes unassisted autoregressive evaluation rollouts for FNO2d\n    and computes relative L2 metrics against Spectral Ground Truth.\n    '

    def __init__(self, model, dataset, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.dataset = dataset

    @torch.no_grad()
    def generate_rollout(self, rollout_idx=0, max_steps=51):
        '\n        Propagates the state forward strictly autoregressively up to max_steps.\n        '
        S_full = self.dataset.S[rollout_idx].to(self.device)
        P_full = self.dataset.P[rollout_idx].to(self.device)
        max_steps = min(max_steps, P_full.shape[0])
        p_prev = P_full[0].unsqueeze(0)
        p_curr = P_full[1].unsqueeze(0)
        m_static = self.dataset.metric_embed.to(self.device)
        predictions = [p_prev.cpu(), p_curr.cpu()]
        for t in range(2, max_steps):
            s_curr = S_full[(t - 1)].unsqueeze(0)
            x_in = torch.cat([p_curr, p_prev, s_curr, m_static], dim=1)
            p_next = self.model(x_in)
            predictions.append(p_next.cpu())
            p_prev = p_curr
            p_curr = p_next
        P_pred_seq = torch.cat(predictions, dim=0)
        P_true_seq = P_full[:max_steps].cpu()
        return (P_true_seq, P_pred_seq)

    def compute_relative_l2_error(self, P_true, P_pred):
        '\n        Calculates the relative L2 error norm over each temporal layer:\n        ||P_pred - P_true||_2 / ||P_true||_2\n        '
        true_flat = P_true.reshape(P_true.shape[0], (- 1))
        pred_flat = P_pred.reshape(P_pred.shape[0], (- 1))
        error_norm = torch.linalg.norm((pred_flat - true_flat), dim=1)
        true_norm = torch.linalg.norm(true_flat, dim=1)
        return (error_norm / torch.clamp(true_norm, min=1e-08)).numpy()

class ChunkedTorusDataset(Dataset):

    def __init__(self, h5_path, t_in=5, t_out=10):
        with h5py.File(h5_path, 'r') as f:
            self.P_raw = torch.from_numpy(f['pressure'][:])[(..., 0)].permute(0, 1, 3, 2)
            self.S_raw = torch.from_numpy(f['source'][:])[(..., 0)].permute(0, 1, 3, 2)
            self.R = f.attrs.get('R', 3.0)
            self.r = f.attrs.get('r', 1.0)
            self.N_theta = f.attrs.get('N_theta', 64)
            self.N_phi = f.attrs.get('N_phi', 64)
        self.t_in = t_in
        self.t_out = t_out
        self.num_rollouts = self.P_raw.shape[0]
        self.time_steps = self.P_raw.shape[1]
        self.chunk_size = (t_in + t_out)
        self.valid_starts = (self.time_steps - self.chunk_size)
        self.p_scale = torch.clamp(torch.max(torch.abs(self.P_raw)), min=0.0001)
        self.s_scale = torch.clamp(torch.max(torch.abs(self.S_raw)), min=0.0001)
        theta = torch.linspace(0, (2 * np.pi), (self.N_theta + 1))[:(- 1)]
        phi = torch.linspace(0, (2 * np.pi), (self.N_phi + 1))[:(- 1)]
        (THETA, PHI) = torch.meshgrid(theta, phi, indexing='ij')
        metric = (self.r * (self.R + (self.r * torch.cos(THETA))))
        m_norm = ((metric - metric.min()) / (metric.max() - metric.min()))
        self.geom_features = torch.stack([m_norm, (THETA / (2 * np.pi)), (PHI / (2 * np.pi))], dim=0).float()

    def __len__(self):
        return (self.num_rollouts * self.valid_starts)

    def __getitem__(self, idx):
        rollout = (idx // self.valid_starts)
        t_start = (idx % self.valid_starts)
        p_in = (self.P_raw[(rollout, t_start:(t_start + self.t_in))] / self.p_scale)
        s_in = (self.S_raw[(rollout, t_start:(t_start + self.t_in))] / self.s_scale)
        p_out = (self.P_raw[(rollout, (t_start + self.t_in):(t_start + self.chunk_size))] / self.p_scale)
        return (p_in.float(), s_in.float(), self.geom_features, p_out.float())

class FastTrainer():

    def __init__(self, model, lr=0.001, weight_decay=1e-05):
        self.device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.scaler = GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.start_epoch = 1

    def load_checkpoint(self, checkpoint_path):
        'Restores model, optimizer, scaler, and history from a serialized state.'
        if (not os.path.exists(checkpoint_path)):
            print(f'[WARNING] Target checkpoint missing at: {checkpoint_path}. Initializing fresh weights.')
            return
        print(f'''
[SYSTEM] Mounting CHORUS model checkpoint from {checkpoint_path}...''')
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        if ('optimizer_state_dict' in checkpoint):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if ('scaler_state_dict' in checkpoint):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.start_epoch = (checkpoint.get('epoch', 0) + 1)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('epoch', 0)
        if ('history' in checkpoint):
            self.history = checkpoint['history']
        print(f'  └─> Architecture restored. Resuming optimization from Epoch {self.start_epoch}.')
        print(f'''  └─> Previous Best Validation Loss: {self.best_val_loss:.6f}
''')

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for (batch_idx, (p_in, s_in, geom, p_target)) in enumerate(loader):
            (p_in, s_in) = (p_in.to(self.device), s_in.to(self.device))
            (geom, p_target) = (geom.to(self.device), p_target.to(self.device))
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device.type, enabled=(self.device.type == 'cuda')):
                p_pred = self.model(p_in, s_in, geom)
                loss = self.criterion(p_pred, p_target)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            if ((((batch_idx + 1) % max(1, (len(loader) // 5))) == 0) or ((batch_idx + 1) == len(loader))):
                sys.stdout.write(f'''  Batch {(batch_idx + 1):03d}/{len(loader)} | Active Rolling MSE: {loss.item():.6f}''')
                sys.stdout.flush()
        print()
        return (total_loss / len(loader))

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        for (p_in, s_in, geom, p_target) in loader:
            (p_in, s_in) = (p_in.to(self.device), s_in.to(self.device))
            (geom, p_target) = (geom.to(self.device), p_target.to(self.device))
            with autocast(self.device.type, enabled=(self.device.type == 'cuda')):
                p_pred = self.model(p_in, s_in, geom)
                loss = self.criterion(p_pred, p_target)
            total_loss += loss.item()
        return (total_loss / len(loader))

    def fit(self, train_loader, val_loader, total_target_epochs, dataset_meta=None, save_best_path='./best_geofno.pt', save_every=50, checkpoint_dir='./checkpoints', print_every=1):
        os.makedirs(checkpoint_dir, exist_ok=True)
        last_epoch_idx = ((self.start_epoch - 2) if (self.start_epoch > 1) else (- 1))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_target_epochs, last_epoch=last_epoch_idx)
        if (dataset_meta is None):
            dataset_meta = {'warning': 'No dataset metadata provided during training.'}
        print(f'=== Initiating Training Loop (Targeting {total_target_epochs} Total Epochs) ===')
        for epoch in range(self.start_epoch, (total_target_epochs + 1)):
            train_loss = self.train_epoch(train_loader)
            val_loss = (self.evaluate(val_loader) if (val_loader is not None) else float('nan'))
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            scheduler.step()
            if ((epoch % print_every) == 0):
                lr = scheduler.get_last_lr()[0]
                print(f'[Epoch {epoch:04d}/{total_target_epochs}] LR: {lr:.2e} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}')
            save_payload = {'project': 'CHORUS_Operator_Mapping', 'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'scaler_state_dict': self.scaler.state_dict(), 'history': self.history, 'best_val_loss': self.best_val_loss, 'dataset_configuration': dataset_meta}
            if (val_loss < self.best_val_loss):
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                save_payload['best_val_loss'] = val_loss
                torch.save(save_payload, save_best_path)
                print(f'  └─> [UPDATE] New historical best model archived. (Val Loss: {val_loss:.6f})')
            if ((epoch % save_every) == 0):
                ckpt_path = os.path.join(checkpoint_dir, f'geofno_epoch_{epoch}.pt')
                torch.save(save_payload, ckpt_path)
                print(f'  └─> [SNAPSHOT] Periodic state saved to {ckpt_path}')
        print(f'''
=== Execution Terminated. Global Best Val Loss: {self.best_val_loss:.6f} (Achieved at Epoch {self.best_epoch}) ===''')

class GeoFNOEvaluator():
    '\n    Executes chunked autoregressive evaluation rollouts for the GeoFNO\n    and computes relative L2 metrics against Spectral Ground Truth.\n    '

    def __init__(self, model, dataset, t_in=3, t_out=30, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.dataset = dataset
        self.t_in = t_in
        self.t_out = t_out

    @torch.no_grad()
    def generate_rollout(self, rollout_idx=0, max_steps=90):
        '\n        Propagates the state forward in chunks of `t_out`.\n        '
        S_full = (self.dataset.S_raw[rollout_idx].unsqueeze(0).to(self.device) / self.dataset.s_scale)
        P_full = (self.dataset.P_raw[rollout_idx].unsqueeze(0).to(self.device) / self.dataset.p_scale)
        geom_features = self.dataset.geom_features.unsqueeze(0).to(self.device)
        p_curr = P_full[(:, 0:self.t_in)]
        predictions = [p_curr.cpu()]
        current_t = self.t_in
        while (current_t < max_steps):
            s_start = (current_t - self.t_in)
            s_end = current_t
            if (s_end <= S_full.shape[1]):
                s_curr = S_full[(:, s_start:s_end)]
            else:
                s_curr = torch.zeros_like(p_curr)
            with torch.amp.autocast(self.device.type, enabled=(self.device.type == 'cuda')):
                p_next = self.model(p_curr.float(), s_curr.float(), geom_features)
            predictions.append(p_next.cpu())
            p_curr = p_next[(:, (- self.t_in):)]
            current_t += self.t_out
        P_pred_seq = torch.cat(predictions, dim=1).squeeze(0)
        P_true_seq = P_full.squeeze(0)[:P_pred_seq.shape[0]].cpu()
        P_pred_seq = (P_pred_seq * self.dataset.p_scale)
        P_true_seq = (P_true_seq * self.dataset.p_scale)
        return (P_true_seq[:max_steps], P_pred_seq[:max_steps])

    def compute_relative_l2_error(self, P_true, P_pred):
        '\n        Calculates the relative L2 error norm over each temporal layer.\n        '
        true_flat = P_true.reshape(P_true.shape[0], (- 1))
        pred_flat = P_pred.reshape(P_pred.shape[0], (- 1))
        error_norm = torch.linalg.norm((pred_flat - true_flat), dim=1)
        true_norm = torch.linalg.norm(true_flat, dim=1)
        return (error_norm / torch.clamp(true_norm, min=1e-08)).numpy()

class CombinedLoss(nn.Module):

    def __init__(self, alpha=1.0, beta=0.1, eps=1e-06):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        diff = (pred - target)
        diff_norm = torch.linalg.vector_norm(diff, dim=(1, 2, 3))
        target_norm = (torch.linalg.vector_norm(target, dim=(1, 2, 3)) + self.eps)
        rel_l2 = (diff_norm / target_norm).mean()
        return ((self.alpha * mse) + (self.beta * rel_l2))

class SpectralLoss(nn.Module):
    'Sobolev Norm Loss: Penalizes spatial errors AND frequency blurring.'

    def __init__(self, lambda_fft=0.1):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_fft = lambda_fft

    def forward(self, pred, target):
        physical_loss = self.mse(pred, target)
        pred_fft = torch.fft.rfft2(pred)
        target_fft = torch.fft.rfft2(target)
        spectral_loss = self.mse(torch.abs(pred_fft), torch.abs(target_fft))
        return (physical_loss + (self.lambda_fft * spectral_loss))

class HaarWavelet2d(nn.Module):
    'Native PyTorch Discrete Wavelet Transform (DWT/IDWT) using Haar bases.'

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[(- 0.5), (- 0.5)], [0.5, 0.5]])
        hl = torch.tensor([[(- 0.5), 0.5], [(- 0.5), 0.5]])
        hh = torch.tensor([[0.5, (- 0.5)], [(- 0.5), 0.5]])
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        kernel = kernel.repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=2, groups=self.channels)

    def inverse(self, x):
        return F.conv_transpose2d(x, self.weight, stride=2, groups=self.channels)

class WNO2d(nn.Module):

    def __init__(self, width=64, in_channels=4, out_channels=1):
        super(WNO2d, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(in_channels, self.width)
        self.conv0 = WaveletConv2d(self.width)
        self.conv1 = WaveletConv2d(self.width)
        self.conv2 = WaveletConv2d(self.width)
        self.conv3 = WaveletConv2d(self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.gelu((self.conv0(x) + self.w0(x)))
        x = F.gelu((self.conv1(x) + self.w1(x)))
        x = F.gelu((self.conv2(x) + self.w2(x)))
        x = F.gelu((self.conv3(x) + self.w3(x)))
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2)
        return x

class WNOEvaluator():
    '\n    Executes unassisted autoregressive evaluation rollouts for WNO2d\n    and computes relative L2 metrics against Spectral Ground Truth.\n    '

    def __init__(self, model, dataset, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.dataset = dataset

    @torch.no_grad()
    def generate_rollout(self, rollout_idx=0, max_steps=51):
        '\n        Propagates the state forward strictly autoregressively up to max_steps.\n        '
        S_full = self.dataset.S[rollout_idx].to(self.device)
        P_full = self.dataset.P[rollout_idx].to(self.device)
        max_steps = min(max_steps, P_full.shape[0])
        p_prev = P_full[0].unsqueeze(0)
        p_curr = P_full[1].unsqueeze(0)
        m_static = self.dataset.metric_embed.to(self.device)
        predictions = [p_prev.cpu(), p_curr.cpu()]
        for t in range(2, max_steps):
            s_curr = S_full[(t - 1)].unsqueeze(0)
            x_in = torch.cat([p_curr, p_prev, s_curr, m_static], dim=1)
            p_next = self.model(x_in)
            predictions.append(p_next.cpu())
            p_prev = p_curr
            p_curr = p_next
        P_pred_seq = torch.cat(predictions, dim=0)
        P_true_seq = P_full[:max_steps].cpu()
        return (P_true_seq, P_pred_seq)

    def compute_relative_l2_error(self, P_true, P_pred):
        '\n        Calculates the relative L2 error norm over each temporal layer.\n        '
        true_flat = P_true.reshape(P_true.shape[0], (- 1))
        pred_flat = P_pred.reshape(P_pred.shape[0], (- 1))
        error_norm = torch.linalg.norm((pred_flat - true_flat), dim=1)
        true_norm = torch.linalg.norm(true_flat, dim=1)
        return (error_norm / torch.clamp(true_norm, min=1e-08)).numpy()

def compute_gradient(f, dtheta: float, dphi: float):
    '\n    Computes \\nabla f = [ \\partial_\\theta f, \\partial_\\phi f ]\n    using 4th-order central differences with periodic boundaries.\n    f shape: (Batch, Channels, N_theta, N_phi)\n    '
    f_pad_theta = torch.nn.functional.pad(f, (0, 0, 2, 2), mode='circular')
    df_dtheta = (((((- f_pad_theta[(:, :, 4:, :)]) + (8 * f_pad_theta[(:, :, 3:(- 1), :)])) - (8 * f_pad_theta[(:, :, 1:(- 3), :)])) + f_pad_theta[(:, :, :(- 4), :)]) / (12 * dtheta))
    f_pad_phi = torch.nn.functional.pad(f, (2, 2, 0, 0), mode='circular')
    df_dphi = (((((- f_pad_phi[(:, :, :, 4:)]) + (8 * f_pad_phi[(:, :, :, 3:(- 1))])) - (8 * f_pad_phi[(:, :, :, 1:(- 3))])) + f_pad_phi[(:, :, :, :(- 4))]) / (12 * dphi))
    return (df_dtheta, df_dphi)

def compute_laplace_beltrami(f: torch.Tensor, geometry: TorusGeometry, dtheta: float, dphi: float):
    '\n    Computes \\Delta_M f = \\frac{1}{\\sqrt{|g|}} \\partial_i (\\sqrt{|g|} g^{ij} \\partial_j f)\n    f shape: (Batch, Channels, N_theta, N_phi)\n    '
    device = f.device
    N_theta = f.shape[2]
    theta_1d = torch.linspace(0, (2 * np.pi), (N_theta + 1), device=device)[:(- 1)]
    theta_grid = theta_1d.view(1, 1, N_theta, 1)
    sqrt_g = geometry.get_sqrt_det_g(theta_grid)
    (g_inv_tt, g_inv_pp) = geometry.get_inverse_metric_elements(theta_grid)
    (df_dtheta, df_dphi) = compute_gradient(f, dtheta, dphi)
    V_theta = ((sqrt_g * g_inv_tt) * df_dtheta)
    V_phi = ((sqrt_g * g_inv_pp) * df_dphi)
    (dV_theta_dtheta, _) = compute_gradient(V_theta, dtheta, dphi)
    (_, dV_phi_dphi) = compute_gradient(V_phi, dtheta, dphi)
    laplacian = ((1.0 / sqrt_g) * (dV_theta_dtheta + dV_phi_dphi))
    return laplacian

def save_simulation_to_h5(P, S, filename, simulator, N_theta, N_phi):
    '\n    Saves NOMAD simulation data to HDF5.\n    Shapes expected: P, S: (Batch, Time, Channels, H, W)\n    '
    with h5py.File(filename, 'w') as f:
        P_save = P.permute(0, 1, 3, 4, 2).numpy()
        S_save = S.permute(0, 1, 3, 4, 2).numpy()
        f.create_dataset('pressure', data=P_save, compression='gzip')
        f.create_dataset('source', data=S_save, compression='gzip')
        f.attrs['R'] = simulator.solver.R
        f.attrs['r'] = simulator.solver.r
        f.attrs['dt'] = simulator.solver.dt
        f.attrs['N_theta'] = N_theta
        f.attrs['N_phi'] = N_phi
        f.attrs['c'] = simulator.solver.c
    print(f'Dataset successfully saved to {filename}')

def generate_dataset(num_examples=10, sequence_length=128):
    all_P = []
    all_S = []
    simulator = TorusSpectralSolver(R=1.5, r=0.5, c=1.0, N_theta=N_THETA, N_phi=N_PHI, CFL=0.1)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Generating high-fidelity spectral dataset on: {device}')
    for i in range(num_examples):
        t0 = np.random.uniform(0.01, 0.05)
        theta0 = np.random.uniform(0, (2 * np.pi))
        phi0 = np.random.uniform(0, (2 * np.pi))
        source_fn = (lambda t, dev: simulator.generate_ricker_pulse(t, t0=t0, sigma_t=0.01, theta0=theta0, phi0=phi0, sigma_s=0.1, amplitude=torch.tensor([1.0, 1.0, 1.0], device=dev), device=dev))
        (P_seq, S_seq) = simulator.simulate(num_steps=sequence_length, source_fn=source_fn, device=device, record_every=1)
        all_P.append(P_seq[0])
        all_S.append(S_seq[0])
        if ((i % 2) == 0):
            print(f'Generated sequence {(i + 1)}/{num_examples}')
    P_tensor = torch.stack(all_P, dim=0)
    S_tensor = torch.stack(all_S, dim=0)
    h5_path = 'training_data.h5'
    save_simulation_to_h5(P_tensor, S_tensor, h5_path, simulator.R, simulator.r, simulator.dt, simulator.N_theta, simulator.N_phi)
    return h5_path

def generate_multi_rollout_dataset(filename, num_rollouts=50, steps_per_rollout=512, record_every=10, R=3.0, r=1.0, N_theta=64, N_phi=64):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Generating Multi-Pulse Dataset on {device}...')
    simulator = TorusAcousticSimulator(R=R, r=r, N_theta=N_theta, N_phi=N_phi, c=1.0)
    (P_list, S_list) = ([], [])
    for i in tqdm(range(num_rollouts), desc='Simulating Rollouts'):
        num_pulses = np.random.randint(1, 4)
        pulses = []
        for p in range(num_pulses):
            pulses.append({'t0': np.random.uniform(0.01, 0.05), 'theta0': np.random.uniform(0, (2 * np.pi)), 'phi0': np.random.uniform(0, (2 * np.pi)), 'sigma_s': np.random.uniform(0.08, 0.3)})

        def multi_pulse_source_fn(t, dev):
            total_S = torch.zeros((3, N_theta, N_phi), device=dev)
            safe_amp = (torch.tensor([1.0, 1.0, 1.0], device=dev) / num_pulses)
            for p_params in pulses:
                total_S += simulator.generate_kicker_pulse(t, t0=p_params['t0'], sigma_t=0.01, theta0=p_params['theta0'], phi0=p_params['phi0'], sigma_s=p_params['sigma_s'], amplitude=safe_amp, device=dev)
            return total_S
        (P_seq, S_seq) = simulator.simulate(num_steps=steps_per_rollout, source_generator_fn=multi_pulse_source_fn, device=device, record_every=record_every)
        P_list.append(P_seq.cpu())
        S_list.append(S_seq.cpu())
    P_all = torch.cat(P_list, dim=0)
    S_all = torch.cat(S_list, dim=0)
    save_simulation_to_h5(P_all, S_all, filename, R, r, simulator.dt, N_theta, N_phi)

def verify_dataset(filepath):
    with h5py.File(filepath, 'r') as f:
        P = f['pressure'][:]
        S = f['source'][:]
        print(f'Dataset successfully loaded shape: {P.shape} (Batch, Time, H, W, Channels)')
        (fig, axes) = plt.subplots(1, 4, figsize=(16, 4))
        time_indices = [0, (P.shape[1] // 4), (P.shape[1] // 2), (P.shape[1] - 1)]
        for (i, t_idx) in enumerate(time_indices):
            field = P[(15, t_idx, :, :, 0)]
            vmax = (np.max(np.abs(field)) + 1e-09)
            vmin = (- vmax)
            im = axes[i].imshow(field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
            axes[i].set_title(f'Time Step {t_idx}')
            axes[i].axis('off')
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
        plt.suptitle('Wave Propagation on Torus (Dataset Ground Truth)')
        plt.show()

def explore_hdf5(filepath):
    print(f'--- Exploring {filepath} ---')
    with h5py.File(filepath, 'r') as f:
        print('Attributes:')
        for (k, v) in f.attrs.items():
            print(f'  {k}: {v}')
        print('\nDatasets:')
        for key in f.keys():
            ds = f[key]
            print(f'  {key}: shape={ds.shape}, dtype={ds.dtype}, chunking={ds.chunks}, compression={ds.compression}')
        P = f['pressure'][:]
        (fig, axes) = plt.subplots(1, 2, figsize=(15, 5))
        (H, W) = (P.shape[2], P.shape[3])
        center_trace = P[(0, :, (H // 2), (W // 2), :)]
        axes[0].plot(center_trace)
        axes[0].set_title(f'Time Trace at Center (θ={(H // 2)}, φ={(W // 2)})')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Pressure Amplitude')
        if (center_trace.shape[1] == 3):
            axes[0].legend(['Channel 0 (R)', 'Channel 1 (G)', 'Channel 2 (B)'])
        max_amp = np.max(np.abs(P[(7, :, :, :, 0)]), axis=0)
        im = axes[1].imshow(max_amp, cmap='magma', aspect='auto')
        axes[1].set_title('Maximum Amplitude Map (Channel 0)')
        axes[1].set_xlabel('φ (Toroidal)')
        axes[1].set_ylabel('θ (Poloidal)')
        fig.colorbar(im, ax=axes[1], label='Max |Pressure|')
        plt.tight_layout()
        plt.show()

def plot_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label='Train Loss', color='blue')
    if history['val']:
        plt.plot(history['val'], label='Validation Loss', color='orange', linestyle='--')
    plt.yscale('log')
    plt.title('Autoregressive Rollout Loss (Manifold Operator Mapping)')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()

def run_training_pipeline(h5_file_path):
    BATCH_SIZE = 4
    SEQ_LEN = 51
    EPOCHS = 1000
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 1e-05
    SPLIT_RATIO = 0.8
    print(('=' * 70))
    print('      INITIALIZING PIPELINE: PERIODIC OPERATOR MANIFOLD RUNNER      ')
    print(('=' * 70))
    if (not os.path.exists(h5_file_path)):
        print(f'Process Aborted. Dataset path targets undefined destination: {h5_file_path}')
        return
    master_dataset = TorusWaveDataset(h5_file_path, seq_len=SEQ_LEN, transform=True)
    total_sequences = len(master_dataset)
    train_size = int((SPLIT_RATIO * total_sequences))
    val_size = (total_sequences - train_size)
    (train_set, val_set) = random_split(master_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    print(f'Data Registry: Created {total_sequences} split window sequences.')
    print(f'├── Training Subarray Stack:   {train_size} sequences')
    print(f'└── Validation Subarray Stack: {val_size} sequences')
    model = PeriodicUNet(n_channels=4, n_classes=1, bilinear=True)
    trainer = DataDrivenTrainer(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f'Execution targeting compute platform: {trainer.device.type.upper()}')
    print('\nBeginning Auto-Regressive Rollout Optimizations...')
    print(('-' * 70))
    history = trainer.train_epochs(train_loader=train_loader, val_loader=val_loader, epochs=EPOCHS, initial_teacher_forcing=0.5)
    print(('-' * 70))
    plot_loss(history)
    checkpoint_destination = os.path.join('.', 'toroidal_operator_net.pt')
    torch.save({'model_state_dict': model.state_dict(), 'history': history, 'p_mean': master_dataset.p_mean, 'p_std': master_dataset.p_std, 's_mean': master_dataset.s_mean, 's_std': master_dataset.s_std}, checkpoint_destination)
    print(f'''Optimization completed successfully. Model checkpoint written to: {checkpoint_destination}
''')

def visualize_rollout_comparison(evaluator, rollout_idx=0, max_steps=50, num_snapshots=5):
    '\n    Plots the spatial wavefield propagation and error maps.\n    '
    print(f'Generating unassisted rollout for index {rollout_idx}...')
    (P_true, P_pred) = evaluator.generate_rollout(rollout_idx=rollout_idx, max_steps=max_steps)
    P_true = evaluator.dataset.denormalize_p(P_true).numpy()
    P_pred = evaluator.dataset.denormalize_p(P_pred).numpy()
    error_matrix = np.abs((P_pred - P_true))
    t_indices = np.linspace(2, (max_steps - 1), num_snapshots, dtype=int)
    (fig, axes) = plt.subplots(3, num_snapshots, figsize=((4 * num_snapshots), 10))
    for (i, t) in enumerate(t_indices):
        true_field = P_true[(t, 0)]
        pred_field = P_pred[(t, 0)]
        err_field = error_matrix[(t, 0)]
        vmax = (np.max(np.abs(true_field)) + 1e-09)
        vmin = (- vmax)
        ax_true = axes[(0, i)]
        im_true = ax_true.imshow(true_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        ax_true.set_title(f'Target (t={t})')
        ax_true.axis('off')
        ax_pred = axes[(1, i)]
        im_pred = ax_pred.imshow(pred_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        ax_pred.set_title(f'Prediction (t={t})')
        ax_pred.axis('off')
        ax_err = axes[(2, i)]
        err_max = (np.max(err_field) + 1e-09)
        im_err = ax_err.imshow(err_field, cmap='magma', vmin=0, vmax=err_max, aspect='auto')
        ax_err.set_title(f'Abs Error (t={t})')
        ax_err.axis('off')
    fig.colorbar(im_true, ax=axes[(0:2, :)].ravel().tolist(), shrink=0.8, label='Pressure Amplitude')
    fig.colorbar(im_err, ax=axes[(2, :)].ravel().tolist(), shrink=0.8, label='Error Magnitude')
    plt.suptitle('Manifold Autoregressive Rollout: Spectral Ground Truth vs Neural Operator', fontsize=16)
    plt.show()
    rel_error = evaluator.compute_relative_l2_error(torch.tensor(P_true), torch.tensor(P_pred))
    plt.figure(figsize=(10, 4))
    plt.plot(rel_error, color='red', linewidth=2)
    plt.title('Accumulated Relative L2 Error Over Time')
    plt.xlabel('Time Step (t)')
    plt.ylabel('Relative Error Norm')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def load_and_evaluate(checkpoint_path, h5_file_path, rollout_idx=45, max_steps=40):
    '\n    Loads the trained PeriodicUNet and its normalization statistics,\n    ensuring proper device placement to avoid CPU/CUDA mismatches,\n    then executes an unassisted autoregressive rollout for evaluation.\n    '
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Executing inference engine on: {device.type.upper()}')
    if (not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(f'Checkpoint file missing at {checkpoint_path}')
    print(f'Loading checkpoint parameters from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print('Reconstructing PeriodicUNet manifold operator...')
    model = PeriodicUNet(n_channels=4, n_classes=1, bilinear=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print('Initializing evaluation dataset and syncing normalization bounds...')
    eval_dataset = TorusWaveDataset(h5_file_path, seq_len=max_steps, transform=False)
    eval_dataset.p_mean = checkpoint['p_mean'].cpu()
    eval_dataset.p_std = checkpoint['p_std'].cpu()
    eval_dataset.s_mean = checkpoint['s_mean'].cpu()
    eval_dataset.s_std = checkpoint['s_std'].cpu()
    eval_dataset.P = ((eval_dataset.P - eval_dataset.p_mean) / eval_dataset.p_std)
    eval_dataset.S = ((eval_dataset.S - eval_dataset.s_mean) / eval_dataset.s_std)
    print(f'Dataset successfully normalized across training bounds on host device.')
    if ('history' in checkpoint):
        print('Rendering recorded training convergence history...')
        history = checkpoint['history']
        plt.figure(figsize=(10, 4))
        plt.plot(history['train'], label='Train Loss', color='blue')
        if history['val']:
            plt.plot(history['val'], label='Validation Loss', color='orange', linestyle='--')
        plt.yscale('log')
        plt.title('Saved Autoregressive Rollout Loss Convergence')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss (log)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.show()
    evaluator = ManifoldEvaluator(model, eval_dataset, device=device)
    visualize_rollout_comparison(evaluator, rollout_idx=rollout_idx, max_steps=max_steps, num_snapshots=5)

def plot_pde_convergence(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label='WNO2d Training Loss', color='#1f77b4', lw=2)
    if history['val']:
        plt.plot(history['val'], label='WNO2d Validation Loss', color='#ff7f0e', linestyle='--', lw=2)
    plt.yscale('log')
    plt.title('Wavelet Operator Optimization Path (Pushforward + Spectral Loss)', fontsize=12)
    plt.xlabel('Epoch Index')
    plt.ylabel('Sobolev Norm Error (Log Scale)')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def run_fno_pipeline(index, h5_file_path):
    '\n    Executed identically on all 8 TPU Cores by xmp.spawn.\n    '
    BATCH_SIZE = 8
    SEQ_LEN = 51
    EPOCHS = 2000
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-05
    SPLIT_RATIO = 0.8
    checkpoint_destination = './fno2d_operator_net_multippulse.pt'
    if xm.is_master_ordinal():
        xm.master_print(('=' * 75))
        xm.master_print('      NOMAD EXECUTIVE WORKFLOW: XLA/TPU FNO DEPLOYMENT       ')
        xm.master_print(('=' * 75))
    master_dataset = TorusWaveDataset(h5_file_path, seq_len=SEQ_LEN, transform=True)
    total_sequences = len(master_dataset)
    train_size = int((SPLIT_RATIO * total_sequences))
    val_size = (total_sequences - train_size)
    torch.manual_seed(42)
    (train_set, val_set) = random_split(master_dataset, [train_size, val_size])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, num_replicas=xr.world_size(), rank=xr.global_ordinal(), shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, num_replicas=xr.world_size(), rank=xr.global_ordinal(), shuffle=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=train_sampler, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, sampler=val_sampler, drop_last=False)
    if xm.is_master_ordinal():
        xm.master_print(f'Subspace Partitions: Extracted {total_sequences} total spatial step slices.')
    model = FNO2d(modes=24, width=64, in_channels=6, out_channels=1, n_layers=6)
    trainer = DataDrivenTrainer(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if xm.is_master_ordinal():
        xm.master_print('\nExecuting Distributed Rollout Minimizations...')
        xm.master_print(('-' * 75))
    history = trainer.train_epochs(train_loader=train_loader, val_loader=val_loader, epochs=EPOCHS, checkpoint_base_path=checkpoint_destination, initial_teacher_forcing=0.1, dataset=master_dataset, save_every=200)
    if xm.is_master_ordinal():
        xm.master_print(('-' * 75))
        xm.save({'epoch': EPOCHS, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': trainer.optimizer.state_dict(), 'history': history, 'p_mean': master_dataset.p_mean, 'p_std': master_dataset.p_std, 's_mean': master_dataset.s_mean, 's_std': master_dataset.s_std}, checkpoint_destination)
        xm.master_print(f'''Pipeline complete. Structural operator weights archived at: {checkpoint_destination}
''')

def resume_fno_pipeline(h5_file_path, checkpoint_to_load, new_total_epochs=3000):
    BATCH_SIZE = 8
    SEQ_LEN = 99
    LEARNING_RATE = 0.0005
    WEIGHT_DECAY = 1e-05
    SPLIT_RATIO = 0.8
    checkpoint_destination = './fno2d_operator_net_multipulse2.pt'
    print(('=' * 75))
    print('      CHORUS EXECUTIVE WORKFLOW: RESUMING TRAINING PIPELINE      ')
    print(('=' * 75))
    master_dataset = TorusWaveDataset(h5_file_path, seq_len=SEQ_LEN, transform=True)
    total_sequences = len(master_dataset)
    train_size = int((SPLIT_RATIO * total_sequences))
    val_size = (total_sequences - train_size)
    (train_set, val_set) = random_split(master_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)
    model = FNO2d(modes=24, width=64, in_channels=6, out_channels=1, n_layers=6)
    trainer = DataDrivenTrainer(model, dataset=master_dataset, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    (start_epoch, best_val_loss) = trainer.load_checkpoint(checkpoint_to_load)
    print(f'''
Resuming optimization from epoch {start_epoch} to {new_total_epochs}...''')
    print(('-' * 75))
    history = trainer.train_epochs(train_loader=train_loader, val_loader=val_loader, total_epochs=new_total_epochs, initial_tf=2.0, save_every=100, print_every=10, checkpoint_base=checkpoint_destination, start_epoch=start_epoch, best_val_loss=best_val_loss)

def visualize_fno_rollout(evaluator, rollout_idx=45, max_steps=51, num_snapshots=5):
    '\n    Generates side-by-side matrix comparisons and displays error accumulation over time.\n    '
    (P_true, P_pred) = evaluator.generate_rollout(rollout_idx=rollout_idx, max_steps=max_steps)
    P_true = evaluator.dataset.denormalize_p(P_true).numpy()
    P_pred = evaluator.dataset.denormalize_p(P_pred).numpy()
    abs_error = np.abs((P_pred - P_true))
    t_indices = np.linspace(2, (max_steps - 1), num_snapshots, dtype=int)
    (fig, axes) = plt.subplots(3, num_snapshots, figsize=((4 * num_snapshots), 9.5))
    for (i, t) in enumerate(t_indices):
        true_field = P_true[(t, 0)]
        pred_field = P_pred[(t, 0)]
        err_field = abs_error[(t, 0)]
        vmax = (np.max(np.abs(true_field)) + 1e-09)
        vmin = (- vmax)
        ax_true = axes[(0, i)]
        im_true = ax_true.imshow(true_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        ax_true.set_title(f'Spectral Target (t={t})', fontsize=10)
        ax_true.axis('off')
        ax_pred = axes[(1, i)]
        im_pred = ax_pred.imshow(pred_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        ax_pred.set_title(f'FNO2d Prediction (t={t})', fontsize=10)
        ax_pred.axis('off')
        ax_err = axes[(2, i)]
        err_max = (np.max(err_field) + 1e-09)
        im_err = ax_err.imshow(err_field, cmap='magma', vmin=0, vmax=err_max, aspect='auto')
        ax_err.set_title(f'Absolute Error (t={t})', fontsize=10)
        ax_err.axis('off')
    fig.colorbar(im_true, ax=axes[(0:2, :)].ravel().tolist(), shrink=0.75, label='Pressure Coordinate Field Magnitude')
    fig.colorbar(im_err, ax=axes[(2, :)].ravel().tolist(), shrink=0.75, label='Absolute Error Intensity')
    plt.suptitle('Unassisted Autoregressive Rollout Evaluation: Spectral Solver vs FNO2d Matrix Space', fontsize=14, y=0.98)
    plt.show()
    rel_errors = evaluator.compute_relative_l2_error(torch.tensor(P_true), torch.tensor(P_pred))
    plt.figure(figsize=(10, 3.5))
    plt.plot(rel_errors, color='#d62728', linewidth=2.5, label='Relative $L_2$ Error Deviation')
    plt.axhline(0.1, color='gray', linestyle=':', alpha=0.7, label='10% Bound Tolerance')
    plt.title('Accumulated Global Relative $L_2$ Error Matrix Norm Over Temporal Horizon', fontsize=11)
    plt.xlabel('Temporal Frame Step Index ($t$)')
    plt.ylabel('Relative Error Amplitude $\\epsilon(t)$')
    plt.xlim(0, (max_steps - 1))
    plt.ylim(0, max(1.1, (np.max(rel_errors) * 1.1)))
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def load_and_compare_fno(checkpoint_path, h5_file_path, target_idx=45):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Loading environment settings... Device set to: {device.type.upper()}')
    if (not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(f'Target optimization weights missing at destination: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FNO2d(modes=12, width=32, in_channels=4, out_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    eval_dataset = TorusWaveDataset(h5_file_path, seq_len=51, transform=False)
    stat_targets = [('p_mean', eval_dataset.P, 'mean'), ('p_std', eval_dataset.P, 'std'), ('s_mean', eval_dataset.S, 'mean'), ('s_std', eval_dataset.S, 'std')]
    for (attr_name, source_tensor, op_type) in stat_targets:
        val = checkpoint.get(attr_name, None)
        if (val is not None):
            if torch.is_tensor(val):
                setattr(eval_dataset, attr_name, val.cpu())
            else:
                setattr(eval_dataset, attr_name, torch.tensor(val, dtype=torch.float32))
        else:
            print(f"[INFO]: '{attr_name}' missing from checkpoint target. Reconstructing from source on host...")
            if (op_type == 'mean'):
                calculated_val = source_tensor.mean()
            else:
                calculated_val = torch.clamp(source_tensor.std(), min=1e-08)
            setattr(eval_dataset, attr_name, calculated_val)
    eval_dataset.P = ((eval_dataset.P - eval_dataset.p_mean) / eval_dataset.p_std)
    eval_dataset.S = ((eval_dataset.S - eval_dataset.s_mean) / eval_dataset.s_std)
    print('Dataset normalization metrics aligned successfully.')
    if (('history' in checkpoint) and (checkpoint['history'] is not None)):
        history = checkpoint['history']
        if (('train' in history) and (len(history['train']) > 0)):
            plt.figure(figsize=(10, 3.5))
            plt.plot(history['train'], label='FNO2d Training History Tracking', color='#1f77b4', lw=2)
            if history.get('val'):
                plt.plot(history['val'], label='FNO2d Validation History Tracking', color='#ff7f0e', linestyle='--', lw=2)
            plt.yscale('log')
            plt.title('Fourier Operator Network Optimization Path')
            plt.xlabel('Training Epochs')
            plt.ylabel('Loss Magnitude (MSE Log Grid)')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.show()
    evaluator = FNOEvaluator(model, eval_dataset, device=device)
    visualize_fno_rollout(evaluator, rollout_idx=target_idx, max_steps=51)

def run_fast_pipeline(h5_file_path):
    T_IN = 3
    T_OUT = 30
    BATCH_SIZE = 8
    EPOCHS = 50
    LR = 0.001
    WD = 1e-05
    print('Loading dataset ...')
    full_ds = ChunkedTorusDataset(h5_file_path, t_in=T_IN, t_out=T_OUT)
    print(f'Original dataset:')
    print(f'  Rollouts: {full_ds.num_rollouts}, time steps: {full_ds.time_steps}')
    print(f'  T_in: {T_IN}, T_out: {T_OUT}, chunk size: {full_ds.chunk_size}')
    print(f'  Valid windows per rollout: {full_ds.valid_starts}')
    print(f'  Total windows: {len(full_ds)}')
    print(f'  Grid: {full_ds.N_theta} x {full_ds.N_phi}')
    print(f'  Global scaling: p_scale={full_ds.p_scale:.3f}, s_scale={full_ds.s_scale:.3f}')
    n_rollouts = full_ds.num_rollouts
    rollout_indices = np.arange(n_rollouts)
    (train_roll, val_roll) = train_test_split(rollout_indices, test_size=0.2, random_state=42)
    vs = full_ds.valid_starts
    train_win = [((r * vs) + t) for r in train_roll for t in range(vs)]
    val_win = [((r * vs) + t) for r in val_roll for t in range(vs)]
    train_ds = Subset(full_ds, train_win)
    val_ds = Subset(full_ds, val_win)
    print(f'''
Stratified split (by rollout):''')
    print(f'  Train rollouts: {len(train_roll)}, windows: {len(train_ds)}')
    print(f'  Val   rollouts: {len(val_roll)}, windows: {len(val_ds)}')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
    model = GeoFNO(modes=20, width=48, t_in=T_IN, t_out=T_OUT, geom_channels=3, n_theta=full_ds.N_theta, n_phi=full_ds.N_phi)
    print(f'''
Model: GeoFNO | params: {sum((p.numel() for p in model.parameters())):,}''')
    trainer = FastTrainer(model, lr=LR, weight_decay=WD)
    print(f'''Training on {trainer.device.type.upper()}
''')
    trainer.fit(train_loader, val_loader, total_target_epochs=EPOCHS, save_best_path='./best_geofno_small_20_48.pt', save_every=20, checkpoint_dir='./checkpoints', print_every=5)
    torch.save({'model_state_dict': model.state_dict(), 'trainer_history': trainer.history}, './geofno_small_20_48.pt')
    print('\nFinal model saved. Done.')

def inspect_checkpoint(filepath):
    '\n    Cracks open a PyTorch .pt file and audits its architecture,\n    metadata, and parameter footprint.\n    '
    if (not os.path.exists(filepath)):
        print(f'[ERROR] File not found at: {filepath}')
        return
    print(('=' * 85))
    print(f'🔍 INSPECTING CHECKPOINT: {filepath}')
    print(('=' * 85))
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f'[ERROR] Failed to load checkpoint. Corrupted file? Details: {e}')
        return
    print('\n[1. TOP-LEVEL KEYS]')
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        print(f'Keys found: {keys}')
    else:
        print('Checkpoint is not a dictionary. It is likely a raw tensor or custom object.')
        return
    print('\n[2. EMBEDDED METADATA]')
    if ('project' in checkpoint):
        print(f"  ├─ Project ID: {checkpoint['project']}")
    if ('epoch' in checkpoint):
        print(f"  ├─ Saved at Epoch: {checkpoint['epoch']}")
    if ('best_val_loss' in checkpoint):
        print(f"  ├─ Best Validation Loss: {checkpoint['best_val_loss']:.6f}")
    if ('dataset_configuration' in checkpoint):
        print('  ├─ Dataset Configuration:')
        for (k, v) in checkpoint['dataset_configuration'].items():
            print(f'  │    ├─ {k}: {v}')
    else:
        print('  ├─ No dataset physics metadata found.')
    print('\n[3. MODEL ARCHITECTURE & WEIGHT TENSORS]')
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if (not isinstance(state_dict, dict)):
        print('[WARNING] Checkpoint contains a full model object, not a state_dict. Audit aborted.')
        return
    print(('-' * 85))
    print(f"{'Layer Name':<45} | {'Tensor Shape':<20} | {'Parameters'}")
    print(('-' * 85))
    total_params = 0
    for (layer_name, weight_tensor) in state_dict.items():
        if ('num_batches_tracked' in layer_name):
            continue
        num_params = weight_tensor.numel()
        total_params += num_params
        shape_str = str(list(weight_tensor.shape))
        display_name = (layer_name if (len(layer_name) <= 45) else (layer_name[:42] + '...'))
        print(f'{display_name:<45} | {shape_str:<20} | {num_params:,}')
    print(('-' * 85))
    print(f'TOTAL LEARNABLE PARAMETERS : {total_params:,}')
    print(f'BARE MODEL SIZE (FP32)     : {((total_params * 4) / (1024 ** 2)):.2f} MB')
    print('\n[4. TRAINING STATES (For Resumption)]')
    if ('optimizer_state_dict' in checkpoint):
        opt_state = checkpoint['optimizer_state_dict']
        print('  ├─ Optimizer State     : [DETECTED]')
        if ('param_groups' in opt_state):
            lr = opt_state['param_groups'][0].get('lr', 'Unknown')
            print(f'  ├─ Last Learning Rate  : {lr}')
    else:
        print('  ├─ Optimizer State     : [MISSING]')
    if ('scaler_state_dict' in checkpoint):
        print('  └─ AMP Scaler State    : [DETECTED]')
    else:
        print('  └─ AMP Scaler State    : [MISSING]')
    print((('=' * 85) + '\n'))

def visualize_geofno_rollout(evaluator, rollout_idx=45, max_steps=55, num_snapshots=5):
    '\n    Generates side-by-side matrix comparisons for the new tensor shapes.\n    '
    (P_true, P_pred) = evaluator.generate_rollout(rollout_idx=rollout_idx, max_steps=max_steps)
    P_true = P_true.numpy()
    P_pred = P_pred.numpy()
    abs_error = np.abs((P_pred - P_true))
    t_indices = np.linspace(evaluator.t_in, (max_steps - 1), num_snapshots, dtype=int)
    (fig, axes) = plt.subplots(3, num_snapshots, figsize=((4 * num_snapshots), 9.5))
    for (i, t) in enumerate(t_indices):
        true_field = P_true[t]
        pred_field = P_pred[t]
        err_field = abs_error[t]
        vmax = (np.max(np.abs(true_field)) + 1e-09)
        vmin = (- vmax)
        ax_true = axes[(0, i)]
        im_true = ax_true.imshow(true_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        ax_true.set_title(f'Spectral Target (t={t})', fontsize=10)
        ax_true.axis('off')
        ax_pred = axes[(1, i)]
        im_pred = ax_pred.imshow(pred_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        ax_pred.set_title(f'Geo-FNO Prediction (t={t})', fontsize=10)
        ax_pred.axis('off')
        ax_err = axes[(2, i)]
        err_max = (np.max(err_field) + 1e-09)
        im_err = ax_err.imshow(err_field, cmap='magma', vmin=0, vmax=err_max, aspect='auto')
        ax_err.set_title(f'Absolute Error (t={t})', fontsize=10)
        ax_err.axis('off')
    fig.colorbar(im_true, ax=axes[(0:2, :)].ravel().tolist(), shrink=0.75, label='Pressure Coordinate Field')
    fig.colorbar(im_err, ax=axes[(2, :)].ravel().tolist(), shrink=0.75, label='Absolute Error Intensity')
    plt.suptitle('Chunked Autoregressive Evaluation: Spectral Solver vs Geo-FNO', fontsize=14, y=0.98)
    plt.show()
    rel_errors = evaluator.compute_relative_l2_error(torch.tensor(P_true), torch.tensor(P_pred))
    plt.figure(figsize=(10, 3.5))
    plt.plot(rel_errors, color='#d62728', linewidth=2.5, label='Relative $L_2$ Error Deviation')
    plt.axhline(0.1, color='gray', linestyle=':', alpha=0.7, label='10% Bound Tolerance')
    plt.axvspan(0, evaluator.t_in, color='blue', alpha=0.1, label='Initial Seed Window')
    plt.title('Accumulated Global Relative $L_2$ Error Over Temporal Horizon', fontsize=11)
    plt.xlabel('Temporal Frame Step Index ($t$)')
    plt.ylabel('Relative Error Amplitude $\\epsilon(t)$')
    plt.xlim(0, (max_steps - 1))
    plt.ylim(0, max(1.1, (np.max(rel_errors) * 1.1)))
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def load_and_evaluate_geofno(checkpoint_path, h5_file_path, target_idx=45):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Loading environment... Device: {device.type.upper()}')
    if (not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(f'Checkpoint missing at: {checkpoint_path}')
    eval_dataset = ChunkedTorusDataset(h5_file_path, t_in=3, t_out=30)
    model = GeoFNO(modes=20, width=48, t_in=3, t_out=30, geom_channels=3, n_theta=256, n_phi=256)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if ('base_grid' in state_dict):
        del state_dict['base_grid']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    evaluator = GeoFNOEvaluator(model, eval_dataset, t_in=3, t_out=30, device=device)
    visualize_geofno_rollout(evaluator, rollout_idx=target_idx, max_steps=295)

def generate_geofno_simulation(checkpoint_path, num_steps=1000, record_every=10):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
    print(('=' * 80))
    print(f'CHORUS FRAMEWORK: MULTI-PULSE GEO-FNO SURROGATE [{device.type.upper()}]')
    print(('=' * 80))
    if (not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(f'Missing checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    meta = checkpoint.get('dataset_configuration', {})
    T_IN = meta.get('t_in', 3)
    T_OUT = meta.get('t_out', 30)
    R_torus = meta.get('R', 3.0)
    r_torus = meta.get('r', 1.0)
    TRAIN_RES = meta.get('N_theta', 64)
    TARGET_RES = 64
    SCALE_FACTOR = (TARGET_RES / TRAIN_RES)
    base_record_every = 10
    sync_record_every = int((base_record_every * SCALE_FACTOR))
    sync_num_steps = int((num_steps * SCALE_FACTOR))
    simulator = TorusAcousticSimulator(R=R_torus, r=r_torus, N_theta=TARGET_RES, N_phi=TARGET_RES, c=1.0)
    model = GeoFNO(modes=16, width=64, t_in=T_IN, t_out=T_OUT, geom_channels=3, n_theta=TARGET_RES, n_phi=TARGET_RES).to(device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if ('base_grid' in state_dict):
        del state_dict['base_grid']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    (theta1, phi1) = (np.random.uniform(0, (2 * np.pi)), np.random.uniform(0, (2 * np.pi)))
    (theta2, phi2) = (np.random.uniform(0, (2 * np.pi)), np.random.uniform(0, (2 * np.pi)))
    t0_1 = np.random.uniform(0.01, 0.03)
    t0_2 = np.random.uniform(0.02, 0.05)
    grid_scale = ((64.0 / TARGET_RES) ** 2)
    safe_amplitude = (0.5 * grid_scale)

    def dual_random_source_fn(t, dev):
        amp = (torch.tensor([1.0], dtype=torch.float32, device=dev) * safe_amplitude)
        pulse1 = simulator.generate_kicker_pulse(t, t0=t0_1, sigma_t=0.01, theta0=theta1, phi0=phi1, sigma_s=0.12, amplitude=amp, device=dev)
        pulse2 = simulator.generate_kicker_pulse(t, t0=t0_2, sigma_t=0.01, theta0=theta2, phi0=phi2, sigma_s=0.1, amplitude=amp, device=dev)
        return (pulse1 + pulse2)
    print(f'Injection 1: (θ={theta1:.2f}, φ={phi1:.2f}) at t={t0_1:.3f}')
    print(f'Injection 2: (θ={theta2:.2f}, φ={phi2:.2f}) at t={t0_2:.3f}')
    print('Computing rigorous numerical ground truth...')
    (P_raw_stack, S_raw_stack) = simulator.simulate(num_steps=sync_num_steps, source_generator_fn=dual_random_source_fn, device=device, record_every=sync_record_every)
    print(f'Synchronizing numerical clock to Neural Operator... (Recording every {sync_record_every} steps)')
    P_raw = P_raw_stack[(0, :, 0)].cpu()
    S_raw = S_raw_stack[(0, :, 0)].cpu()
    p_scale = torch.clamp(torch.max(torch.abs(P_raw)), min=0.0001)
    s_scale = torch.clamp(torch.max(torch.abs(S_raw)), min=0.0001)
    P_norm = (P_raw / p_scale)
    S_norm = (S_raw / s_scale)
    theta_grid = torch.linspace(0, (2 * np.pi), (TARGET_RES + 1))[:(- 1)]
    phi_grid = torch.linspace(0, (2 * np.pi), (TARGET_RES + 1))[:(- 1)]
    (THETA, PHI) = torch.meshgrid(theta_grid, phi_grid, indexing='ij')
    metric = (r_torus * (R_torus + (r_torus * torch.cos(THETA))))
    m_norm = ((metric - metric.min()) / (metric.max() - metric.min()))
    geom_features = torch.stack([m_norm, (THETA / (2 * np.pi)), (PHI / (2 * np.pi))], dim=0).float().unsqueeze(0).to(device)
    print(f'Initiating Neural Operator integration... (Mapping T_in={T_IN} to T_out={T_OUT})')
    max_steps = P_norm.shape[0]
    p_curr = P_norm[0:T_IN].unsqueeze(0).to(device)
    predictions = [p_curr.cpu()]
    current_t = T_IN
    with torch.no_grad():
        with torch.amp.autocast(device.type, enabled=(device.type == 'cuda')):
            while (current_t < max_steps):
                s_start = (current_t - T_IN)
                s_end = current_t
                if (s_end <= S_norm.shape[0]):
                    s_curr = S_norm[s_start:s_end].unsqueeze(0).to(device)
                else:
                    s_curr = torch.zeros_like(p_curr)
                p_next = model(p_curr.float(), s_curr.float(), geom_features)
                predictions.append(p_next.cpu())
                p_curr = p_next[(:, (- T_IN):)]
                current_t += T_OUT
    P_pred_norm = torch.cat(predictions, dim=1).squeeze(0)
    P_pred_norm = P_pred_norm[:max_steps]
    P_pred_phys = (P_pred_norm * p_scale).detach().numpy()
    P_true_phys = P_raw.detach().numpy()
    t_indices = [(T_IN - 1), (T_IN + (T_OUT // 2)), ((T_IN + T_OUT) + 5), min((max_steps - 1), (T_IN + (2 * T_OUT)))]
    num_snapshots = len(t_indices)
    (fig, axes) = plt.subplots(3, num_snapshots, figsize=((3.8 * num_snapshots), 9.5))
    abs_error = np.abs((P_pred_phys - P_true_phys))
    for (i, t) in enumerate(t_indices):
        if (t >= max_steps):
            continue
        true_field = P_true_phys[t]
        pred_field = P_pred_phys[t]
        err_field = abs_error[t]
        vmax = (max(np.max(np.abs(true_field)), np.max(np.abs(pred_field))) + 1e-09)
        vmin = (- vmax)
        axes[(0, i)].imshow(true_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        axes[(0, i)].set_title(f'Spectral Ground Truth (t={t})', fontsize=10)
        axes[(0, i)].axis('off')
        axes[(1, i)].imshow(pred_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        axes[(1, i)].set_title((f'Geo-FNO Field (t={t})' if (t >= T_IN) else f'Identity Seed (t={t})'), fontsize=10)
        axes[(1, i)].axis('off')
        im_err = axes[(2, i)].imshow(err_field, cmap='magma', vmin=0, vmax=(np.max(err_field) + 1e-09), aspect='auto')
        axes[(2, i)].set_title(f'Absolute Residual (t={t})', fontsize=10)
        axes[(2, i)].axis('off')
    fig.colorbar(im_err, ax=axes[(2, :)].ravel().tolist(), shrink=0.8, label='Error Amplitude')
    plt.suptitle(f'Non-Euclidean Surrogate Validation: Interference State Space (Mesh: {TARGET_RES}²)', fontsize=13)
    plt.show()

def resume_geofno_pipeline(checkpoint_path='./best_geofno.pt', h5_path='torus_simulation_data_multipulse.h5', save_dest='./best_geofno_resumed.pt', target_epochs=100):
    '\n    Resumes Geo-FNO optimization using the enterprise-grade FastTrainer API.\n    '
    T_IN = 3
    T_OUT = 30
    BATCH_SIZE = 8
    RESUME_LR = 0.009
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Initializing resumption pipeline on {device.type.upper()}...')
    master_dataset = ChunkedTorusDataset(h5_path, t_in=T_IN, t_out=T_OUT)
    total_rollouts = master_dataset.num_rollouts
    train_rollouts = int((0.8 * total_rollouts))
    gen = torch.Generator().manual_seed(32)
    indices = torch.randperm(total_rollouts, generator=gen).tolist()
    train_idx = indices[:train_rollouts]
    val_idx = indices[train_rollouts:]
    valid_starts = master_dataset.valid_starts
    train_window_indices = [((r * valid_starts) + t) for r in train_idx for t in range(valid_starts)]
    val_window_indices = [((r * valid_starts) + t) for r in val_idx for t in range(valid_starts)]
    train_set = Subset(master_dataset, train_window_indices)
    val_set = Subset(master_dataset, val_window_indices)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=2)
    print(f'Data Partitioned: {len(train_window_indices)} Train Chunks | {len(val_window_indices)} Val Chunks')
    model = GeoFNO(modes=20, width=48, t_in=T_IN, t_out=T_OUT, geom_channels=3, n_theta=master_dataset.N_theta, n_phi=master_dataset.N_phi)
    metadata_payload = {'project': 'CHORUS_Acoustic_Surrogate', 't_in': T_IN, 't_out': T_OUT, 'N_theta': master_dataset.N_theta, 'N_phi': master_dataset.N_phi, 'R': master_dataset.R, 'r': master_dataset.r}
    trainer = FastTrainer(model, lr=RESUME_LR)
    trainer.load_checkpoint(checkpoint_path)
    trainer.fit(train_loader=train_loader, val_loader=val_loader, total_target_epochs=target_epochs, dataset_meta=metadata_payload, save_best_path=save_dest, save_every=50)

def export_and_quantize_fno(checkpoint_path='./best_geofno_resumed.pt'):
    print(('=' * 80))
    print('🚀 CHORUS FRAMEWORK: ONNX COMPRESSION ENGINE')
    print(('=' * 80))
    device = torch.device('cpu')
    T_IN = 3
    T_OUT = 30
    HI_RES = 64
    print('1. Initializing GeoFNO (151M Params)...')
    model = GeoFNO(modes=24, width=128, t_in=T_IN, t_out=T_OUT, geom_channels=3, n_theta=HI_RES, n_phi=HI_RES).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if ('base_grid' in state_dict):
        del state_dict['base_grid']
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print('2. Generating dynamic tracer inputs...')
    dummy_p_in = torch.randn(1, T_IN, HI_RES, HI_RES, device=device)
    dummy_s_in = torch.randn(1, T_IN, HI_RES, HI_RES, device=device)
    dummy_geom = torch.randn(1, 3, HI_RES, HI_RES, device=device)
    onnx_fp32_path = 'geofno_base_fp32.onnx'
    print(f'3. Compiling computational graph to {onnx_fp32_path} (Opset 17)...')
    torch.onnx.export(model, (dummy_p_in, dummy_s_in, dummy_geom), onnx_fp32_path, export_params=True, opset_version=17, do_constant_folding=True, input_names=['p_in', 's_in', 'geom_features'], output_names=['p_out'], dynamic_axes={'p_in': {0: 'batch_size'}, 's_in': {0: 'batch_size'}, 'geom_features': {0: 'batch_size'}, 'p_out': {0: 'batch_size'}})
    onnx_fp16_path = 'geofno_quantized_fp16.onnx'
    print(f'4. Applying FP16 Quantization -> {onnx_fp16_path}')
    onnx_model = onnx.load(onnx_fp32_path)
    onnx_model_fp16 = float16.convert_float_to_float16(onnx_model)
    onnx.save(onnx_model_fp16, onnx_fp16_path)
    onnx_int8_path = 'geofno_quantized_int8.onnx'
    print(f'5. Applying INT8 Dynamic Quantization -> {onnx_int8_path}')
    quantize_dynamic(model_input=onnx_fp32_path, model_output=onnx_int8_path, weight_type=QuantType.QUInt8)
    print(('\n' + ('=' * 80)))
    print('📊 COMPRESSION RESULTS')
    print(('=' * 80))

    def get_size_mb(filepath):
        return (os.path.getsize(filepath) / (1024 * 1024))
    fp32_size = get_size_mb(onnx_fp32_path)
    fp16_size = get_size_mb(onnx_fp16_path)
    int8_size = get_size_mb(onnx_int8_path)
    print(f'PyTorch Checkpoint (Includes Optimizer) : ~1,730.00 MB')
    print(f'ONNX FP32 (Bare Model)                  : {fp32_size:.2f} MB')
    print(f'ONNX FP16 (Half Precision)              : {fp16_size:.2f} MB  (Compression: {(fp32_size / fp16_size):.2f}x)')
    print(f'ONNX INT8 (Integer Precision)           : {int8_size:.2f} MB  (Compression: {(fp32_size / int8_size):.2f}x)')
    print(('=' * 80))

def generate_high_res_simulation(checkpoint_path, num_steps=400, record_every=10):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
    print(('=' * 80))
    print(f'MULTI-PULSE INTERFERENCE ROLLOUT: [{device.type.upper()}]')
    print(('=' * 80))
    if (not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(f'Missing checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FNO2d(modes=12, width=32, in_channels=4, out_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    HI_RES = 64
    simulator = TorusAcousticSimulator(R=3.0, r=1.0, N_theta=HI_RES, N_phi=HI_RES, c=1.0)
    (theta1, phi1) = (np.random.uniform(0, (2 * np.pi)), np.random.uniform(0, (2 * np.pi)))
    (theta2, phi2) = (np.random.uniform(0, (2 * np.pi)), np.random.uniform(0, (2 * np.pi)))
    t0_1 = np.random.uniform(0.01, 0.03)
    t0_2 = np.random.uniform(0.02, 0.05)
    grid_scale = ((64.0 / HI_RES) ** 2)
    safe_amplitude = (0.5 * grid_scale)

    def dual_random_source_fn(t, dev):
        amp = (torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=dev) * safe_amplitude)
        pulse1 = simulator.generate_kicker_pulse(t, t0=t0_1, sigma_t=0.01, theta0=theta1, phi0=phi1, sigma_s=0.12, amplitude=amp, device=dev)
        pulse2 = simulator.generate_kicker_pulse(t, t0=t0_2, sigma_t=0.01, theta0=theta2, phi0=phi2, sigma_s=0.1, amplitude=amp, device=dev)
        return (pulse1 + pulse2)
    print(f'Source 1: (θ={theta1:.2f}, φ={phi1:.2f}) at t={t0_1:.3f}')
    print(f'Source 2: (θ={theta2:.2f}, φ={phi2:.2f}) at t={t0_2:.3f}')
    print('Running numerical reference solver with dual pulses...')
    (P_raw_stack, S_raw_stack) = simulator.simulate(num_steps=num_steps, source_generator_fn=dual_random_source_fn, device=device, record_every=record_every)
    P_raw = P_raw_stack[(0, :, 0:1)].cpu()
    S_raw = S_raw_stack[(0, :, 0:1)].cpu()
    p_mean = checkpoint['p_mean'].cpu()
    p_std = checkpoint['p_std'].cpu()
    s_mean = checkpoint['s_mean'].cpu()
    s_std = checkpoint['s_std'].cpu()
    P_norm = ((P_raw - p_mean) / p_std)
    S_norm = ((S_raw - s_mean) / s_std)
    theta_grid = torch.linspace(0, (2 * np.pi), (HI_RES + 1))[:(- 1)]
    phi_grid = torch.linspace(0, (2 * np.pi), (HI_RES + 1))[:(- 1)]
    (THETA, _) = torch.meshgrid(theta_grid, phi_grid, indexing='ij')
    metric = (simulator.solver.r * (simulator.solver.R + (simulator.solver.r * torch.cos(THETA))))
    m_min = (simulator.solver.r * (simulator.solver.R - simulator.solver.r))
    m_max = (simulator.solver.r * (simulator.solver.R + simulator.solver.r))
    metric_norm = ((metric - m_min) / (m_max - m_min))
    m_static = metric_norm.unsqueeze(0).unsqueeze(0).to(device)
    print('Executing unassisted autoregressive model unrolling...')
    max_steps = P_norm.shape[0]
    p_prev = P_norm[0].unsqueeze(0).to(device)
    p_curr = P_norm[1].unsqueeze(0).to(device)
    predictions = [p_prev.cpu(), p_curr.cpu()]
    with torch.no_grad():
        for t in range(2, max_steps):
            s_curr = S_norm[(t - 1)].unsqueeze(0).to(device)
            x_in = torch.cat([p_curr, p_prev, s_curr, m_static], dim=1)
            if (t == 2):
                print(f'DIAGNOSTIC (t=2) -> Max S_curr input: {s_curr.max().item():.4f}')
                print(f'DIAGNOSTIC (t=2) -> Max P_curr input: {p_curr.max().item():.4f}')
            p_next = model(x_in)
            predictions.append(p_next.cpu())
            p_prev = p_curr
            p_curr = p_next
    P_pred_norm = torch.cat(predictions, dim=0)
    P_pred_phys = ((P_pred_norm * p_std) + p_mean).detach().numpy()
    P_true_phys = P_raw.detach().numpy()
    t_indices = [0, 1, 2, 30]
    num_snapshots = len(t_indices)
    (fig, axes) = plt.subplots(3, num_snapshots, figsize=((3.8 * num_snapshots), 9.5))
    abs_error = np.abs((P_pred_phys - P_true_phys))
    for (i, t) in enumerate(t_indices):
        true_field = P_true_phys[(t, 0)]
        pred_field = P_pred_phys[(t, 0)]
        err_field = abs_error[(t, 0)]
        vmax = (max(np.max(np.abs(true_field)), np.max(np.abs(pred_field))) + 1e-09)
        vmin = (- vmax)
        axes[(0, i)].imshow(true_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        axes[(0, i)].set_title(f'Spectral Solver (t={t})', fontsize=10)
        axes[(0, i)].axis('off')
        axes[(1, i)].imshow(pred_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        axes[(1, i)].set_title((f'FNO Prediction (t={t})' if (t > 1) else f'Identity Feed (t={t})'), fontsize=10)
        axes[(1, i)].axis('off')
        im_err = axes[(2, i)].imshow(err_field, cmap='magma', vmin=0, vmax=(np.max(err_field) + 1e-09), aspect='auto')
        axes[(2, i)].set_title(f'Absolute Error (t={t})', fontsize=10)
        axes[(2, i)].axis('off')
    fig.colorbar(im_err, ax=axes[(2, :)].ravel().tolist(), shrink=0.8, label='Error Amplitude')
    plt.suptitle(f'Multi-Pulse Interference Simulation (Mesh: {HI_RES}²)', fontsize=13)
    plt.show()

def generate_single_pulse_simulation(checkpoint_path, num_steps=400, record_every=10):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
    print(('=' * 80))
    print(f'SINGLE-PULSE ROLLOUT EVALUATION: [{device.type.upper()}]')
    print(('=' * 80))
    if (not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(f'Missing checkpoint: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = FNO2d(modes=12, width=32, in_channels=4, out_channels=1).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    HI_RES = 64
    simulator = TorusAcousticSimulator(R=3.0, r=1.0, N_theta=HI_RES, N_phi=HI_RES, c=1.0)
    (theta1, phi1) = (np.random.uniform(0, (2 * np.pi)), np.random.uniform(0, (2 * np.pi)))
    t0_1 = np.random.uniform(0.01, 0.04)
    grid_scale = ((64.0 / HI_RES) ** 2)

    def single_random_source_fn(t, dev):
        amp = (torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=dev) * grid_scale)
        return simulator.generate_kicker_pulse(t, t0=t0_1, sigma_t=0.01, theta0=theta1, phi0=phi1, sigma_s=0.12, amplitude=amp, device=dev)
    print(f'Source 1: (θ={theta1:.2f}, φ={phi1:.2f}) at t={t0_1:.3f}')
    print('Running numerical reference solver with a single pulse...')
    (P_raw_stack, S_raw_stack) = simulator.simulate(num_steps=num_steps, source_generator_fn=single_random_source_fn, device=device, record_every=record_every)
    P_raw = P_raw_stack[(0, :, 0:1)].cpu()
    S_raw = S_raw_stack[(0, :, 0:1)].cpu()
    p_mean = checkpoint['p_mean'].cpu()
    p_std = checkpoint['p_std'].cpu()
    s_mean = checkpoint['s_mean'].cpu()
    s_std = checkpoint['s_std'].cpu()
    P_norm = ((P_raw - p_mean) / p_std)
    S_norm = ((S_raw - s_mean) / s_std)
    theta_grid = torch.linspace(0, (2 * np.pi), (HI_RES + 1))[:(- 1)]
    phi_grid = torch.linspace(0, (2 * np.pi), (HI_RES + 1))[:(- 1)]
    (THETA, _) = torch.meshgrid(theta_grid, phi_grid, indexing='ij')
    metric = (simulator.solver.r * (simulator.solver.R + (simulator.solver.r * torch.cos(THETA))))
    m_min = (simulator.solver.r * (simulator.solver.R - simulator.solver.r))
    m_max = (simulator.solver.r * (simulator.solver.R + simulator.solver.r))
    metric_norm = ((metric - m_min) / (m_max - m_min))
    m_static = metric_norm.unsqueeze(0).unsqueeze(0).to(device)
    print('Executing unassisted autoregressive model unrolling...')
    max_steps = P_norm.shape[0]
    p_prev = P_norm[0].unsqueeze(0).to(device)
    p_curr = P_norm[1].unsqueeze(0).to(device)
    predictions = [p_prev.cpu(), p_curr.cpu()]
    with torch.no_grad():
        for t in range(2, max_steps):
            s_curr = S_norm[(t - 1)].unsqueeze(0).to(device)
            x_in = torch.cat([p_curr, p_prev, s_curr, m_static], dim=1)
            if (t == 2):
                print(f'DIAGNOSTIC (t=2) -> Max S_curr input: {s_curr.max().item():.4f}')
                print(f'DIAGNOSTIC (t=2) -> Max P_curr input: {p_curr.max().item():.4f}')
            p_next = model(x_in)
            predictions.append(p_next.cpu())
            p_prev = p_curr
            p_curr = p_next
    P_pred_norm = torch.cat(predictions, dim=0)
    P_pred_phys = ((P_pred_norm * p_std) + p_mean).detach().numpy()
    P_true_phys = P_raw.detach().numpy()
    t_indices = [1, 2, 20, 30]
    num_snapshots = len(t_indices)
    (fig, axes) = plt.subplots(3, num_snapshots, figsize=((3.8 * num_snapshots), 9.5))
    abs_error = np.abs((P_pred_phys - P_true_phys))
    for (i, t) in enumerate(t_indices):
        true_field = P_true_phys[(t, 0)]
        pred_field = P_pred_phys[(t, 0)]
        err_field = abs_error[(t, 0)]
        vmax = (max(np.max(np.abs(true_field)), np.max(np.abs(pred_field))) + 1e-09)
        vmin = (- vmax)
        axes[(0, i)].imshow(true_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        axes[(0, i)].set_title(f'Spectral Solver (t={t})', fontsize=10)
        axes[(0, i)].axis('off')
        axes[(1, i)].imshow(pred_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        axes[(1, i)].set_title((f'FNO Prediction (t={t})' if (t > 1) else f'Identity Feed (t={t})'), fontsize=10)
        axes[(1, i)].axis('off')
        im_err = axes[(2, i)].imshow(err_field, cmap='magma', vmin=0, vmax=(np.max(err_field) + 1e-09), aspect='auto')
        axes[(2, i)].set_title(f'Absolute Error (t={t})', fontsize=10)
        axes[(2, i)].axis('off')
    fig.colorbar(im_err, ax=axes[(2, :)].ravel().tolist(), shrink=0.8, label='Error Amplitude')
    plt.suptitle(f'Single-Pulse Simulation (Mesh: {HI_RES}²)', fontsize=13)
    plt.show()

def extend_training_pipeline(checkpoint_path, h5_file_path, additional_epochs=500, save_every=500):
    '\n    Loads a saved checkpoint, restores the training history and normalization bounds,\n    and extends the optimization path for additional epochs with periodic checkpointing.\n    '
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
    print(('=' * 75))
    print(f'RESUMING OPERATOR OPTIMIZATION PATH ON: {device.type.upper()}')
    print(('=' * 75))
    if (not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(f'Checkpoint archive missing at: {checkpoint_path}')
    print(f'Unpacking checkpoint parameters from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print('Re-indexing dataset and mapping training statistics...')
    master_dataset = TorusWaveDataset(h5_file_path, seq_len=51, transform=False)
    stat_targets = [('p_mean', master_dataset.P, 'mean'), ('p_std', master_dataset.P, 'std'), ('s_mean', master_dataset.S, 'mean'), ('s_std', master_dataset.S, 'std')]
    for (attr_name, source_tensor, op_type) in stat_targets:
        val = checkpoint.get(attr_name, None)
        if (val is not None):
            if torch.is_tensor(val):
                setattr(master_dataset, attr_name, val.cpu())
            else:
                setattr(master_dataset, attr_name, torch.tensor(val, dtype=torch.float32))
        else:
            print(f"[INFO]: '{attr_name}' missing from checkpoint target. Reconstructing from source on host...")
            if (op_type == 'mean'):
                calculated_val = source_tensor.mean()
            else:
                calculated_val = torch.clamp(source_tensor.std(), min=1e-08)
            setattr(master_dataset, attr_name, calculated_val)
    master_dataset.P = ((master_dataset.P - master_dataset.p_mean) / master_dataset.p_std)
    master_dataset.S = ((master_dataset.S - master_dataset.s_mean) / master_dataset.s_std)
    print('Dataset normalization metrics aligned successfully.')
    SPLIT_RATIO = 0.8
    total_sequences = len(master_dataset)
    train_size = int((SPLIT_RATIO * total_sequences))
    val_size = (total_sequences - train_size)
    (train_set, val_set) = random_split(master_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False, drop_last=False)
    model = FNO2d(modes=12, width=32, in_channels=4, out_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    trainer = DataDrivenTrainer(model, lr=0.0001, weight_decay=1e-05)
    if (('history' in checkpoint) and (checkpoint['history'] is not None)):
        trainer.history = checkpoint['history']
        previous_epochs = len(trainer.history['train'])
        print(f'Archived history parsed successfully. Found {previous_epochs} completed epochs.')
    else:
        trainer.history = {'train': [], 'val': []}
        previous_epochs = 0
        print('[WARNING]: No history found in checkpoint. Initializing blank metrics.')
    if (('optimizer_state_dict' in checkpoint) and (checkpoint['optimizer_state_dict'] is not None)):
        print('Restoring AdamW running moments (m_t, v_t) to ensure gradient continuity...')
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("[CRITICAL NOTICE]: 'optimizer_state_dict' was not found or was empty.")
        print('The optimization path will re-initialize smoothly, but expect a momentary')
        print('loss perturbation during the initial adjustment iterations.')
    print(f'''
Launching optimization extension for +{additional_epochs} epochs (Total Horizon: {(previous_epochs + additional_epochs)})...''')
    print(('-' * 75))
    extended_history = trainer.train_epochs(train_loader=train_loader, val_loader=val_loader, epochs=additional_epochs, initial_teacher_forcing=0.0, save_every=save_every, checkpoint_base_path=checkpoint_path, dataset=master_dataset)
    print(('-' * 75))
    plt.figure(figsize=(10, 5))
    plt.plot(extended_history['train'], label='Extended Train Loss', color='#1f77b4', lw=2)
    if extended_history['val']:
        plt.plot(extended_history['val'], label='Extended Val Loss', color='#ff7f0e', linestyle='--', lw=2)
    if (previous_epochs > 0):
        plt.axvline(x=previous_epochs, color='purple', linestyle=':', alpha=0.8, lw=2, label=f'Warm Restart Checkpoint (Epoch {previous_epochs})')
    plt.yscale('log')
    plt.title('Continuous Fourier Operator Optimization History (Extended Training Sequence)', fontsize=11)
    plt.xlabel('Total Consolidated Epochs')
    plt.ylabel('Mean Squared Error (Log Grid)')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    torch.save({'epoch': (previous_epochs + additional_epochs), 'model_state_dict': model.state_dict(), 'optimizer_state_dict': trainer.optimizer.state_dict(), 'history': extended_history, 'p_mean': master_dataset.p_mean, 'p_std': master_dataset.p_std, 's_mean': master_dataset.s_mean, 's_std': master_dataset.s_std}, checkpoint_path)
    print(f'''Consolidated parameters updated successfully at destination: {checkpoint_path}
''')

def run_wno_pipeline(h5_file_path):
    BATCH_SIZE = 16
    SEQ_LEN = 51
    EPOCHS = 2000
    LEARNING_RATE = 0.0007
    WEIGHT_DECAY = 1e-05
    SPLIT_RATIO = 0.8
    SAVE_EVERY = 500
    checkpoint_destination = './wno2d_operator_net.pt'
    print(('=' * 75))
    print('      NOMAD WORKFLOW: WAVELET NEURAL OPERATOR (WNO) DEPLOYMENT       ')
    print(('=' * 75))
    master_dataset = TorusWaveDataset(h5_file_path, seq_len=SEQ_LEN, transform=True)
    total_sequences = len(master_dataset)
    train_size = int((SPLIT_RATIO * total_sequences))
    val_size = (total_sequences - train_size)
    (train_set, val_set) = random_split(master_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    model = WNO2d(width=64, in_channels=4, out_channels=1)
    trainer = DataDrivenTrainer(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print('\nExecuting Localized Wavelet Multi-Step Minimizations...')
    history = trainer.train_epochs(train_loader=train_loader, val_loader=val_loader, epochs=EPOCHS, checkpoint_base_path=checkpoint_destination, initial_teacher_forcing=0.5, dataset=master_dataset, save_every=SAVE_EVERY)
    plot_pde_convergence(history)
    torch.save({'epoch': EPOCHS, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': trainer.optimizer.state_dict(), 'history': history, 'p_mean': master_dataset.p_mean, 'p_std': master_dataset.p_std, 's_mean': master_dataset.s_mean, 's_std': master_dataset.s_std}, checkpoint_destination)
    print(f'''Pipeline complete. Operator weights archived at: {checkpoint_destination}
''')

def extend_wno_pipeline(checkpoint_path, h5_file_path, additional_epochs=2000, save_every=500):
    '\n    Loads a WNO2d checkpoint, restores the AdamW optimizer phase space,\n    aligns the dataset manifold, and extends the training horizon.\n    '
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
    print(('=' * 80))
    print(f'INITIATING WARM RESTART EXTENSION ON: [{device.type.upper()}]')
    print(('=' * 80))
    if (not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(f'Checkpoint archive missing at: {checkpoint_path}')
    print(f'Unpacking checkpoint parameters from {checkpoint_path}...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print('Re-indexing dataset and mapping historical statistics...')
    master_dataset = TorusWaveDataset(h5_file_path, seq_len=51, transform=False)
    master_dataset.p_mean = checkpoint['p_mean'].cpu()
    master_dataset.p_std = checkpoint['p_std'].cpu()
    master_dataset.s_mean = checkpoint['s_mean'].cpu()
    master_dataset.s_std = checkpoint['s_std'].cpu()
    master_dataset.P = ((master_dataset.P - master_dataset.p_mean) / master_dataset.p_std)
    master_dataset.S = ((master_dataset.S - master_dataset.s_mean) / master_dataset.s_std)
    SPLIT_RATIO = 0.8
    total_sequences = len(master_dataset)
    train_size = int((SPLIT_RATIO * total_sequences))
    val_size = (total_sequences - train_size)
    (train_set, val_set) = random_split(master_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    BATCH_SIZE = 16
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    model = WNO2d(width=64, in_channels=4, out_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    trainer = DataDrivenTrainer(model, lr=0.0007, weight_decay=1e-05)
    if (('history' in checkpoint) and (checkpoint['history'] is not None)):
        trainer.history = checkpoint['history']
        previous_epochs = len(trainer.history['train'])
        print(f'Archived history parsed. Resuming from Epoch {previous_epochs}.')
    else:
        trainer.history = {'train': [], 'val': []}
        previous_epochs = 0
    if (('optimizer_state_dict' in checkpoint) and (checkpoint['optimizer_state_dict'] is not None)):
        print('Restoring AdamW running moments to ensure gradient trajectory continuity...')
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        print("[WARNING]: 'optimizer_state_dict' missing. Expect a loss perturbation spike.")
    print(f'''
Launching optimization extension for +{additional_epochs} epochs (Total Target: {(previous_epochs + additional_epochs)})...''')
    print(('-' * 75))
    extended_history = trainer.train_epochs(train_loader=train_loader, val_loader=val_loader, epochs=additional_epochs, initial_teacher_forcing=0.0, save_every=save_every, checkpoint_base_path=checkpoint_path, dataset=master_dataset)
    print(('-' * 75))
    plt.figure(figsize=(10, 5))
    plt.plot(extended_history['train'], label='Extended Train Loss', color='#1f77b4', lw=2)
    if extended_history['val']:
        plt.plot(extended_history['val'], label='Extended Val Loss', color='#ff7f0e', linestyle='--', lw=2)
    if (previous_epochs > 0):
        plt.axvline(x=previous_epochs, color='purple', linestyle=':', alpha=0.8, lw=2, label=f'Warm Restart (Epoch {previous_epochs})')
    plt.yscale('log')
    plt.title('Continuous Wavelet Operator Optimization (Pushforward + Spectral Loss)', fontsize=11)
    plt.xlabel('Total Consolidated Epochs')
    plt.ylabel('Sobolev Norm Error (Log Scale)')
    plt.legend(loc='upper right')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    total_epochs_completed = (previous_epochs + additional_epochs)
    torch.save({'epoch': total_epochs_completed, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': trainer.optimizer.state_dict(), 'history': extended_history, 'p_mean': master_dataset.p_mean, 'p_std': master_dataset.p_std, 's_mean': master_dataset.s_mean, 's_std': master_dataset.s_std}, checkpoint_path)
    print(f'''Extended state preserved successfully at: {checkpoint_path}
''')

def visualize_wno_rollout(evaluator, rollout_idx=45, max_steps=51, num_snapshots=5):
    '\n    Generates side-by-side matrix comparisons and displays error accumulation over time.\n    '
    (P_true, P_pred) = evaluator.generate_rollout(rollout_idx=rollout_idx, max_steps=max_steps)
    P_true = evaluator.dataset.denormalize_p(P_true).numpy()
    P_pred = evaluator.dataset.denormalize_p(P_pred).numpy()
    abs_error = np.abs((P_pred - P_true))
    t_indices = np.linspace(2, (max_steps - 1), num_snapshots, dtype=int)
    (fig, axes) = plt.subplots(3, num_snapshots, figsize=((4 * num_snapshots), 9.5))
    for (i, t) in enumerate(t_indices):
        true_field = P_true[(t, 0)]
        pred_field = P_pred[(t, 0)]
        err_field = abs_error[(t, 0)]
        vmax = (np.max(np.abs(true_field)) + 1e-09)
        vmin = (- vmax)
        ax_true = axes[(0, i)]
        im_true = ax_true.imshow(true_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        ax_true.set_title(f'Spectral Target (t={t})', fontsize=10)
        ax_true.axis('off')
        ax_pred = axes[(1, i)]
        im_pred = ax_pred.imshow(pred_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        ax_pred.set_title(f'WNO2d Prediction (t={t})', fontsize=10)
        ax_pred.axis('off')
        ax_err = axes[(2, i)]
        err_max = (np.max(err_field) + 1e-09)
        im_err = ax_err.imshow(err_field, cmap='magma', vmin=0, vmax=err_max, aspect='auto')
        ax_err.set_title(f'Absolute Error (t={t})', fontsize=10)
        ax_err.axis('off')
    fig.colorbar(im_true, ax=axes[(0:2, :)].ravel().tolist(), shrink=0.75, label='Pressure Coordinate Field Magnitude')
    fig.colorbar(im_err, ax=axes[(2, :)].ravel().tolist(), shrink=0.75, label='Absolute Error Intensity')
    plt.suptitle('Unassisted Autoregressive Rollout Evaluation: Spectral Solver vs WNO2d Wavelet Space', fontsize=14, y=0.98)
    plt.show()
    rel_errors = evaluator.compute_relative_l2_error(torch.tensor(P_true), torch.tensor(P_pred))
    plt.figure(figsize=(10, 3.5))
    plt.plot(rel_errors, color='#d62728', linewidth=2.5, label='Relative $L_2$ Error Deviation')
    plt.axhline(0.1, color='gray', linestyle=':', alpha=0.7, label='10% Bound Tolerance')
    plt.title('Accumulated Global Relative $L_2$ Error Matrix Norm Over Temporal Horizon', fontsize=11)
    plt.xlabel('Temporal Frame Step Index ($t$)')
    plt.ylabel('Relative Error Amplitude $\\epsilon(t)$')
    plt.xlim(0, (max_steps - 1))
    plt.ylim(0, max(1.1, (np.max(rel_errors) * 1.1)))
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def load_and_compare_wno(checkpoint_path, h5_file_path, target_idx=45):
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f'Loading environment settings... Device set to: {device.type.upper()}')
    if (not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(f'Target optimization weights missing at destination: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = WNO2d(width=64, in_channels=4, out_channels=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    eval_dataset = TorusWaveDataset(h5_file_path, seq_len=51, transform=False)
    eval_dataset.p_mean = (checkpoint.get('p_mean', eval_dataset.P.mean()).cpu() if torch.is_tensor(checkpoint.get('p_mean')) else checkpoint.get('p_mean'))
    eval_dataset.p_std = (checkpoint.get('p_std', eval_dataset.P.std()).cpu() if torch.is_tensor(checkpoint.get('p_std')) else checkpoint.get('p_std'))
    eval_dataset.s_mean = (checkpoint.get('s_mean', eval_dataset.S.mean()).cpu() if torch.is_tensor(checkpoint.get('s_mean')) else checkpoint.get('s_mean'))
    eval_dataset.s_std = (checkpoint.get('s_std', eval_dataset.S.std()).cpu() if torch.is_tensor(checkpoint.get('s_std')) else checkpoint.get('s_std'))
    eval_dataset.P = ((eval_dataset.P - eval_dataset.p_mean) / eval_dataset.p_std)
    eval_dataset.S = ((eval_dataset.S - eval_dataset.s_mean) / eval_dataset.s_std)
    if (('history' in checkpoint) and (checkpoint['history'] is not None)):
        history = checkpoint['history']
        if (len(history.get('train', [])) > 0):
            plt.figure(figsize=(10, 3.5))
            plt.plot(history['train'], label='WNO2d Training History Tracking', color='#1f77b4', lw=2)
            if history.get('val'):
                plt.plot(history['val'], label='WNO2d Validation History Tracking', color='#ff7f0e', linestyle='--', lw=2)
            plt.yscale('log')
            plt.title(f"Wavelet Operator Optimization Path ({len(history['train'])} Epoch Total Horizon)")
            plt.xlabel('Training Epochs')
            plt.ylabel('Loss Magnitude (MSE/Sobolev Log Grid)')
            plt.legend()
            plt.grid(True, which='both', linestyle='--', alpha=0.4)
            plt.tight_layout()
            plt.show()
    evaluator = WNOEvaluator(model, eval_dataset, device=device)
    visualize_wno_rollout(evaluator, rollout_idx=target_idx, max_steps=51)
