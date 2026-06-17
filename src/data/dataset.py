import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
import os

class TorusWaveDataset(Dataset):
    """
    Manifold-Aware Dataset Engine for Acoustic Wave Operators.
    Extracts scalar target dimensions and injects spatial metric configurations.
    """
    def __init__(self, h5_path, seq_len=4, transform=True):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Simulation matrix missing at: {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            # Slicing [..., 0:1] reduces multi-channel vectors to rank-0 scalars (C=1)
            raw_P = torch.from_numpy(f['pressure'][:])[..., 0:1]
            raw_S = torch.from_numpy(f['source'][:])[..., 0:1]

            # Permute to PyTorch standard coordinate alignment: (B, T, C, H, W)
            self.P = raw_P.permute(0, 1, 4, 2, 3)
            self.S = raw_S.permute(0, 1, 4, 2, 3)

            self.R = f.attrs['R']
            self.r = f.attrs['r']
            self.N_theta = f.attrs['N_theta']
            self.N_phi = f.attrs['N_phi']

        self.seq_len = seq_len
        self.num_rollouts = self.P.shape[0]
        self.time_steps = self.P.shape[1]

        self.valid_starts_per_rollout = max(0, self.time_steps - self.seq_len)
        if self.valid_starts_per_rollout == 0:
            raise ValueError(f"Simulation steps ({self.time_steps}) insufficient for window ({self.seq_len})")

        self.transform = transform
        if self.transform:
            self._compute_stats()
            self._normalize()

        # Compute metric scaling transformations map
        theta_grid = torch.linspace(0, 2*np.pi, self.N_theta + 1)[:-1]
        phi_grid = torch.linspace(0, 2*np.pi, self.N_phi + 1)[:-1]
        THETA, _ = torch.meshgrid(theta_grid, phi_grid, indexing='ij')

        metric = self.r * (self.R + self.r * torch.cos(THETA))
        m_min = self.r * (self.R - self.r)
        m_max = self.r * (self.R + self.r)
        metric_norm = (metric - m_min) / (m_max - m_min)

        self.metric_embed = metric_norm.unsqueeze(0).unsqueeze(0).to(torch.float32)

    def _compute_stats(self):
        self.p_mean, self.p_std = self.P.mean(), torch.clamp(self.P.std(), min=1e-8)
        self.s_mean, self.s_std = self.S.mean(), torch.clamp(self.S.std(), min=1e-8)

    def _normalize(self):
        self.P = (self.P - self.p_mean) / self.p_std
        self.S = (self.S - self.s_mean) / self.s_std

    def denormalize_p(self, p_tensor):
        return p_tensor * self.p_std + self.p_mean

    def denormalize_s(self, s_tensor):
        return s_tensor * self.s_std + self.s_mean

    def __len__(self):
        return self.num_rollouts * self.valid_starts_per_rollout

    def __getitem__(self, idx):
        rollout_idx = idx // self.valid_starts_per_rollout
        t_start = idx % self.valid_starts_per_rollout
        t_end = t_start + self.seq_len

        s_seq = self.S[rollout_idx, t_start:t_end]
        p_seq = self.P[rollout_idx, t_start:t_end]
        m_seq = self.metric_embed.expand(self.seq_len, -1, -1, -1)

        return s_seq.float(), p_seq.float(), m_seq.float()


class ChunkedTorusDataset(Dataset):
    def __init__(self, h5_path, t_in=5, t_out=10):
        with h5py.File(h5_path, 'r') as f:
            # squeeze channel dimension
            self.P_raw = torch.from_numpy(f['pressure'][:])[..., 0].permute(0, 1, 3, 2)
            self.S_raw = torch.from_numpy(f['source'][:])[..., 0].permute(0, 1, 3, 2)
            self.R = f.attrs.get('R', 3.0)
            self.r = f.attrs.get('r', 1.0)
            self.N_theta = f.attrs.get('N_theta', 64)
            self.N_phi   = f.attrs.get('N_phi', 64)

        self.t_in = t_in
        self.t_out = t_out
        self.num_rollouts = self.P_raw.shape[0]
        self.time_steps = self.P_raw.shape[1]
        self.chunk_size = t_in + t_out
        self.valid_starts = self.time_steps - self.chunk_size

        # global max scaling (keeps inputs roughly in [-1, 1])
        self.p_scale = torch.clamp(torch.max(torch.abs(self.P_raw)), min=1e-4)
        self.s_scale = torch.clamp(torch.max(torch.abs(self.S_raw)), min=1e-4)

        # static geometry: normalised metric, theta, phi
        theta = torch.linspace(0, 2*np.pi, self.N_theta+1)[:-1]
        phi   = torch.linspace(0, 2*np.pi, self.N_phi+1)[:-1]
        THETA, PHI = torch.meshgrid(theta, phi, indexing='ij')
        metric = self.r * (self.R + self.r * torch.cos(THETA))
        m_norm = (metric - metric.min()) / (metric.max() - metric.min())
        self.geom_features = torch.stack([m_norm, THETA/(2*np.pi), PHI/(2*np.pi)], dim=0).float()

    def __len__(self):
        return self.num_rollouts * self.valid_starts

    def __getitem__(self, idx):
        rollout = idx // self.valid_starts
        t_start = idx % self.valid_starts

        # input and source history
        p_in = self.P_raw[rollout, t_start : t_start+self.t_in] / self.p_scale
        s_in = self.S_raw[rollout, t_start : t_start+self.t_in] / self.s_scale

        # target future pressure
        p_out = self.P_raw[rollout, t_start+self.t_in : t_start+self.chunk_size] / self.p_scale

        return p_in.float(), s_in.float(), self.geom_features, p_out.float()
