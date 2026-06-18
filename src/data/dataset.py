import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
import os
from typing import Tuple, Optional

class TorusWaveDataset(Dataset):
    """
    Manifold-Aware Dataset Engine for Acoustic Wave Operators.
    Extracts scalar target dimensions and injects spatial metric configurations.
    """
    def __init__(self, h5_path: str, seq_len: int = 4, transform: bool = True):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Simulation matrix missing at: {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            # Slicing [..., 0:1] reduces multi-channel vectors to rank-0 scalars (C=1)
            raw_P = torch.from_numpy(f['pressure'][:])[..., 0:1]
            raw_S = torch.from_numpy(f['source'][:])[..., 0:1]

            # Permute to PyTorch standard coordinate alignment: (B, T, C, H, W)
            self.P = raw_P.permute(0, 1, 4, 2, 3)
            self.S = raw_S.permute(0, 1, 4, 2, 3)

            self.R = float(f.attrs['R'])
            self.r = float(f.attrs['r'])
            self.N_theta = int(f.attrs['N_theta'])
            self.N_phi = int(f.attrs['N_phi'])

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

    def _compute_stats(self) -> None:
        self.p_mean = self.P.mean()
        self.p_std = torch.clamp(self.P.std(), min=1e-8)
        self.s_mean = self.S.mean()
        self.s_std = torch.clamp(self.S.std(), min=1e-8)

    def _normalize(self) -> None:
        self.P = (self.P - self.p_mean) / self.p_std
        self.S = (self.S - self.s_mean) / self.s_std

    def denormalize_p(self, p_tensor: torch.Tensor) -> torch.Tensor:
        return p_tensor * self.p_std + self.p_mean

    def denormalize_s(self, s_tensor: torch.Tensor) -> torch.Tensor:
        return s_tensor * self.s_std + self.s_mean

    def __len__(self) -> int:
        return self.num_rollouts * self.valid_starts_per_rollout

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rollout_idx = idx // self.valid_starts_per_rollout
        t_start = idx % self.valid_starts_per_rollout
        t_end = t_start + self.seq_len

        s_seq = self.S[rollout_idx, t_start:t_end]
        p_seq = self.P[rollout_idx, t_start:t_end]
        m_seq = self.metric_embed.expand(self.seq_len, -1, -1, -1)

        return s_seq.float(), p_seq.float(), m_seq.float()


class ChunkedTorusDataset(Dataset):
    """
    Lazy-loading dataset optimized for massive HDF5 sequences.
    Supports dynamic unrolling for Pushforward autoregressive training.
    """
    def __init__(self, h5_path: str, t_in: int = 5, t_out: int = 10, unroll_steps: int = 1):
        self.h5_path = h5_path
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 dataset not found: {h5_path}")
            
        self.t_in = t_in
        self.t_out = t_out
        self.unroll_steps = unroll_steps
        self.chunk_size = t_in + unroll_steps * t_out

        # We probe the HDF5 file once to gather metadata and shapes.
        # We do NOT load the entire dataset into RAM (prevents OOM).
        with h5py.File(self.h5_path, 'r') as f:
            self.num_rollouts, self.time_steps, _, _, _ = f['pressure'].shape
            self.R = float(f.attrs.get('R', 3.0))
            self.r = float(f.attrs.get('r', 1.0))
            self.N_theta = int(f.attrs.get('N_theta', 64))
            self.N_phi   = int(f.attrs.get('N_phi', 64))
            self.c = float(f.attrs.get('c', 1.0))
            self.dt_macro = float(f.attrs.get('dt_macro', 1.0))
            
            # To compute max scaling without loading everything into memory simultaneously,
            # we can sub-sample. For performance, we'll read a stride of the data.
            # If the file is small enough, it's fine, but we'll do a robust check.
            stride = max(1, self.num_rollouts // 5)
            p_sample = torch.from_numpy(f['pressure'][::stride, ::10, ..., 0])
            s_sample = torch.from_numpy(f['source'][::stride, ::10, ..., 0])
            
            self.p_scale = torch.clamp(torch.max(torch.abs(p_sample)), min=1e-4)
            self.s_scale = torch.clamp(torch.max(torch.abs(s_sample)), min=1e-4)

        self.valid_starts = max(0, self.time_steps - self.chunk_size + 1)
        if self.valid_starts == 0:
            raise ValueError(f"Simulation steps ({self.time_steps}) insufficient for unrolled chunk ({self.chunk_size})")

        # Static geometry features
        theta = torch.linspace(0, 2*np.pi, self.N_theta+1)[:-1]
        phi   = torch.linspace(0, 2*np.pi, self.N_phi+1)[:-1]
        THETA, PHI = torch.meshgrid(theta, phi, indexing='ij')
        metric = self.r * (self.R + self.r * torch.cos(THETA))
        m_norm = (metric - metric.min()) / (metric.max() - metric.min())
        
        self.geom_features = torch.stack([m_norm, THETA/(2*np.pi), PHI/(2*np.pi)], dim=0).float()
        
        # We will hold an open handle to the H5 file per worker
        self._h5_file: Optional[h5py.File] = None

    def _get_h5_file(self) -> h5py.File:
        """Lazily initialize the H5 file to ensure fork-safety with PyTorch DataLoader workers."""
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, 'r', swmr=True)
        return self._h5_file

    def __len__(self) -> int:
        return self.num_rollouts * self.valid_starts

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rollout = idx // self.valid_starts
        t_start = idx % self.valid_starts

        f = self._get_h5_file()
        
        # Read precisely the required slice from disk using h5py slice semantics.
        # The stored shape is (B, T, H, W, C). We select C=0 and permute to (T, W, H).
        # Actually stored from solver is: P_save = P.permute(0, 1, 3, 4, 2) -> (B, T, N_theta, N_phi, Channels)
        
        chunk_slice_p = f['pressure'][rollout, t_start : t_start + self.chunk_size, ..., 0]
        chunk_slice_s = f['source'][rollout, t_start : t_start + self.chunk_size, ..., 0]
        
        # Convert to PyTorch and permute to standard (Time, H, W) mapping
        P_raw = torch.from_numpy(chunk_slice_p).permute(0, 2, 1) # if original was 3,4,2
        S_raw = torch.from_numpy(chunk_slice_s).permute(0, 2, 1)

        # Scale
        P_raw = P_raw / self.p_scale
        S_raw = S_raw / self.s_scale

        # input state
        p_in = P_raw[0 : self.t_in]
        # full source trajectory (t_in + unroll_steps * t_out)
        s_unrolled = S_raw
        # full future target trajectory (unroll_steps * t_out)
        p_target = P_raw[self.t_in : self.t_in + self.unroll_steps * self.t_out]

        return p_in.float(), s_unrolled.float(), self.geom_features, p_target.float()
