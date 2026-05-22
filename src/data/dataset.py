import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
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

# if __name__ == "__main__":
#     # Test path
#     path = "simulation_results.h5"
#     if os.path.exists(path):
#         dataset = TorusWaveDataset(path)
#         s, p, m = dataset[0]
#         print(f"Loaded S shape: {s.shape}, P shape: {p.shape}, M shape: {m.shape}")
#         print(f"P stats - Mean: {dataset.p_mean:.4f}, Std: {dataset.p_std:.4f}")
#     else:
#         print("Test skipped: .h5 file not found.")


# Test path
# path = "torus_simulation_data.h5"
# if os.path.exists(path):
#     dataset = TorusWaveDataset(path, seq_len=511)
#     s, p, m = dataset[0]
#     print(f"Loaded S shape: {s.shape}, P shape: {p.shape}, M shape: {m.shape}")
#     print(f"P stats - Mean: {dataset.p_mean:.4f}, Std: {dataset.p_std:.4f}")
# else:
#     print("Test skipped: .h5 file not found.")
