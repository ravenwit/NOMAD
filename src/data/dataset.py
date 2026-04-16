import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import os

class TorusWaveDataset(Dataset):
    """
    NOMAD Dataset for Acoustic Waves on a Torus.
    Loads simulation data from HDF5 files and prepares them for training.
    
    Tensor Shaping: (Batch, Time, Channels, Theta, Phi) -> outputs overlapping sequences.
    Normalization: Z-score standardization (retains stats for denormalization).
    """
    def __init__(self, h5_path, seq_len=4, transform=True):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Simulation data not found at {h5_path}")
            
        with h5py.File(h5_path, 'r') as f:
            self.P = torch.from_numpy(f['pressure'][:]).permute(0, 1, 4, 2, 3)
            self.S = torch.from_numpy(f['source'][:]).permute(0, 1, 4, 2, 3)
            self.R = f.attrs['R']
            self.r = f.attrs['r']
            self.N_theta = f.attrs['N_theta']
            self.N_phi = f.attrs['N_phi']
            
        self.seq_len = seq_len
        self.num_rollouts = self.P.shape[0]
        self.time_steps = self.P.shape[1]
        self.valid_starts_per_rollout = self.time_steps - self.seq_len
        
        self.transform = transform
        if self.transform:
            self._compute_stats()
            self._normalize()

        # Precompute the geometric metric channel
        # Metric g = r(R + r * cos(theta))
        theta_grid = torch.linspace(0, 2*np.pi, self.N_theta + 1)[:-1]
        phi_grid = torch.linspace(0, 2*np.pi, self.N_phi + 1)[:-1]
        THETA, _ = torch.meshgrid(theta_grid, phi_grid, indexing='ij')
        
        metric = self.r * (self.R + self.r * torch.cos(THETA))
        # Normalize metric approximately to [0, 1] range to help neural networks
        m_min = self.r * (self.R - self.r)
        m_max = self.r * (self.R + self.r)
        metric_norm = (metric - m_min) / (m_max - m_min)
        
        # metric_norm: (1, 1, H, W). We'll expand it during getitem
        self.metric_embed = metric_norm.unsqueeze(0).unsqueeze(0).to(torch.float32)

    def _compute_stats(self):
        """Calculates mean and std for Z-score normalization."""
        self.p_mean = self.P.mean()
        self.p_std = self.P.std()
        self.s_mean = self.S.mean()
        self.s_std = self.S.std()
        
        self.p_std = torch.clamp(self.p_std, min=1e-8)
        self.s_std = torch.clamp(self.s_std, min=1e-8)

    def _normalize(self):
        """Applies Z-score standardization."""
        self.P = (self.P - self.p_mean) / self.p_std
        self.S = (self.S - self.s_mean) / self.s_std

    def denormalize_p(self, p_tensor):
        return p_tensor * self.p_std + self.p_mean

    def denormalize_s(self, s_tensor):
        return s_tensor * self.s_std + self.s_mean

    def __len__(self):
        return self.num_rollouts * self.valid_starts_per_rollout
        
    def __getitem__(self, idx):
        # Decode the flattened index into (rollout_idx, time_start)
        rollout_idx = idx // self.valid_starts_per_rollout
        t_start = idx % self.valid_starts_per_rollout
        t_end = t_start + self.seq_len
        
        s_seq = self.S[rollout_idx, t_start:t_end]
        p_seq = self.P[rollout_idx, t_start:t_end]
        
        # We append the static geometry metric map along the sequence so the
        # network inherently knows where the curvature compresses/expands waves.
        # Shape: (Seq_len, 1, H, W)
        m_seq = self.metric_embed.expand(self.seq_len, -1, -1, -1)
        
        return s_seq.float(), p_seq.float(), m_seq.float()

def get_dataloader(h5_path, batch_size=16, seq_len=4, shuffle=True):
    dataset = TorusWaveDataset(h5_path, seq_len=seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # Test path
    path = "simulation_results.h5"
    if os.path.exists(path):
        dataset = TorusWaveDataset(path)
        s, p, m = dataset[0]
        print(f"Loaded S shape: {s.shape}, P shape: {p.shape}, M shape: {m.shape}")
        print(f"P stats - Mean: {dataset.p_mean:.4f}, Std: {dataset.p_std:.4f}")
    else:
        print("Test skipped: .h5 file not found.")
