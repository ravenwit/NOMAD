import torch
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import os

class TorusWaveDataset(Dataset):
    """
    NOMAD Dataset for Acoustic Waves on a Torus.
    Loads simulation data from HDF5 files and prepares them for training.
    
    Tensor Shaping: (Batch, Time, Channels, Theta, Phi)
    Normalization: Z-score standardization (retains stats for denormalization).
    """
    def __init__(self, h5_path, transform=True):
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Simulation data not found at {h5_path}")
            
        with h5py.File(h5_path, 'r') as f:
            # Data expected shape from solver: (B, T, H, W, C) -> we transpose to (B, T, C, H, W)
            # Or (B, T, C, H, W) directly if saved that way.
            # Based on solver.py logic: P_save = P.permute(0, 1, 3, 4, 2).numpy() which is (B, T, H, W, C)
            self.P = torch.from_numpy(f['pressure'][:]).permute(0, 1, 4, 2, 3)
            self.S = torch.from_numpy(f['source'][:]).permute(0, 1, 4, 2, 3)
            
        self.transform = transform
        if self.transform:
            self._compute_stats()
            self._normalize()

    def _compute_stats(self):
        """Calculates mean and std for Z-score normalization."""
        self.p_mean = self.P.mean()
        self.p_std = self.P.std()
        self.s_mean = self.S.mean()
        self.s_std = self.S.std()
        
        # Ensure std is not zero
        self.p_std = torch.clamp(self.p_std, min=1e-8)
        self.s_std = torch.clamp(self.s_std, min=1e-8)

    def _normalize(self):
        """Applies Z-score standardization."""
        self.P = (self.P - self.p_mean) / self.p_std
        self.S = (self.S - self.s_mean) / self.s_std

    def denormalize_p(self, p_tensor):
        """Retains statistical constants to denormalize for physics-informed loss."""
        return p_tensor * self.p_std + self.p_mean

    def denormalize_s(self, s_tensor):
        return s_tensor * self.s_std + self.s_mean

    def __len__(self):
        return self.P.shape[0] # Number of batches saved in H5
        
    def __getitem__(self, idx):
        # We return the whole sequence for a given batch entry
        # Shapes: (Time, Channels, H, W)
        return self.S[idx], self.P[idx]

def get_dataloader(h5_path, batch_size=1, shuffle=True):
    dataset = TorusWaveDataset(h5_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # Test path
    path = "simulation_results.h5"
    if os.path.exists(path):
        dataset = TorusWaveDataset(path)
        s, p = dataset[0]
        print(f"Loaded S shape: {s.shape}, P shape: {p.shape}")
        print(f"P stats - Mean: {dataset.p_mean:.4f}, Std: {dataset.p_std:.4f}")
    else:
        print("Test skipped: .h5 file not found.")
