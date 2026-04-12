import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os

class TorusWaveDataset(Dataset):
    """
    Dataset to load 2D Torus wave simulation data.
    Input: [u(t), u(t-1)] -> Output: u(t+1)
    Supports noise injection to simulate sparse/noisy sensors.
    """
    def __init__(self, bin_path, nx=64, ny=64, noise_std=0.01):
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Simulation data not found at {bin_path}")
            
        # Load binary float32 data
        raw_data = np.fromfile(bin_path, dtype=np.float32)
        
        # Reshape to (frames, 1, H, W)
        self.nx, self.ny = nx, ny
        self.data = torch.from_numpy(raw_data).view(-1, 1, nx, ny)
        self.noise_std = noise_std
        
        # Create input/output pairs (History of 2 frames required for u_tt)
        self.inputs = []
        self.targets = []
        for i in range(1, len(self.data) - 1):
            # Input is cat([u_t, u_{t-1}]) -> Shape (2, H, W)
            self.inputs.append(torch.cat([self.data[i], self.data[i-1]], dim=0))
            self.targets.append(self.data[i+1])
            
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        x = self.inputs[idx].clone()
        y = self.targets[idx].clone()
        
        # Inject noise if specified
        if self.noise_std > 0:
            x += torch.randn_like(x) * self.noise_std
            
        return x, y

def get_dataloader(bin_path, batch_size=8, shuffle=True, noise_std=0.01):
    dataset = TorusWaveDataset(bin_path, noise_std=noise_std)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # Test loader
    path = "web/public/simulation_data.bin"
    if os.path.exists(path):
        loader = get_dataloader(path)
        x, y = next(iter(loader))
        print(f"Batch shape: {x.shape} -> {y.shape}") # Expect (B, 2, 64, 64) -> (B, 1, 64, 64)
        print(f"Value range: {x.min().item():.4f} to {x.max().item():.4f}")
    else:
        print("Test skipped: bin file not found.")
