import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset import TorusWaveDataset
from torch.utils.data import DataLoader
from src.numerical.operators import compute_laplace_beltrami
from src.numerical.geometry import TorusGeometry
from src.models.pinn import PINN
import numpy as np
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Physical parameters (must match geometry.py)
    R = 3.0
    r = 1.0
    c = 343.0 # Speed of sound
    dt = 0.001
    geometry = TorusGeometry(R, r)
    
    # 1. Setup Data
    h5_path = "simulation_results.h5"
    if not os.path.exists(h5_path):
        print(f"ERROR: {h5_path} not found. Run simulation first.")
        return

    dataset = TorusWaveDataset(h5_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Grid steps for LB
    N_theta, N_phi = dataset.P.shape[-2:]
    dx = (2 * np.pi) / N_theta
    dy = (2 * np.pi) / N_phi
    
    # 2. Setup Model (assuming input is [u_t, u_t-1] - 2 channels)
    model = PINN(hidden_dim=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    mse_loss = nn.MSELoss()
    
    # 3. Training Loop
    epochs = 50
    pde_lambda = 0.5 # Weight of physics constraint
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for s_seq, p_seq in dataloader:
            s_seq, p_seq = s_seq.to(device), p_seq.to(device)
            # Sample a random triplet in time
            t_idx = np.random.randint(1, p_seq.shape[1] - 1)
            
            p_prev = p_seq[:, t_idx-1]
            p_curr = p_seq[:, t_idx]
            p_next_target = p_seq[:, t_idx+1]
            s_curr = s_seq[:, t_idx]
            
            # Pack input: [u_t, u_{t-1}]
            x_in = torch.cat([p_curr[:, :1], p_prev[:, :1]], dim=1)
            
            # Predict u_{t+1}
            p_next_pred = model(x_in)
            
            # Data Loss
            loss_data = mse_loss(p_next_pred, p_next_target[:, :1])
            
            # PDE Residual Loss (Denormalize for physical consistency)
            p_next_phys = dataset.denormalize_p(p_next_pred)
            p_curr_phys = dataset.denormalize_p(p_curr[:, :1])
            p_prev_phys = dataset.denormalize_p(p_prev[:, :1])
            s_phys = dataset.denormalize_s(s_curr[:, :1])
            
            u_tt = (p_next_phys - 2*p_curr_phys + p_prev_phys) / (dt**2)
            # Use the unified high-accuracy operator
            laplacian = compute_laplace_beltrami(p_curr_phys, geometry, dx, dy)
            loss_pde = torch.mean((u_tt - (c**2) * (laplacian + s_phys))**2)
            
            total_loss = loss_data + pde_lambda * loss_pde
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(dataloader):.6f}")

    torch.save(model.state_dict(), "src/models/torus_pinn.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()
