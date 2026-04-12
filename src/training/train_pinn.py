import torch
import torch.nn as nn
import torch.optim as optim
from src.data.dataset import get_dataloader
from src.models.pinn import PINN
import numpy as np
import os

# Physical parameters (must match geometry.py)
R = 1.5
r = 0.5
c = 1.0 # Wave speed
dt = 0.05
dx = (2 * np.pi) / 64
dy = (2 * np.pi) / 64

def laplace_beltrami_pytorch(u, R, r, dx, dy):
    """
    Differentiable Laplace-Beltrami operator for a Torus in PyTorch.
    u: tensor of shape (B, 1, H, W) where H is theta, W is phi
    """
    B, C, N_theta, N_phi = u.shape
    
    # Precompute geometric terms
    theta = torch.linspace(0, 2*np.pi, N_theta, device=u.device).view(1, 1, -1, 1)
    sqrt_g = r * (R + r * torch.cos(theta))
    inv_r = 1.0 / r
    
    # Use circular padding for derivatives to maintain Toroidal topology
    def deriv_theta(f):
        f_pad = torch.cat([f[:, :, -1:], f, f[:, :, :1]], dim=2)
        return (f_pad[:, :, 2:] - f_pad[:, :, :-2]) / (2 * dx)
        
    def deriv_phi(f):
        f_pad = torch.cat([f[:, :, :, -1:], f, f[:, :, :, :1]], dim=3)
        return (f_pad[:, :, :, 2:] - f_pad[:, :, :, :-2]) / (2 * dy)

    # term1: d_theta ( (R + r*cos_theta) * (1/r) * d_theta u )
    du_dtheta = deriv_theta(u)
    term1_inner = (R + r * torch.cos(theta)) * inv_r * du_dtheta
    term1 = deriv_theta(term1_inner)
    
    # term2: d_phi ( (r / (R + r*cos_theta)) * d_phi u )
    du_dphi = deriv_phi(u)
    term2_inner = (r / (R + r * torch.cos(theta))) * du_dphi
    term2 = deriv_phi(term2_inner)
    
    return (1.0 / sqrt_g) * (term1 + term2)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # 1. Setup Data
    bin_path = "web/public/simulation_data.bin"
    dataloader = get_dataloader(bin_path, batch_size=8, shuffle=True, noise_std=0.005)
    
    # 2. Setup Model
    model = PINN(hidden_dim=64).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    mse_loss = nn.MSELoss()
    
    # 3. Training Loop
    epochs = 100
    pde_lambda = 0.1 # Weight of physics constraint
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_pde = 0
        
        for x, y_target in dataloader:
            x, y_target = x.to(device), y_target.to(device)
            # x is [u_t, u_{t-1}]
            u_t = x[:, 0:1, :, :]
            u_tm1 = x[:, 1:2, :, :]
            
            # Predict u_{t+1}
            u_tp1_pred = model(x)
            
            # Data Loss (MSE against ground truth from solver)
            loss_data = mse_loss(u_tp1_pred, y_target)
            
            # PDE Residual Loss
            # u_tt = c^2 * Delta_g u
            u_tt = (u_tp1_pred - 2*u_t + u_tm1) / (dt**2)
            laplacian = laplace_beltrami_pytorch(u_t, R, r, dx, dy)
            loss_pde = torch.mean((u_tt - (c**2) * laplacian)**2)
            
            total_loss = loss_data + pde_lambda * loss_pde
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += loss_data.item()
            epoch_pde += loss_pde.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Data Loss: {epoch_loss/len(dataloader):.6f} | PDE Loss: {epoch_pde/len(dataloader):.6f}")

    # Save Model
    torch.save(model.state_dict(), "src/models/torus_pinn.pth")
    print("Model saved to src/models/torus_pinn.pth")

if __name__ == "__main__":
    train()
