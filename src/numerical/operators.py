import torch
import numpy as np
from .geometry import TorusGeometry

def compute_gradient(f: torch.Tensor, dtheta: float, dphi: float, order: int = 4):
    r\"\"\"
    Computes \nabla f = [ \partial_\theta f, \partial_\phi f ] 
    using central differences with periodic (circular) boundaries.
    Default: 4th-order accurate.
    f shape: (Batch, Channels, N_theta, N_phi)
    \"\"\"
    if order == 4:
        # 4th order: (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / 12h
        f_pad_theta = torch.nn.functional.pad(f, (0, 0, 2, 2), mode='circular')
        df_dtheta = (-f_pad_theta[:, :, 4:, :] + 8*f_pad_theta[:, :, 3:-1, :] - 
                     8*f_pad_theta[:, :, 1:-3, :] + f_pad_theta[:, :, :-4, :]) / (12 * dtheta)
        
        f_pad_phi = torch.nn.functional.pad(f, (2, 2, 0, 0), mode='circular')
        df_dphi = (-f_pad_phi[:, :, :, 4:] + 8*f_pad_phi[:, :, :, 3:-1] - 
                   8*f_pad_phi[:, :, :, 1:-3] + f_pad_phi[:, :, :, :-4]) / (12 * dphi)
    else:
        # 2nd order: (f_{i+1} - f_{i-1}) / 2h
        f_pad_theta = torch.nn.functional.pad(f, (0, 0, 1, 1), mode='circular')
        df_dtheta = (f_pad_theta[:, :, 2:, :] - f_pad_theta[:, :, :-2, :]) / (2 * dtheta)
        
        f_pad_phi = torch.nn.functional.pad(f, (1, 1, 0, 0), mode='circular')
        df_dphi = (f_pad_phi[:, :, :, 2:] - f_pad_phi[:, :, :, :-2]) / (2 * dphi)
               
    return df_dtheta, df_dphi

def compute_laplace_beltrami(f: torch.Tensor, geometry: TorusGeometry, dtheta: float, dphi: float, order: int = 4):
    r\"\"\"
    Computes numerically the Laplace-Beltrami operator on a Torus:
    \Delta_M f = \frac{1}{\sqrt{|g|}} \partial_i (\sqrt{|g|} g^{ij} \partial_j f)
    f shape: (Batch, Channels, N_theta, N_phi)
    \"\"\"
    device = f.device
    N_theta = f.shape[2]
    
    # Generate theta grid on-the-fly for geometric scaling factors
    theta_1d = torch.linspace(0, 2*np.pi, N_theta + 1, device=device)[:-1]
    theta_grid = theta_1d.view(1, 1, N_theta, 1) # (1, 1, H, 1)
    
    sqrt_g = geometry.get_sqrt_det_g(theta_grid)
    g_inv_tt, g_inv_pp = geometry.get_inverse_metric_elements(theta_grid)
    
    # 1. Gradient of f: \partial_j f
    df_dtheta, df_dphi = compute_gradient(f, dtheta, dphi, order=order)
    
    # 2. Pre-divergence terms: V^i = \sqrt{|g|} g^{ij} \partial_j f
    V_theta = sqrt_g * g_inv_tt * df_dtheta
    V_phi   = sqrt_g * g_inv_pp * df_dphi
    
    # 3. Covariant Divergence: \partial_i V^i
    dV_theta_dtheta, _ = compute_gradient(V_theta, dtheta, dphi, order=order)
    _, dV_phi_dphi = compute_gradient(V_phi, dtheta, dphi, order=order)
    
    # 4. Scaling by 1/\sqrt{|g|}
    laplacian = (1.0 / sqrt_g) * (dV_theta_dtheta + dV_phi_dphi)
    
    return laplacian
