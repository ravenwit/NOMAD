import torch
import numpy as np

class TorusGeometry:
    def __init__(self, R: float = 1.0, r: float = 0.3):
        """
        Differential geometry of a 2D Torus.
        Coordinates: (theta, phi)
        theta \in [0, 2pi) - Poloidal angle (around the tube)
        phi \in [0, 2pi)   - Toroidal angle (around the main axis)
        
        Metric g:
        ds^2 = r^2 d\theta^2 + (R + r \cos\theta)^2 d\phi^2
        """
        self.R = R
        self.r = r

    def get_metric_elements(self, theta: torch.Tensor):
        """
        Returns the non-zero components of the metric tensor: g_{\theta\theta} and g_{\phi\phi}
        """
        g_tt = torch.full_like(theta, self.r**2)
        g_pp = (self.R + self.r * torch.cos(theta))**2
        return g_tt, g_pp

    def get_inverse_metric_elements(self, theta: torch.Tensor):
        """
        Returns g^{\theta\theta} and g^{\phi\phi}
        """
        g_tt, g_pp = self.get_metric_elements(theta)
        return 1.0 / g_tt, 1.0 / g_pp

    def get_sqrt_det_g(self, theta: torch.Tensor):
        """
        Returns \sqrt{|g|} = r(R + r \cos\theta)
        """
        return self.r * (self.R + self.r * torch.cos(theta))


def compute_gradient(f, dtheta: float, dphi: float):
    """
    Computes \nabla f = [ \partial_\theta f, \partial_\phi f ] 
    using 4th-order central differences with periodic boundaries.
    f shape: (Batch, Channels, N_theta, N_phi)
    """
    # 4th order: (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2}) / 12h
    # Pad circularly
    f_pad_theta = torch.nn.functional.pad(f, (0, 0, 2, 2), mode='circular')
    df_dtheta = (-f_pad_theta[:, :, 4:, :] + 8*f_pad_theta[:, :, 3:-1, :] - 
                 8*f_pad_theta[:, :, 1:-3, :] + f_pad_theta[:, :, :-4, :]) / (12 * dtheta)
    
    f_pad_phi = torch.nn.functional.pad(f, (2, 2, 0, 0), mode='circular')
    df_dphi = (-f_pad_phi[:, :, :, 4:] + 8*f_pad_phi[:, :, :, 3:-1] - 
               8*f_pad_phi[:, :, :, 1:-3] + f_pad_phi[:, :, :, :-4]) / (12 * dphi)
               
    return df_dtheta, df_dphi


def compute_laplace_beltrami(f: torch.Tensor, geometry: TorusGeometry, dtheta: float, dphi: float):
    """
    Computes \Delta_M f = \frac{1}{\sqrt{|g|}} \partial_i (\sqrt{|g|} g^{ij} \partial_j f)
    f shape: (Batch, Channels, N_theta, N_phi)
    """
    device = f.device
    N_theta = f.shape[2]
    
    # 1D theta grid matching spatial resolution
    theta_1d = torch.linspace(0, 2*np.pi, N_theta + 1, device=device)[:-1]
    theta_grid = theta_1d.view(1, 1, N_theta, 1) # Expand shapes for broadcasting
    
    # Differential geometry terms
    sqrt_g = geometry.get_sqrt_det_g(theta_grid)
    g_inv_tt, g_inv_pp = geometry.get_inverse_metric_elements(theta_grid)
    
    # 1. Gradient: \partial_j f
    df_dtheta, df_dphi = compute_gradient(f, dtheta, dphi)
    
    # 2. Multiply by \sqrt{|g|} g^{ij}
    V_theta = sqrt_g * g_inv_tt * df_dtheta
    V_phi   = sqrt_g * g_inv_pp * df_dphi
    
    # 3. Divergence: \partial_i V^i
    dV_theta_dtheta, _ = compute_gradient(V_theta, dtheta, dphi)
    _, dV_phi_dphi = compute_gradient(V_phi, dtheta, dphi)
    
    # 4. Multiply by 1/\sqrt{|g|}
    laplacian = (1.0 / sqrt_g) * (dV_theta_dtheta + dV_phi_dphi)
    
    return laplacian
