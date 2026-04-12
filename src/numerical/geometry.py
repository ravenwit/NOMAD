import torch
import numpy as np

class TorusGeometry:
    def __init__(self, R: float = 3.0, r: float = 1.0):
        r"""
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
        r"""
        Returns the non-zero components of the metric tensor: g_{\theta\theta} and g_{\phi\phi}
        """
        # g_tt = r^2
        g_tt = torch.full_like(theta, self.r**2)
        # g_pp = (R + r * cos(theta))^2
        g_pp = (self.R + self.r * torch.cos(theta))**2
        return g_tt, g_pp

    def get_inverse_metric_elements(self, theta: torch.Tensor):
        r"""
        Returns g^{\theta\theta} and g^{\phi\phi}
        """
        g_tt, g_pp = self.get_metric_elements(theta)
        return 1.0 / g_tt, 1.0 / g_pp

    def get_sqrt_det_g(self, theta: torch.Tensor):
        r"""
        Returns \sqrt{|g|} = r(R + r \cos\theta)
        """
        return self.r * (self.R + self.r * torch.cos(theta))
