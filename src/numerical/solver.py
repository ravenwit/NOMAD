import torch
import numpy as np
from typing import Callable, Optional, Tuple
from .geometry import TorusGeometry, compute_laplace_beltrami

class TorusWaveSolverRK4:
    def __init__(self, R: float = 1.0, r: float = 0.3, c: float = 1.0, 
                 N_theta: int = 256, N_phi: int = 256, CFL: float = 0.5):
        """
        High-fidelity 4th-Order Runge-Kutta acoustic wave solver on the Torus.
        """
        self.geom = TorusGeometry(R, r)
        self.c = c
        self.N_theta = N_theta
        self.N_phi = N_phi
        
        self.dtheta = 2 * np.pi / N_theta
        self.dphi = 2 * np.pi / N_phi
        
        # Strictest grid size occurs at the inner equator (theta = pi)
        # where g_{\phi\phi} = (R - r)^2, so physical distance dphi_phys = (R-r)*dphi
        min_dx = min(r * self.dtheta, (R - r) * self.dphi)
        
        # Time step governed by Courant-Friedrichs-Lewy condition
        self.dt = CFL * min_dx / c
        print(f"Initialized TorusSolver: Grid {N_theta}x{N_phi}. Required dt: {self.dt:.6f}")

    def generate_gaussian_pulse(self, t: float, t0: float, sigma_t: float, 
                                theta0: float, phi0: float, sigma_s: float, 
                                amplitude: torch.Tensor, device: torch.device):
        """
        A pure Gaussian source pulse mimicking an acoustic impact or speaker.
        """
        theta_1d = torch.linspace(0, 2*np.pi, self.N_theta + 1, device=device)[:-1]
        phi_1d = torch.linspace(0, 2*np.pi, self.N_phi + 1, device=device)[:-1]
        
        theta_grid, phi_grid = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
        
        # Geodesic-approximate spatial decay
        dtheta_dist = (theta_grid - theta0 + np.pi) % (2*np.pi) - np.pi
        dphi_dist = (phi_grid - phi0 + np.pi) % (2*np.pi) - np.pi
        
        # Physical distances using base radii
        phys_dist_sq = (self.geom.r * dtheta_dist)**2 + ((self.geom.R + self.geom.r * np.cos(theta0)) * dphi_dist)**2
        
        spatial = torch.exp(-phys_dist_sq / (2 * sigma_s**2))
        temporal = np.exp(-(t - t0)**2 / (2 * sigma_t**2))
        
        S_base = spatial * temporal
        S = S_base.unsqueeze(-1) * amplitude
        # Add batch dim and format to (B, C, H, W)
        S = S.unsqueeze(0).permute(0, 3, 1, 2)
        return S

    def _rk4_step(self, P: torch.Tensor, Q: torch.Tensor, S: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wave equation as first order system:
        dP/dt = Q
        dQ/dt = c^2 \Delta_M P + S
        """
        def dP_dt(q):
            return q
            
        def dQ_dt(p, s):
            LB = compute_laplace_beltrami(p, self.geom, self.dtheta, self.dphi)
            return self.c**2 * LB + s

        k1_P = dP_dt(Q)
        k1_Q = dQ_dt(P, S)

        P2 = P + 0.5 * self.dt * k1_P
        Q2 = Q + 0.5 * self.dt * k1_Q
        k2_P = dP_dt(Q2)
        k2_Q = dQ_dt(P2, S)

        P3 = P + 0.5 * self.dt * k2_P
        Q3 = Q + 0.5 * self.dt * k2_Q
        k3_P = dP_dt(Q3)
        k3_Q = dQ_dt(P3, S)

        P4 = P + self.dt * k3_P
        Q4 = Q + self.dt * k3_Q
        k4_P = dP_dt(Q4)
        k4_Q = dQ_dt(P4, S)

        P_new = P + (self.dt / 6.0) * (k1_P + 2*k2_P + 2*k3_P + k4_P)
        Q_new = Q + (self.dt / 6.0) * (k1_Q + 2*k2_Q + 2*k3_Q + k4_Q)

        return P_new, Q_new

    def simulate(self, num_steps: int, source_fn: Optional[Callable], 
                 device: torch.device, record_every: int = 10, channels: int = 1):
        
        P = torch.zeros((1, channels, self.N_theta, self.N_phi), device=device)
        Q = torch.zeros_like(P)
        
        history_P = []
        history_S = []
        
        t = 0.0
        for step in range(num_steps):
            if source_fn:
                S = source_fn(t, device)
            else:
                S = torch.zeros_like(P)
                
            P, Q = self._rk4_step(P, Q, S)
            t += self.dt
            
            if step % record_every == 0:
                history_P.append(P.clone().cpu())
                history_S.append(S.clone().cpu())
                
                if step % (num_steps//10) == 0:
                    print(f"Simulating progress: {100*step/num_steps:.1f}% (t={t:.4f}s)")
                    
                    # Watch for divergence blow-up
                    if not torch.isfinite(P).all() or P.abs().max() > 1e6:
                        print("WARNING: Numerical instability detected!")
                        break

        # Stack to (Batch, Time, Channels, H, W)
        return torch.stack(history_P, dim=1), torch.stack(history_S, dim=1)
