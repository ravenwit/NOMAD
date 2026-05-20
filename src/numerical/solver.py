import torch
import numpy as np
import h5py
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

    def generate_ricker_pulse(self, t: float, t0: float, sigma_t: float, 
                             theta0: float, phi0: float, sigma_s: float, 
                             amplitude: torch.Tensor, device: torch.device):
        """
        A zero-mean Ricker Wavelet (Mexican Hat) source pulse.
        """
        theta_1d = torch.linspace(0, 2*np.pi, self.N_theta + 1, device=device)[:-1]
        phi_1d = torch.linspace(0, 2*np.pi, self.N_phi + 1, device=device)[:-1]
        
        theta_grid, phi_grid = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
        
        dtheta_dist = (theta_grid - theta0 + np.pi) % (2*np.pi) - np.pi
        dphi_dist = (phi_grid - phi0 + np.pi) % (2*np.pi) - np.pi
        
        # Physical squared distance
        r_sq = (self.geom.r * dtheta_dist)**2 + ((self.geom.R + self.geom.r * np.cos(theta0)) * dphi_dist)**2
        
        # Ricker Wavelet: (1 - r^2/sigma^2) * exp(-r^2/(2*sigma^2))
        r_sq_over_sigma_sq = r_sq / (sigma_s ** 2)
        spatial = (2.0 - r_sq_over_sigma_sq) * torch.exp(-r_sq / (2 * sigma_s ** 2))
        
        # Subtract mean to ensure zero-mean (Mexican Hat filter)
        spatial = spatial - spatial.mean()
        
        temporal = np.exp(-(t - t0)**2 / (2 * sigma_t**2))
        
        S = spatial * temporal
        S = S.unsqueeze(-1) * amplitude
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

class TorusSpectralSolver:
    def __init__(self, R: float = 3.0, r: float = 1.0, c: float = 343.0, 
                 N_theta: int = 256, N_phi: int = 256, CFL: float = 0.1):
        """
        Implementation of the Fourier Pseudospectral method for acoustic waves on a torus.
        As described in acoustic-spectral.md
        """
        self.R = R
        self.r = r
        self.c = c
        self.N_theta = N_theta
        self.N_phi = N_phi
        
        self.d_theta = 2 * np.pi / N_theta
        self.d_phi = 2 * np.pi / N_phi
        
        # CFL Condition: dt = CFL * min_dx / c
        min_dx = min(r * self.d_theta, (R - r) * self.d_phi)
        self.dt = CFL * min_dx / c
        
        # Wavenumbers for spectral differentiation
        # k = [0, 1, ..., N/2-1, -N/2, ..., -1]
        self.k_theta = torch.fft.fftfreq(N_theta).to(torch.float32) * N_theta
        self.k_phi = torch.fft.fftfreq(N_phi).to(torch.float32) * N_phi
        
        self.K_THETA, self.K_PHI = torch.meshgrid(self.k_theta, self.k_phi, indexing='ij')

        # Static Geometric Terms (Physical Space)
        theta_grid = torch.linspace(0, 2*np.pi, N_theta + 1)[:-1]
        phi_grid = torch.linspace(0, 2*np.pi, N_phi + 1)[:-1]
        THETA, _ = torch.meshgrid(theta_grid, phi_grid, indexing='ij')
        
        self.THETA = THETA
        self.g_inv_tt = 1.0 / (r**2)
        self.g_inv_pp = 1.0 / (R + r * torch.cos(THETA))**2
        self.gamma_term = -torch.sin(THETA) / (r * (R + r * torch.cos(THETA)))
        self.sqrt_g = r * (R + r * torch.cos(THETA))

    def generate_ricker_pulse(self, t: float, t0: float, sigma_t: float, 
                             theta0: float, phi0: float, sigma_s: float, 
                             amplitude: torch.Tensor, device: torch.device):
        """
        Generate a zero-mean Ricker Wavelet (Mexican Hat) source pulse in 2D.
        """
        theta_1d = torch.linspace(0, 2*np.pi, self.N_theta + 1, device=device)[:-1]
        phi_1d = torch.linspace(0, 2*np.pi, self.N_phi + 1, device=device)[:-1]
        
        theta_grid, phi_grid = torch.meshgrid(theta_1d, phi_1d, indexing='ij')
        
        dtheta_dist = (theta_grid - theta0 + np.pi) % (2*np.pi) - np.pi
        dphi_dist = (phi_grid - phi0 + np.pi) % (2*np.pi) - np.pi
        
        # Physical squared distance
        r_sq = (self.r * dtheta_dist)**2 + ((self.R + self.r * np.cos(theta0)) * dphi_dist)**2
        
        r_sq_over_sigma_sq = r_sq / (sigma_s ** 2)
        spatial = (2.0 - r_sq_over_sigma_sq) * torch.exp(-r_sq / (2 * sigma_s ** 2))
        spatial = spatial - spatial.mean()
        
        temporal = np.exp(-(t - t0)**2 / (2 * sigma_t**2))
        
        # S base shape: (N_theta, N_phi)
        S_base = spatial * temporal
        
        # Expand based on amplitude (C, H, W)
        # amplitude shape: (C,)
        S = S_base.unsqueeze(0) * amplitude.view(-1, 1, 1)
        return S # (C, H, W)

    def compute_laplace_beltrami(self, P: torch.Tensor) -> torch.Tensor:
        """
        Computes the Laplacian in spectral space.
        """
        device = P.device
        # Move precomputed tensors to the correct device if needed
        self.K_THETA = self.K_THETA.to(device)
        self.K_PHI = self.K_PHI.to(device)
        self.g_inv_pp = self.g_inv_pp.to(device)
        self.gamma_term = self.gamma_term.to(device)
        self.sqrt_g = self.sqrt_g.to(device)

        # Forward FFT
        P_hat = torch.fft.fft2(P)
        
        # First derivative w.r.t theta: F^-1 [ i * k_theta * P_hat ]
        dP_dtheta_hat = 1j * self.K_THETA * P_hat
        dP_dtheta = torch.real(torch.fft.ifft2(dP_dtheta_hat))
        
        # Second derivative w.r.t theta: F^-1 [ -k_theta^2 * P_hat ]
        d2P_dtheta2_hat = -(self.K_THETA**2) * P_hat
        d2P_dtheta2 = torch.real(torch.fft.ifft2(d2P_dtheta2_hat))
        
        # Second derivative w.r.t phi: F^-1 [ -k_phi^2 * P_hat ]
        d2P_dphi2_hat = -(self.K_PHI**2) * P_hat
        d2P_dphi2 = torch.real(torch.fft.ifft2(d2P_dphi2_hat))
        
        # Assemble Laplacian in physical space
        # LB = (1/r^2) * d2P/dtheta^2 + gamma * dP/dtheta + g^pp * d2P/dphi^2
        # Note: self.g_inv_tt is 1/r^2
        laplace = (self.g_inv_tt * d2P_dtheta2) + \
                  (self.gamma_term * dP_dtheta) + \
                  (self.g_inv_pp * d2P_dphi2)
                  
        return laplace

    def simulate(self, num_steps: int, source_fn: Optional[Callable], 
                 device: torch.device, record_every: int = 10, channels: int = 3):
        """
        Explicit Leapfrog Time-Stepping as per acoustic-spectral.md
        """
        P_curr = torch.zeros((channels, self.N_theta, self.N_phi), device=device)
        P_prev = torch.zeros_like(P_curr)
        
        history_P = []
        history_S = []
        t = 0.0
        
        for step in range(num_steps):
            S_curr = source_fn(t, device) if source_fn else torch.zeros_like(P_curr)
            
            # Evaluate Laplacian
            laplacian = self.compute_laplace_beltrami(P_curr)
            
            # Acceleration: c^2 * (Laplacian + S)
            accel = (self.c**2) * (laplacian + S_curr)
            
            # Leapfrog: P_next = 2*P_curr - P_prev + dt^2 * acceleration
            P_next = 2 * P_curr - P_prev + (self.dt**2) * accel
            
            # Update state
            P_prev = P_curr
            P_curr = P_next
            t += self.dt
            
            if step % record_every == 0:
                history_P.append(P_curr.clone().cpu())
                history_S.append(S_curr.clone().cpu())
                
                if step % (max(1, num_steps//10)) == 0:
                    print(f"Spectral Sim: {100*step/num_steps:.1f}% (t={t:.4f}s)")
                    if not torch.isfinite(P_curr).all():
                        print("ERROR: Divergence in spectral solver!")
                        break

        # Return (Batch=1, Time, Channels, H, W) to match the expected deep learning dataset shape
        P_stack = torch.stack(history_P, dim=0).unsqueeze(0)
        S_stack = torch.stack(history_S, dim=0).unsqueeze(0)
        return P_stack, S_stack

class TorusAcousticSimulator:
    def __init__(self, R=3.0, r=1.0, c=343.0, N_theta=128, N_phi=128, dt=None):
        self.solver = TorusSpectralSolver(R, r, c, N_theta, N_phi, CFL=0.1)
        # Use solver's dt if we want strictly CFL-safe, or override
        self.dt = dt if dt else self.solver.dt
        self.solver.dt = self.dt
        
    def generate_gaussian_source(self, t, t0=0.05, sigma_t=0.01, theta0=np.pi, phi0=np.pi, sigma_s=0.5, amplitude=None, device='cpu'):
        """Deprecated: use generate_ricker_pulse instead."""
        return self.generate_ricker_pulse(t, t0, sigma_t, theta0, phi0, sigma_s, amplitude, device)

    def generate_ricker_pulse(self, t, t0=0.05, sigma_t=0.01, theta0=np.pi, phi0=np.pi, sigma_s=0.5, amplitude=None, device='cpu'):
        if amplitude is None: amplitude = torch.tensor([1.0], device=device)
        return self.solver.generate_ricker_pulse(t, t0, sigma_t, theta0, phi0, sigma_s, amplitude, device)

    def generate_kicker_pulse(self, t, t0=0.05, sigma_t=0.01, theta0=np.pi, phi0=np.pi, sigma_s=0.5, amplitude=None, device='cpu'):
        """Alias for generate_ricker_pulse."""
        return self.generate_ricker_pulse(t, t0, sigma_t, theta0, phi0, sigma_s, amplitude, device)

    def simulate(self, num_steps=500, source_generator_fn=None, device='cpu', record_every=10):
        return self.solver.simulate(num_steps, source_generator_fn, device, record_every)

    def save_to_h5(self, P, S, filename):
        save_simulation_to_h5(P, S, filename, self.solver.R, self.solver.r, self.solver.dt, self.solver.N_theta, self.solver.N_phi)

def save_simulation_to_h5(P, S, filename, R, r, dt, N_theta, N_phi):
    """
    Saves NOMAD simulation data to HDF5.
    Shapes expected: P, S: (Batch, Time, Channels, H, W)
    """
    with h5py.File(filename, 'w') as f:
        # Move Channels to end for storage: (B, T, C, H, W) -> (B, T, H, W, C)
        P_save = P.permute(0, 1, 3, 4, 2).numpy()
        S_save = S.permute(0, 1, 3, 4, 2).numpy()
        f.create_dataset('pressure', data=P_save, compression="gzip")
        f.create_dataset('source', data=S_save, compression="gzip")
        f.attrs['R'] = R
        f.attrs['r'] = r
        f.attrs['dt'] = dt
        f.attrs['N_theta'] = N_theta
        f.attrs['N_phi'] = N_phi
    print(f"Dataset successfully saved to {filename}")
