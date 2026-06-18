import pytest
import torch
import numpy as np
from src.numerical.geometry import TorusGeometry, compute_laplace_beltrami
from src.numerical.solver import TorusAcousticSimulator, TorusSpectralSolver

def test_torus_geometry():
    R, r = 3.0, 1.0
    geom = TorusGeometry(R=R, r=r)
    assert geom.R == R
    assert geom.r == r
    
    # Test valid domains
    theta, phi = torch.tensor([0.0, np.pi]), torch.tensor([0.0, np.pi])
    x, y, z = geom.compute_cartesian(theta, phi)
    
    assert x.shape == theta.shape
    assert y.shape == theta.shape
    assert z.shape == theta.shape
    
    # Check max radius
    max_radius = torch.sqrt(x[0]**2 + y[0]**2)
    assert torch.isclose(max_radius, torch.tensor(R + r))

def test_laplace_beltrami_finite():
    # Ensure Laplacian computation does not produce NaNs for zero or uniform fields
    geom = TorusGeometry(3.0, 1.0)
    p = torch.ones((1, 1, 64, 64))
    dtheta = 2 * np.pi / 64
    dphi = 2 * np.pi / 64
    
    laplacian = compute_laplace_beltrami(p, geom, dtheta, dphi)
    assert torch.isfinite(laplacian).all()
    # Laplacian of a constant field should be near zero
    assert torch.max(torch.abs(laplacian)) < 1e-5

def test_simulator_cfl():
    simulator = TorusAcousticSimulator(R=3.0, r=1.0, N_theta=64, N_phi=64, c=343.0)
    assert simulator.dt > 0.0
    
    # Ensure CFL-derived dt is small enough for stability
    min_dx = min(1.0 * (2*np.pi/64), 2.0 * (2*np.pi/64))
    expected_dt = 0.1 * min_dx / 343.0
    assert np.isclose(simulator.dt, expected_dt)

def test_spectral_solver_shapes():
    solver = TorusSpectralSolver(N_theta=32, N_phi=32)
    p_stack, s_stack = solver.simulate(
        num_steps=10,
        source_fn=None,
        device=torch.device('cpu'),
        record_every=2,
        channels=1
    )
    # 10 steps, record every 2 -> 5 frames
    assert p_stack.shape == (1, 5, 1, 32, 32)
    assert s_stack.shape == (1, 5, 1, 32, 32)
