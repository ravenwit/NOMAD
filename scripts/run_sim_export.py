import sys
import os
import torch
import json
import numpy as np

# Add src to python path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.numerical.solver import TorusWaveSolverRK4

def run_and_export():
    print("Starting high-fidelity numerical simulation on Torus...")
    
    # 1. Setup Solver
    # Resolution chosen for balance of accuracy and browser memory (approx 3MB for 100 frames)
    N_theta, N_phi = 64, 64 
    solver = TorusWaveSolverRK4(R=1.5, r=0.5, c=1.0, N_theta=N_theta, N_phi=N_phi, CFL=0.25)
    
    device = torch.device("cpu")
    
    # 2. Setup Source Function: Gaussian impact at center
    def source_func(t, d):
        # Pulse between t=0.1 and t=0.3
        return solver.generate_gaussian_pulse(
            t=t, t0=0.1, sigma_t=0.05, 
            theta0=np.pi, phi0=np.pi, sigma_s=0.2, 
            amplitude=torch.ones(1, device=d) * 10.0, device=d
        )
    
    # 3. Predict 200 time steps, recording every step
    # P_seq shape: (Batch=1, Time, Channels=1, H=64, W=64)
    print("Simulating...")
    P_seq, S_seq = solver.simulate(num_steps=200, source_fn=source_func, device=device, record_every=2)
    P_seq = P_seq.squeeze() # (Time, H, W)
    
    time_steps = P_seq.shape[0]
    
    # 4. Export as raw sequence of Float32 binary for ultra-fast WebGL parsing
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../web/public'))
    os.makedirs(out_dir, exist_ok=True)
    
    bin_path = os.path.join(out_dir, 'simulation_data.bin')
    # Flatten and save as direct raw bytes
    np_arr = P_seq.numpy().astype(np.float32)
    with open(bin_path, 'wb') as f:
        f.write(np_arr.tobytes())
        
    # Export config
    config = {
        "N_theta": N_theta,
        "N_phi": N_phi,
        "frames": time_steps,
        "dt": solver.dt * 2 # Account for record_every=2
    }
    with open(os.path.join(out_dir, 'sim_config.json'), 'w') as f:
        json.dump(config, f)
        
    print(f"Exported {time_steps} frames of {N_theta}x{N_phi} to {bin_path}")

if __name__ == "__main__":
    run_and_export()
