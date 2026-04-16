import torch
import numpy as np
import h5py
import os
import argparse
from tqdm import tqdm
from src.numerical.solver import TorusAcousticSimulator, save_simulation_to_h5

def generate_multi_rollout_dataset(filename, num_rollouts=10, steps_per_rollout=500, record_every=10, R=3.0, r=1.0, N_theta=64, N_phi=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating dataset on {device}...")
    
    # We use a lower resolution (64x64) by default to keep dataset size manageable for training iterations,
    # but still capturing the essential dynamics.
    simulator = TorusAcousticSimulator(R=R, r=r, N_theta=N_theta, N_phi=N_phi)
    
    P_list = []
    S_list = []
    
    for i in tqdm(range(num_rollouts), desc="Simulating Rollouts"):
        # Randomize source location and time
        t0 = np.random.uniform(0.01, 0.1)
        theta0 = np.random.uniform(0, 2*np.pi)
        phi0 = np.random.uniform(0, 2*np.pi)
        sigma_s = np.random.uniform(0.3, 0.8)
        
        def random_source_fn(t, dev):
            return simulator.generate_kicker_pulse(
                t, t0=t0, sigma_t=0.01, theta0=theta0, phi0=phi0, 
                sigma_s=sigma_s, amplitude=torch.tensor([1.0], device=dev), device=dev
            )
            
        P_seq, S_seq = simulator.simulate(num_steps=steps_per_rollout, source_generator_fn=random_source_fn, device=device, record_every=record_every, channels=1)
        P_list.append(P_seq)
        S_list.append(S_seq)
        
    P_all = torch.cat(P_list, dim=0) # (num_rollouts, Time, C, H, W)
    S_all = torch.cat(S_list, dim=0) # (num_rollouts, Time, C, H, W)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    save_simulation_to_h5(P_all, S_all, filename, R, r, simulator.dt, N_theta, N_phi)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='src/data/simulation_results_diverse.h5', help='Output dataset path')
    parser.add_argument('--rollouts', type=int, default=20, help='Number of independent simulations')
    parser.add_argument('--steps', type=int, default=500, help='Timesteps per simulation')
    parser.add_argument('--N_theta', type=int, default=64, help='Grid resolution theta')
    parser.add_argument('--N_phi', type=int, default=64, help='Grid resolution phi')
    args = parser.parse_args()
    
    generate_multi_rollout_dataset(args.output, num_rollouts=args.rollouts, steps_per_rollout=args.steps, 
                                   N_theta=args.N_theta, N_phi=args.N_phi)
