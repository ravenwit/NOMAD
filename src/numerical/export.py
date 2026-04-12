import h5py
import torch
import os

def export_simulation_to_h5(P_seq: torch.Tensor, S_seq: torch.Tensor, 
                            R: float, r: float, dt: float, 
                            N_theta: int, N_phi: int,
                            filename: str = "data/torus_numerical_sim.h5"):
    """
    Exports High-Fidelity numerical simulation tensor data to an HDF5 archive.
    P_seq: (Batch, Time, Channels, N_theta, N_phi)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with h5py.File(filename, 'w') as f:
        # Move channel dimension to the end for web/JS convenience
        # From (B, T, C, H, W) -> (B, T, H, W, C)
        P_save = P_seq.permute(0, 1, 3, 4, 2).numpy()
        S_save = S_seq.permute(0, 1, 3, 4, 2).numpy()
        
        f.create_dataset('pressure', data=P_save, compression="gzip", compression_opts=4)
        f.create_dataset('source', data=S_save, compression="gzip", compression_opts=4)
        
        f.attrs['R'] = R
        f.attrs['r'] = r
        f.attrs['dt'] = dt
        f.attrs['N_theta'] = N_theta
        f.attrs['N_phi'] = N_phi
        f.attrs['channels'] = P_seq.shape[2]
        
    print(f"Successfully exported simulation data to {filename}")
