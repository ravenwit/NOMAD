import argparse
import os
import torch
import numpy as np

# Import from refactored src modules
from src.numerical.solver import TorusAcousticSimulator
from src.data.dataset import ChunkedTorusDataset
from src.models.geofno import GeoFNO
from src.training.trainer import FastTrainer
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def prepare_dataset(h5_file_path, num_rollouts=10, time_steps=200):
    print(f"=== Step 1: Prepare Dataset ===")
    if os.path.exists(h5_file_path):
        print(f"Dataset already exists at {h5_file_path}. Skipping generation.")
        return

    print(f"Generating data with {num_rollouts} rollouts of {time_steps} steps...")
    
    # ------------------------------------------------------------------
    # BOILERPLATE FOR DATA GENERATION
    # User: You can add your customized dataset generation logic here.
    # ------------------------------------------------------------------
    simulator = TorusAcousticSimulator(R=3.0, r=1.0, c=343.0, N_theta=64, N_phi=64)
    P_all, S_all = [], []
    for i in range(num_rollouts):
        # random source position
        theta0 = np.random.uniform(0, 2*np.pi)
        phi0 = np.random.uniform(0, 2*np.pi)
        
        def source_fn(t, device):
            return simulator.generate_ricker_pulse(t, t0=0.01, sigma_t=0.005, 
                                                  theta0=theta0, phi0=phi0, 
                                                  sigma_s=0.2, device=device)
        
        P, S = simulator.simulate(num_steps=time_steps, source_generator_fn=source_fn, 
                                  device='cuda' if torch.cuda.is_available() else 'cpu', record_every=2)
        P_all.append(P)
        S_all.append(S)
    
    if len(P_all) > 0:
        P_tensor = torch.cat(P_all, dim=0)
        S_tensor = torch.cat(S_all, dim=0)
        simulator.save_to_h5(P_tensor, S_tensor, h5_file_path)
    else:
        print("No rollouts generated. Check your parameters.")

def train_geofno(h5_file_path, epochs=50, batch_size=8):
    print(f"\n=== Step 2: Train GeoFNO ===")
    if not os.path.exists(h5_file_path):
        print(f"Error: Dataset {h5_file_path} not found. Please run --prepare first.")
        return

    T_IN = 3
    T_OUT = 30
    LR = 1e-3
    WD = 1e-5

    print("Loading dataset ...")
    full_ds = ChunkedTorusDataset(h5_file_path, t_in=T_IN, t_out=T_OUT)
    
    n_rollouts = full_ds.num_rollouts
    rollout_indices = np.arange(n_rollouts)
    
    if n_rollouts < 2:
        train_roll = rollout_indices
        val_roll = rollout_indices
    else:
        train_roll, val_roll = train_test_split(rollout_indices, test_size=0.2, random_state=42)

    vs = full_ds.valid_starts
    train_win = [r * vs + t for r in train_roll for t in range(vs)]
    val_win   = [r * vs + t for r in val_roll   for t in range(vs)]

    train_ds = Subset(full_ds, train_win)
    val_ds   = Subset(full_ds, val_win)

    bs = min(batch_size, len(train_ds))
    if bs == 0: bs = 1
         
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, pin_memory=True, num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False, pin_memory=True, num_workers=0)

    model = GeoFNO(modes=20, width=48, t_in=T_IN, t_out=T_OUT, geom_channels=3,
                   n_theta=full_ds.N_theta, n_phi=full_ds.N_phi)
    
    trainer = FastTrainer(model, lr=LR, weight_decay=WD)
    
    trainer.fit(
        train_loader, val_loader,
        total_target_epochs=epochs,
        save_best_path="./best_geofno.pt",
        save_every=10,
        checkpoint_dir="./checkpoints",
        print_every=1
    )
    
    torch.save({'model_state_dict': model.state_dict(), 'trainer_history': trainer.history}, "./geofno_final.pt")
    print("Training complete.")

def evaluate_geofno(h5_file_path, model_path="./best_geofno.pt"):
    print(f"\n=== Step 3: Evaluate GeoFNO ===")
    if not os.path.exists(h5_file_path):
        print(f"Error: Dataset {h5_file_path} not found.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint {model_path} not found. Please run --train first.")
        return

    T_IN = 3
    T_OUT = 30
    
    print("Loading test data...")
    full_ds = ChunkedTorusDataset(h5_file_path, t_in=T_IN, t_out=T_OUT)
    test_loader = DataLoader(full_ds, batch_size=8, shuffle=False)
    
    model = GeoFNO(modes=20, width=48, t_in=T_IN, t_out=T_OUT, geom_channels=3,
                   n_theta=full_ds.N_theta, n_phi=full_ds.N_phi)
    
    trainer = FastTrainer(model)
    trainer.load_checkpoint(model_path)
    
    val_loss = trainer.evaluate(test_loader)
    print(f"Evaluation MSE on entire dataset: {val_loss:.6f}")
    
    # User: You can add more detailed evaluation metrics, visualizations, or save predictions here.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NOMAD Pipeline Orchestrator")
    parser.add_argument("--prepare", action="store_true", help="Prepare the dataset")
    parser.add_argument("--train", action="store_true", help="Train the GeoFNO model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the GeoFNO model")
    parser.add_argument("--data_path", type=str, default="torus_wave_data.h5", help="Path to HDF5 dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline sequentially")
    
    args = parser.parse_args()
    
    if args.all:
        args.prepare = True
        args.train = True
        args.evaluate = True
        
    if not (args.prepare or args.train or args.evaluate):
        print("No action specified. Use --help for options. Running full pipeline by default for testing...")
        args.prepare = True
        args.train = True
        args.epochs = 1
    
    if args.prepare:
        prepare_dataset(args.data_path, num_rollouts=2, time_steps=100) # Using small defaults for testing
        
    if args.train:
        train_geofno(args.data_path, epochs=args.epochs)
        
    if args.evaluate:
        evaluate_geofno(args.data_path, model_path="./best_geofno.pt")
