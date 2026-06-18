import argparse
import os
import torch
import numpy as np
import tqdm
import yaml

from src.numerical.solver import TorusAcousticSimulator
from src.data.dataset import ChunkedTorusDataset
from src.models.geofno import GeoFNO
from src.training.trainer import FastTrainer
from torch.utils.data import DataLoader, Subset

def load_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

# =====================================================================
# 0. DATASET GENERATION (SUPERPOSITION & MULTI-PULSE)
# =====================================================================
def generate_multi_rollout_dataset(filename, num_rollouts=50, steps_per_rollout=512, record_every=10, R=3.0, r=1.0, N_theta=64, N_phi=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating Multi-Pulse Dataset on {device}...")
    simulator = TorusAcousticSimulator(R=R, r=r, N_theta=N_theta, N_phi=N_phi, c=1.0)

    P_list, S_list = [], []

    for i in tqdm.tqdm(range(num_rollouts), desc="Simulating Rollouts"):
        pulses = []
        num_pulses = np.random.randint(6, 12)

        for p in range(num_pulses):
            pulses.append({
                't0': np.random.uniform(0.01, 0.1),
                'theta0': np.random.uniform(0, 2*np.pi),
                'phi0': np.random.uniform(0, 2*np.pi),
                'sigma_s': np.random.uniform(0.2, 0.5),
                'amplitude': np.random.uniform(0.5, 1.5)
            })

        def multi_pulse_source_fn(t, dev):
            total_S = torch.zeros((3, N_theta, N_phi), device=dev)
            for p_params in pulses:
                total_S += simulator.generate_kicker_pulse(
                    t, t0=p_params['t0'], sigma_t=0.01,
                    theta0=p_params['theta0'], phi0=p_params['phi0'],
                    sigma_s=p_params['sigma_s'],
                    amplitude=torch.tensor([p_params['amplitude']]*3, device=dev),
                    device=dev
                )
            return total_S

        P_seq, S_seq = simulator.simulate(num_steps=steps_per_rollout, source_generator_fn=multi_pulse_source_fn, device=device, record_every=record_every)
        P_list.append(P_seq.cpu())
        S_list.append(S_seq.cpu())

    P_all = torch.cat(P_list, dim=0)
    S_all = torch.cat(S_list, dim=0)
    simulator.save_to_h5(P_all, S_all, filename)


import json
from typing import Optional, Dict, Any

# =====================================================================
# 1. TRAINING PIPELINE
# =====================================================================
def run_fast_pipeline(h5_file_path: str, config: Optional[Dict[str, Any]] = None) -> None:
    if config is None:
        config = {}
        
    T_IN = config.get("t_in", 3)
    T_OUT = config.get("t_out", 30)
    UNROLL_STEPS = config.get("unroll_steps", 1)
    BATCH_SIZE = config.get("batch_size", 8)
    EPOCHS = config.get("epochs", 50)
    LR = float(config.get("lr", 5e-4))
    WD = float(config.get("wd", 1e-5))
    MODES = config.get("modes", 20)
    WIDTH = config.get("width", 48)

    print("Loading dataset ...")
    full_ds = ChunkedTorusDataset(h5_file_path, t_in=T_IN, t_out=T_OUT, unroll_steps=UNROLL_STEPS)
    print(f"Original dataset:")
    print(f"  Rollouts: {full_ds.num_rollouts}, time steps: {full_ds.time_steps}")
    print(f"  T_in: {T_IN}, T_out: {T_OUT}, Unroll: {UNROLL_STEPS}, chunk size: {full_ds.chunk_size}")
    print(f"  Valid windows per rollout: {full_ds.valid_starts}")
    print(f"  Total windows: {len(full_ds)}")
    print(f"  Grid: {full_ds.N_theta} x {full_ds.N_phi}")
    print(f"  Global scaling: p_scale={full_ds.p_scale:.3f}, s_scale={full_ds.s_scale:.3f}")

    n_rollouts = full_ds.num_rollouts
    rollout_indices = np.arange(n_rollouts)
    
    if n_rollouts < 2:
        train_roll = rollout_indices
        val_roll = rollout_indices
    else:
        np.random.seed(42)
        shuffled = np.random.permutation(n_rollouts)
        split_idx = int(n_rollouts * 0.8)
        train_roll = shuffled[:split_idx]
        val_roll = shuffled[split_idx:]

    vs = full_ds.valid_starts
    train_win = [r * vs + t for r in train_roll for t in range(vs)]
    val_win   = [r * vs + t for r in val_roll   for t in range(vs)]

    train_ds = Subset(full_ds, train_win)
    val_ds   = Subset(full_ds, val_win)

    print(f"\nStratified split (by rollout):")
    print(f"  Train rollouts: {len(train_roll)}, windows: {len(train_ds)}")
    print(f"  Val   rollouts: {len(val_roll)}, windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              pin_memory=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              pin_memory=True, num_workers=2)

    model = GeoFNO(modes=MODES, width=WIDTH, t_in=T_IN, t_out=T_OUT, geom_channels=3,
                   n_theta=full_ds.N_theta, n_phi=full_ds.N_phi)
    print(f"\nModel: GeoFNO | params: {sum(p.numel() for p in model.parameters()):,}")

    trainer = FastTrainer(model, lr=LR, weight_decay=WD, unroll_steps=UNROLL_STEPS)
    print(f"Training on {trainer.device.type.upper()}\n")

    dataset_meta = {
       "t_in": T_IN,
       "t_out": T_OUT,
       "modes": MODES,
       "width": WIDTH,
       "unroll_steps": UNROLL_STEPS,
       "dt_macro": full_ds.dt_macro,
       "c": full_ds.c,
       "r": full_ds.r,
       "R": full_ds.R,
       "model_filename": f"best_geofno_small_{MODES}_{WIDTH}.pt",
    }

    trainer.fit(
        train_loader, val_loader,
        total_target_epochs=EPOCHS,
        dataset_meta=dataset_meta,
        save_best_path=f"./best_geofno_small_{MODES}_{WIDTH}.pt",
        save_every=20,
        checkpoint_dir="./checkpoints",
        print_every=5
    )

    torch.save({
        'model_state_dict': model.state_dict(),
        'trainer_history': trainer.history,
    }, f"./geofno_small_{MODES}_{WIDTH}.pt")
    print("\nFinal model saved.")

    # Decouple Web Application configuration by exporting the metadata directly
    web_dir = os.path.join(os.path.dirname(__file__), "web", "public")
    os.makedirs(web_dir, exist_ok=True)
    try:
        with open(os.path.join(web_dir, "sim_config.json"), "w") as f:
            json.dump(dataset_meta, f, indent=4)
        print(f"  └─> Web decoupled sim_config.json exported to {web_dir}")
    except Exception as e:
        print(f"  └─> [WARNING] Failed to export web config: {e}")
    print("Done.")


from src.evaluation.geofno_evaluator import load_and_evaluate_geofno

# =====================================================================
# 2. EVALUATION PIPELINE
# =====================================================================
def evaluate_geofno(h5_file_path, model_path, config=None):
    load_and_evaluate_geofno(model_path, h5_file_path, config=config, target_idx=config.get("target_idx", 45))


# =====================================================================
# CLI ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NOMAD Pipeline Orchestrator")
    parser.add_argument("--prepare", action="store_true", help="Prepare the dataset")
    parser.add_argument("--train", action="store_true", help="Train the GeoFNO model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the GeoFNO model")
    parser.add_argument("--all", action="store_true", help="Run the entire pipeline sequentially")
    
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    # Optional direct overrides
    parser.add_argument("--data_path", type=str, help="Path to HDF5 dataset")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    
    args = parser.parse_args()
    config = load_config(args.config)
    
    # Overrides from CLI
    if args.data_path:
        config["data_path"] = args.data_path
    if args.epochs is not None:
        if "training" not in config:
            config["training"] = {}
        config["training"]["epochs"] = args.epochs
        
    data_path = config.get("data_path", "torus_simulation_data_multipulse_256_bench.h5")
    
    if args.all:
        args.prepare = True
        args.train = True
        args.evaluate = True
        
    if not (args.prepare or args.train or args.evaluate):
        print("No action specified. Use --help for options. Running full pipeline by default for testing...")
        args.prepare = True
        args.train = True
        args.evaluate = True
    
    if args.prepare:
        gen_cfg = config.get("generation", {})
        num_rollouts = gen_cfg.get("num_rollouts", 5)
        train_res = gen_cfg.get("train_res", 256)
        scale_factor = train_res / 64.0
        sync_record_every = int(gen_cfg.get("record_every_base", 5) * scale_factor)
        steps = int(gen_cfg.get("steps_base", 1500) * scale_factor)
        
        if os.path.exists(data_path):
            print(f"Dataset already exists at {data_path}. Skipping generation.")
        else:
            generate_multi_rollout_dataset(
                data_path, 
                num_rollouts=num_rollouts,
                steps_per_rollout=steps,
                record_every=sync_record_every,
                R=gen_cfg.get("R", 3.0),
                r=gen_cfg.get("r", 1.0),
                N_theta=train_res,
                N_phi=train_res
            )
        
    if args.train:
        train_cfg = config.get("training", {})
        run_fast_pipeline(data_path, config=train_cfg)
        
    if args.evaluate:
        train_cfg = config.get("training", {})
        eval_cfg = config.get("evaluation", {})
        
        eval_data_path = eval_cfg.get("data_path", "torus_simulation_data_eval.h5")
        eval_res = eval_cfg.get("eval_res", 256)
        
        if not os.path.exists(eval_data_path):
            print(f"\nEvaluation dataset not found at {eval_data_path}. Generating now...")
            gen_cfg = config.get("generation", {})
            num_rollouts = eval_cfg.get("num_rollouts", 2)
            scale_factor = eval_res / 64.0
            sync_record_every = int(gen_cfg.get("record_every_base", 5) * scale_factor)
            steps = int(gen_cfg.get("steps_base", 1500) * scale_factor)
            
            generate_multi_rollout_dataset(
                eval_data_path, 
                num_rollouts=num_rollouts,
                steps_per_rollout=steps,
                record_every=sync_record_every,
                R=gen_cfg.get("R", 3.0),
                r=gen_cfg.get("r", 1.0),
                N_theta=eval_res,
                N_phi=eval_res
            )
        else:
            print(f"\nEvaluation dataset found at {eval_data_path}.")

        modes = train_cfg.get("modes", 24)
        width = train_cfg.get("width", 128)
        model_path = config.get("model_path", f"./best_geofno_small_{modes}_{width}.pt")
        
        # Merge training and evaluation configs so the evaluator gets everything it needs
        merged_cfg = {**train_cfg, **eval_cfg}
        
        from src.evaluation.geofno_evaluator import load_and_evaluate_geofno
        load_and_evaluate_geofno(model_path, eval_data_path, config=merged_cfg, target_idx=eval_cfg.get("target_idx", 1))
