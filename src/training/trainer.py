import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import os
import sys
from typing import Dict, Any, Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARNING] mlflow is not installed. Experiment tracking will be disabled.")

class FastTrainer:
    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-5, unroll_steps: int = 1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.scaler = GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        self.unroll_steps = unroll_steps

        # State Tracking
        self.history: Dict[str, list] = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.start_epoch = 1

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Restores model, optimizer, scaler, and history from a serialized state."""
        if not os.path.exists(checkpoint_path):
            print(f"[WARNING] Target checkpoint missing at: {checkpoint_path}. Initializing fresh weights.")
            return

        print(f"\n[SYSTEM] Mounting CHORUS model checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 1. Restore Network & Optimizer States
        self.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # 2. Restore Historical Tracking
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch = checkpoint.get('epoch', 0)

        if 'history' in checkpoint:
            self.history = checkpoint['history']

        print(f"  └─> Architecture restored. Resuming optimization from Epoch {self.start_epoch}.")
        print(f"  └─> Previous Best Validation Loss: {self.best_val_loss:.6f}\n")

    def train_epoch(self, loader: DataLoader) -> float:
        """
        Executes a single training epoch using Pushforward Autoregressive Training.

        --- MATHEMATICAL FORMULATION ---
        Given a dynamical system, a standard 1-step predictor minimizes:
           L = || P_pred(t_1...t_out) - P_true(t_1...t_out) ||^2
        where P_pred is generated from the true past P_true(-t_in...0).
        
        However, during pure autoregressive evaluation, the model feeds its own predictions
        as input for future steps. Any small approximation error epsilon compounding recursively
        leads to exponential drift.

        Pushforward Training unrolls the model dynamically during the training phase.
        For an unroll_step = K:
        1) k = 1: 
             P_pred_1 = Model(P_true_in, S_in_1)
             Loss_1 = || P_pred_1 - P_true_target_1 ||^2
        2) k = 2:
             We construct a new input by concatenating [P_true_in[t_out:], P_pred_1].
             P_pred_2 = Model(P_hybrid_in, S_in_2)
             Loss_2 = || P_pred_2 - P_true_target_2 ||^2
        ...
        3) Total Loss = Sum(Loss_k) for k=1..K

        This forces the FNO to learn to map slightly perturbed, erroneous inputs (P_pred_k)
        back to the true manifold, stabilizing the autoregressive trajectory. If unroll_steps=1,
        this collapses perfectly back into standard one-step supervised training.
        --------------------------------
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (p_in, s_unrolled, geom, p_target) in enumerate(loader):
            # p_in: [B, t_in, H, W]
            # s_unrolled: [B, t_in + unroll_steps * t_out, H, W]
            # geom: [B, 3, H, W]
            # p_target: [B, unroll_steps * t_out, H, W]
            
            p_in = p_in.to(self.device)
            s_unrolled = s_unrolled.to(self.device)
            geom = geom.to(self.device)
            p_target = p_target.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            
            batch_loss = 0.0
            current_p_in = p_in
            t_in = p_in.size(1)
            t_out = p_target.size(1) // self.unroll_steps

            with autocast(self.device.type, enabled=(self.device.type == 'cuda')):
                for k in range(self.unroll_steps):
                    # Slice the correct portion of the source for this step.
                    # The source input needs to match the temporal window of the current input
                    # plus the target horizon: length = t_in + t_out.
                    # At step k, the window starts at k * t_out
                    s_step = s_unrolled[:, k * t_out : k * t_out + t_in + t_out]
                    
                    # Target for step k
                    target_step = p_target[:, k * t_out : (k + 1) * t_out]
                    
                    # Predict future
                    p_pred = self.model(current_p_in, s_step, geom)
                    
                    # Accumulate loss
                    step_loss = self.criterion(p_pred, target_step)
                    
                    # Guard against NaN divergence immediately
                    if not torch.isfinite(step_loss):
                        raise RuntimeError(f"NaN loss encountered at unroll step {k+1}. Check learning rate or data scaling.")
                        
                    batch_loss += step_loss
                    
                    # Pushforward: Construct next input by shifting time window
                    if k < self.unroll_steps - 1:
                        if t_in > t_out:
                            # Keep the last (t_in - t_out) steps of the current input, append the new prediction
                            current_p_in = torch.cat([current_p_in[:, t_out:], p_pred], dim=1)
                        else:
                            # If t_out >= t_in, we just take the last t_in steps of the prediction
                            current_p_in = p_pred[:, -t_in:]

            # Backward pass on accumulated loss
            self.scaler.scale(batch_loss / self.unroll_steps).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += (batch_loss.item() / self.unroll_steps)
            
            if (batch_idx + 1) % max(1, len(loader) // 5) == 0 or batch_idx + 1 == len(loader):
                sys.stdout.write(f"\r  Batch {batch_idx+1:03d}/{len(loader)} | Active Rolling MSE: {batch_loss.item() / self.unroll_steps:.6f}")
                sys.stdout.flush()
        print()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        for p_in, s_unrolled, geom, p_target in loader:
            p_in = p_in.to(self.device)
            s_unrolled = s_unrolled.to(self.device)
            geom = geom.to(self.device)
            p_target = p_target.to(self.device)
            
            t_in = p_in.size(1)
            t_out = p_target.size(1) // self.unroll_steps

            with autocast(self.device.type, enabled=(self.device.type == 'cuda')):
                # For validation, we evaluate on the FIRST step only to keep metrics consistent 
                # and directly comparable across different unroll_steps configurations.
                s_step = s_unrolled[:, 0 : t_in + t_out]
                target_step = p_target[:, 0 : t_out]
                
                p_pred = self.model(p_in, s_step, geom)
                loss = self.criterion(p_pred, target_step)
                
            total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader], total_target_epochs: int, 
            dataset_meta: Optional[Dict[str, Any]] = None, save_best_path: str = "./best_geofno.pt", 
            save_every: int = 50, checkpoint_dir: str = "./checkpoints", print_every: int = 1) -> None:

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Adjust scheduler to account for resumed epochs if starting mid-way
        last_epoch_idx = self.start_epoch - 2 if self.start_epoch > 1 else -1
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_target_epochs, last_epoch=last_epoch_idx)

        if dataset_meta is None:
            dataset_meta = {"warning": "No dataset metadata provided during training."}

        print(f"=== Initiating Training Loop (Targeting {total_target_epochs} Total Epochs, Unroll={self.unroll_steps}) ===")

        # MLflow Tracking
        mlflow_run = None
        if MLFLOW_AVAILABLE:
            mlflow.set_experiment("NOMAD_GeoFNO_Acoustics")
            mlflow_run = mlflow.start_run(run_name=f"GeoFNO_Unroll_{self.unroll_steps}")
            mlflow.log_params({
                "unroll_steps": self.unroll_steps,
                "initial_lr": self.optimizer.param_groups[0]['initial_lr'] if 'initial_lr' in self.optimizer.param_groups[0] else self.optimizer.param_groups[0]['lr'],
                "weight_decay": self.optimizer.param_groups[0]['weight_decay'],
                "total_epochs": total_target_epochs,
                "dataset_meta": str(dataset_meta)
            })

        try:
            for epoch in range(self.start_epoch, total_target_epochs + 1):
                train_loss = self.train_epoch(train_loader)
                val_loss = self.evaluate(val_loader) if val_loader is not None else float('nan')

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                
                lr = scheduler.get_last_lr()[0]
                scheduler.step()

                if epoch % print_every == 0:
                    print(f"[Epoch {epoch:04d}/{total_target_epochs}] LR: {lr:.2e} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

                if MLFLOW_AVAILABLE:
                    mlflow.log_metrics({
                        "train_mse": train_loss,
                        "val_mse": val_loss,
                        "learning_rate": lr
                    }, step=epoch)

                save_payload = {
                    'project': 'CHORUS_Operator_Mapping',
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'history': self.history,
                    'best_val_loss': self.best_val_loss,
                    'dataset_configuration': dataset_meta
                }

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    save_payload['best_val_loss'] = val_loss
                    torch.save(save_payload, save_best_path)
                    print(f"  └─> [UPDATE] New historical best model archived. (Val Loss: {val_loss:.6f})")

                if epoch % save_every == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"geofno_epoch_{epoch}.pt")
                    torch.save(save_payload, ckpt_path)
                    print(f"  └─> [SNAPSHOT] Periodic state saved to {ckpt_path}")

        finally:
            if MLFLOW_AVAILABLE and mlflow_run is not None:
                mlflow.end_run()

        print(f"\n=== Execution Terminated. Global Best Val Loss: {self.best_val_loss:.6f} (Achieved at Epoch {self.best_epoch}) ===")

