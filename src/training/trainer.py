import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
import sys

class FastTrainer:
    def __init__(self, model, lr=1e-3, weight_decay=1e-5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.scaler = GradScaler('cuda', enabled=(self.device.type == 'cuda'))

        # State Tracking
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.start_epoch = 1

    def load_checkpoint(self, checkpoint_path):
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

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for batch_idx, (p_in, s_in, geom, p_target) in enumerate(loader):
            p_in, s_in = p_in.to(self.device), s_in.to(self.device)
            geom, p_target = geom.to(self.device), p_target.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            with autocast(self.device.type, enabled=(self.device.type == 'cuda')):
                p_pred = self.model(p_in, s_in, geom)
                loss = self.criterion(p_pred, p_target)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            if (batch_idx+1) % max(1, len(loader)//5) == 0 or batch_idx+1 == len(loader):
                sys.stdout.write(f"\r  Batch {batch_idx+1:03d}/{len(loader)} | Active Rolling MSE: {loss.item():.6f}")
                sys.stdout.flush()
        print()
        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        for p_in, s_in, geom, p_target in loader:
            p_in, s_in = p_in.to(self.device), s_in.to(self.device)
            geom, p_target = geom.to(self.device), p_target.to(self.device)
            with autocast(self.device.type, enabled=(self.device.type == 'cuda')):
                p_pred = self.model(p_in, s_in, geom)
                loss = self.criterion(p_pred, p_target)
            total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, train_loader, val_loader, total_target_epochs, dataset_meta=None,
            save_best_path="./best_geofno.pt", save_every=50,
            checkpoint_dir="./checkpoints", print_every=1):

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Adjust scheduler to account for resumed epochs if starting mid-way
        last_epoch_idx = self.start_epoch - 2 if self.start_epoch > 1 else -1
        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_target_epochs, last_epoch=last_epoch_idx)

        if dataset_meta is None:
            dataset_meta = {"warning": "No dataset metadata provided during training."}

        print(f"=== Initiating Training Loop (Targeting {total_target_epochs} Total Epochs) ===")

        for epoch in range(self.start_epoch, total_target_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader) if val_loader is not None else float('nan')

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            scheduler.step()

            if epoch % print_every == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"[Epoch {epoch:04d}/{total_target_epochs}] LR: {lr:.2e} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")

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

        print(f"\n=== Execution Terminated. Global Best Val Loss: {self.best_val_loss:.6f} (Achieved at Epoch {self.best_epoch}) ===")
