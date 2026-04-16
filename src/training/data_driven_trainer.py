import torch
import torch.nn as nn
import torch.optim as optim

class DataDrivenTrainer:
    """
    Autoregressive Multi-Step Trainer for Data-Driven PDE discovery.
    Calculates cumulative loss over a rollout window to force the model to capture
    long-term physical dynamics rather than memorizing single-step lazy transitions.
    """
    def __init__(self, model, lr=1e-4, weight_decay=1e-5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader, teacher_forcing_ratio=0.0):
        self.model.train()
        total_loss = 0.0

        for s_seq, p_seq, m_seq in dataloader:
            s_seq = s_seq.to(self.device)
            p_seq = p_seq.to(self.device)
            m_seq = m_seq.to(self.device)

            seq_len = p_seq.shape[1]
            if seq_len < 3:
                continue

            # Initial inputs (requires 2 history frames to encode velocity for wave PDE)
            # Shapes: (Batch, Channels, H, W) where Channels=1 usually.
            p_prev = p_seq[:, 0]
            p_curr = p_seq[:, 1]
            
            # Static metric embed (same across time)
            m_static = m_seq[:, 0]

            batch_loss = 0.0

            # Autoregressive Rollout
            for t in range(2, seq_len):
                s_curr = s_seq[:, t-1] # Source applied at t-1 dictates change leading to t
                
                # Model input: [P(t-1), P(t-2), S(t-1), M] -> predicts P(t)
                x_in = torch.cat([p_curr, p_prev, s_curr, m_static], dim=1)
                
                p_next_pred = self.model(x_in)
                p_next_target = p_seq[:, t]
                
                # Accumulate loss
                batch_loss += self.criterion(p_next_pred, p_next_target)
                
                # Roll state forward
                p_prev = p_curr
                
                # Teacher forcing? Sometime we use ground truth to stabilize early training
                use_teacher = torch.rand(1).item() < teacher_forcing_ratio
                p_curr = p_next_target if use_teacher else p_next_pred

            # Average loss over the rollout steps so it doesn't artificially inflate gradients
            batch_loss = batch_loss / (seq_len - 2)

            self.optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += batch_loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for s_seq, p_seq, m_seq in dataloader:
                s_seq = s_seq.to(self.device)
                p_seq = p_seq.to(self.device)
                m_seq = m_seq.to(self.device)

                seq_len = p_seq.shape[1]
                if seq_len < 3:
                    continue

                p_prev = p_seq[:, 0]
                p_curr = p_seq[:, 1]
                m_static = m_seq[:, 0]

                batch_loss = 0.0

                for t in range(2, seq_len):
                    s_curr = s_seq[:, t-1]
                    
                    x_in = torch.cat([p_curr, p_prev, s_curr, m_static], dim=1)
                    p_next_pred = self.model(x_in)
                    p_next_target = p_seq[:, t]
                    
                    batch_loss += self.criterion(p_next_pred, p_next_target)
                    
                    p_prev = p_curr
                    p_curr = p_next_pred

                batch_loss = batch_loss / (seq_len - 2)
                total_loss += batch_loss.item()

        return total_loss / len(dataloader)

    def train_epochs(self, train_loader, val_loader=None, epochs=10, initial_teacher_forcing=0.5):
        for epoch in range(epochs):
            # Anneal teacher forcing over time
            tf_ratio = initial_teacher_forcing * (1.0 - epoch/epochs)
            train_loss = self.train_epoch(train_loader, teacher_forcing_ratio=tf_ratio)
            
            if val_loader:
                val_loss = self.evaluate(val_loader)
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}")
