import torch
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
from src.data.dataset import ChunkedTorusDataset
from src.models.geofno import GeoFNO

# =====================================================================
# 1. GEO-FNO ARCHITECTURAL EVALUATOR
# =====================================================================
class GeoFNOEvaluator:
    """
    Executes chunked autoregressive evaluation rollouts for the GeoFNO
    and computes relative L2 metrics against Spectral Ground Truth.
    """
    def __init__(self, model, dataset, t_in=3, t_out=30, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.dataset = dataset
        self.t_in = t_in
        self.t_out = t_out

    @torch.no_grad()
    def generate_rollout(self, rollout_idx=0, max_steps=90):
        """
        Propagates the state forward in chunks of `t_out`.
        """
        # 1. Extract full sequence matrices and scale them
        # P_raw shape in dataset is (N_rollouts, Time, H, W)
        S_full = self.dataset.S_raw[rollout_idx].unsqueeze(0).to(self.device) / self.dataset.s_scale
        P_full = self.dataset.P_raw[rollout_idx].unsqueeze(0).to(self.device) / self.dataset.p_scale

        geom_features = self.dataset.geom_features.unsqueeze(0).to(self.device)

        # 2. Seed the model using the initial ground truth window
        p_curr = P_full[:, 0:self.t_in]
        predictions = [p_curr.cpu()]

        current_t = self.t_in

        # 3. Chunked Autoregressive Loop
        while current_t < max_steps:
            # Align the source input window to the pressure input window
            s_start = current_t - self.t_in
            s_end = current_t
            if s_end <= S_full.shape[1]:
                s_curr = S_full[:, s_start:s_end]
            else:
                s_curr = torch.zeros_like(p_curr)

            # Forward mapping: predicts t_out steps simultaneously
            with torch.amp.autocast(self.device.type, enabled=(self.device.type == 'cuda')):
                p_next = self.model(p_curr.float(), s_curr.float(), geom_features)

            predictions.append(p_next.cpu())

            # Shift the window: The new seed is the final t_in steps of the prediction
            p_curr = p_next[:, -self.t_in:]
            current_t += self.t_out

        # 4. Assemble and Denormalize
        P_pred_seq = torch.cat(predictions, dim=1).squeeze(0) # (Total_T, H, W)
        P_true_seq = P_full.squeeze(0)[:P_pred_seq.shape[0]].cpu()

        P_pred_seq = P_pred_seq * self.dataset.p_scale
        P_true_seq = P_true_seq * self.dataset.p_scale

        # Crop exactly to requested max_steps
        return P_true_seq[:max_steps], P_pred_seq[:max_steps]

    def compute_relative_l2_error(self, P_true, P_pred):
        """
        Calculates the relative L2 error norm over each temporal layer.
        """
        true_flat = P_true.reshape(P_true.shape[0], -1)
        pred_flat = P_pred.reshape(P_pred.shape[0], -1)

        error_norm = torch.linalg.norm(pred_flat - true_flat, dim=1)
        true_norm = torch.linalg.norm(true_flat, dim=1)

        return (error_norm / torch.clamp(true_norm, min=1e-8)).numpy()

# =====================================================================
# 2. VISUAL ANALYTICS AND ERROR PLOTTER
# =====================================================================
def visualize_geofno_rollout(evaluator, rollout_idx=45, max_steps=55, num_snapshots=5):
    """
    Generates side-by-side matrix comparisons for the new tensor shapes.
    """
    P_true, P_pred = evaluator.generate_rollout(rollout_idx=rollout_idx, max_steps=max_steps)

    P_true = P_true.numpy()
    P_pred = P_pred.numpy()

    abs_error = np.abs(P_pred - P_true)

    # Start snapshots right after the initial seed window
    t_indices = np.linspace(evaluator.t_in, max_steps - 1, num_snapshots, dtype=int)

    fig, axes = plt.subplots(3, num_snapshots, figsize=(4 * num_snapshots, 9.5))

    for i, t in enumerate(t_indices):
        true_field = P_true[t]
        pred_field = P_pred[t]
        err_field = abs_error[t]

        vmax = np.max(np.abs(true_field)) + 1e-9
        vmin = -vmax

        # Row 1: Ground Truth
        ax_true = axes[0, i]
        im_true = ax_true.imshow(true_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        ax_true.set_title(f"Spectral Target (t={t})", fontsize=10)
        ax_true.axis('off')

        # Row 2: Geo-FNO Prediction
        ax_pred = axes[1, i]
        im_pred = ax_pred.imshow(pred_field, cmap='RdBu', vmin=vmin, vmax=vmax, aspect='auto')
        ax_pred.set_title(f"Geo-FNO Prediction (t={t})", fontsize=10)
        ax_pred.axis('off')

        # Row 3: Absolute Error
        ax_err = axes[2, i]
        err_max = np.max(err_field) + 1e-9
        im_err = ax_err.imshow(err_field, cmap='magma', vmin=0, vmax=err_max, aspect='auto')
        ax_err.set_title(f"Absolute Error (t={t})", fontsize=10)
        ax_err.axis('off')

    fig.colorbar(im_true, ax=axes[0:2, :].ravel().tolist(), shrink=0.75, label="Pressure Coordinate Field")
    fig.colorbar(im_err, ax=axes[2, :].ravel().tolist(), shrink=0.75, label="Absolute Error Intensity")

    plt.suptitle("Chunked Autoregressive Evaluation: Spectral Solver vs Geo-FNO", fontsize=14, y=0.98)
    plt.show()

    # Render Relative L2 Norm Evolution Graph
    rel_errors = evaluator.compute_relative_l2_error(torch.tensor(P_true), torch.tensor(P_pred))

    plt.figure(figsize=(10, 3.5))
    plt.plot(rel_errors, color='#d62728', linewidth=2.5, label='Relative $L_2$ Error Deviation')
    plt.axhline(0.1, color='gray', linestyle=':', alpha=0.7, label='10% Bound Tolerance')

    # Highlight the initial feed window
    plt.axvspan(0, evaluator.t_in, color='blue', alpha=0.1, label='Initial Seed Window')

    plt.title("Accumulated Global Relative $L_2$ Error Over Temporal Horizon", fontsize=11)
    plt.xlabel("Temporal Frame Step Index ($t$)")
    plt.ylabel("Relative Error Amplitude $\epsilon(t)$")
    plt.xlim(0, max_steps - 1)
    plt.ylim(0, max(1.1, np.max(rel_errors) * 1.1))
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# =====================================================================
# 3. MASTER EXECUTION LOADER
# =====================================================================
def load_and_evaluate_geofno(checkpoint_path, h5_file_path, config=None, target_idx=45):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n=== Step 3: Evaluate GeoFNO ===")
    print(f"Loading environment... Device: {device.type.upper()}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint missing at: {checkpoint_path}")
    if not os.path.exists(h5_file_path):
        raise FileNotFoundError(f"Dataset missing at: {h5_file_path}")

    if config is None:
        config = {}

    T_IN = config.get("t_in", 3)
    T_OUT = config.get("t_out", 30)
    MODES = config.get("modes", 24)
    WIDTH = config.get("width", 128)
    MAX_STEPS = config.get("max_eval_steps", 299)

    # 1. Initialize dataset to grab parameters and geometric embeddings
    eval_dataset = ChunkedTorusDataset(h5_file_path, t_in=T_IN, t_out=T_OUT)

    # 2. Reconstruct the GeoFNO network
    model = GeoFNO(modes=MODES, width=WIDTH, t_in=T_IN, t_out=T_OUT, geom_channels=3,
                   n_theta=eval_dataset.N_theta, n_phi=eval_dataset.N_phi)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    # Delete the old low-res base_grid from the checkpoint so it doesn't overwrite our high-res grid
    if 'base_grid' in state_dict:
        del state_dict['base_grid']

    # Load the weights (strict=False tells PyTorch not to panic about the missing base_grid key)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    # 3. Trigger Evaluation
    evaluator = GeoFNOEvaluator(model, eval_dataset, t_in=T_IN, t_out=T_OUT, device=device)
    
    # Adjust target_idx if we have fewer rollouts than target_idx
    if target_idx >= eval_dataset.num_rollouts:
        print(f"Warning: target_idx {target_idx} is larger than dataset rollouts {eval_dataset.num_rollouts}. Adjusting to {eval_dataset.num_rollouts - 1}")
        target_idx = max(0, eval_dataset.num_rollouts - 1)
        
    visualize_geofno_rollout(evaluator, rollout_idx=target_idx, max_steps=MAX_STEPS)
