import os
import sys
import torch
import numpy as np

# Add project root to path smartly to allow modular execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.geofno import GeoFNO
from src.numerical.solver import TorusAcousticSimulator

print("=" * 80)
print("🎬 CHORUS BAKING STUDIO v5: Raw Binary Generation")
print("=" * 80)

# 1. Configuration
RES = 128
T_IN = 3
T_OUT = 30
SCALE_FACTOR = RES / 64
RECORD_EVERY = int(5 * SCALE_FACTOR)
NUM_STEPS = int(5 * 500 * SCALE_FACTOR)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------------
# THE FIX: STATIC PHYSICAL SCALING
FIXED_P_SCALE = 0.018
FIXED_S_SCALE = 2.7
# -------------------------------------------------------------------

print("1. Booting Model and Simulator...")
model = GeoFNO(modes=24, width=128, t_in=T_IN, t_out=T_OUT, geom_channels=3, n_theta=RES, n_phi=RES).to(DEVICE)

# Safely load weights
checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "best_geofno_resumed.pt"))
try:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if 'base_grid' in state_dict: del state_dict['base_grid']
    model.load_state_dict(state_dict, strict=False)
    print(" ✅ Model weights loaded successfully.")
except FileNotFoundError:
    print(f" ⚠️ Warning: {checkpoint_path} not found. Generating with untrained weights.")
model.eval()

simulator = TorusAcousticSimulator(R=3.0, r=1.0, N_theta=RES, N_phi=RES, c=1.0)

# 2. Geometry
theta = torch.linspace(0, 2*np.pi, RES+1)[:-1]
phi   = torch.linspace(0, 2*np.pi, RES+1)[:-1]
THETA, PHI = torch.meshgrid(theta, phi, indexing='ij')
metric = 1.0 * (3.0 + 1.0 * torch.cos(THETA))
m_norm = (metric - metric.min()) / (metric.max() - metric.min())
geom_tensor = torch.stack([m_norm, THETA/(2*np.pi), PHI/(2*np.pi)], dim=0).float().unsqueeze(0).to(DEVICE)

# 3. Scenario Generator
def bake_complex_scenario(scenario_name, pulse_list):
    print(f"\n--- Baking Scenario: {scenario_name} ---")

    def multi_source_fn(t, device):
        total_source = torch.zeros((3, RES, RES), device=device)
        for p in pulse_list:
            pulse = simulator.generate_ricker_pulse(
                t, t0=p.get('t0', 0.05), sigma_t=p.get('sigma_t', 0.01),
                theta0=p.get('theta0', np.pi), phi0=p.get('phi0', np.pi),
                sigma_s=p.get('sigma_s', 0.3), device=device
            )
            total_source += pulse
        return total_source

    print(" -> Running Ground Truth Simulator...")
    P, S = simulator.simulate(num_steps=NUM_STEPS, source_generator_fn=multi_source_fn,
                              device=DEVICE, record_every=RECORD_EVERY)

    # Extract shape [Time, Theta, Phi]
    P_clean = P[0, :, 0, :, :]
    S_clean = S[0, :, 0, :, :]

    # ALIGNMENT 1: Transpose spatial dims to match dataset's permute(0, 1, 3, 2)
    # Swaps dim 1 (Theta) and dim 2 (Phi)
    P_clean = P_clean.permute(0, 2, 1).cpu()
    S_clean = S_clean.permute(0, 2, 1).cpu()

    p_in = (P_clean[:T_IN] / FIXED_P_SCALE).unsqueeze(0).to(DEVICE)
    s_in = (S_clean[:T_IN] / FIXED_S_SCALE).unsqueeze(0).to(DEVICE)
    p_gt = P_clean[T_IN:]

    print(" -> Running Geo-FNO Prediction...")
    target_length = p_gt.shape[0]
    all_preds = []
    curr_p_in = p_in.clone()
    curr_s_in = s_in.clone()
    frames_predicted = 0

    with torch.no_grad():
        while frames_predicted < target_length:
            p_pred_chunk = model(curr_p_in, curr_s_in, geom_tensor)
            all_preds.append(p_pred_chunk[0].cpu())
            frames_predicted += T_OUT
            curr_p_in = p_pred_chunk[:, -T_IN:, :, :].clone()
            curr_s_in = torch.zeros_like(curr_p_in)

    p_pred_full = torch.cat(all_preds, dim=0)[:target_length]

    p_pred = p_pred_full * FIXED_P_SCALE * 2

    p_gt = p_gt.permute(0, 2, 1)
    p_pred = p_pred.permute(0, 2, 1)

    # ---------------------------------------------------------
    # BINARY EXPORT PROTOCOL
    # ---------------------------------------------------------
    print(" -> Exporting to Flat Binary (.bin)...")

    # Cast to strictly 32-bit floats. This prevents browser rendering bugs.
    frames_gt = p_gt.numpy().astype(np.float32)
    frames_pred = p_pred.numpy().astype(np.float32)

    # Route automatically to the React App's public data folder
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'web', 'public', 'data'))
    os.makedirs(output_dir, exist_ok=True)
    
    gt_filename = os.path.join(output_dir, f"{scenario_name}_gt.bin")
    pred_filename = os.path.join(output_dir, f"{scenario_name}_pred.bin")

    # .tofile() writes the pure C-style contiguous memory block to disk
    frames_gt.tofile(gt_filename)
    frames_pred.tofile(pred_filename)

    gt_mb = os.path.getsize(gt_filename) / (1024 * 1024)
    pred_mb = os.path.getsize(pred_filename) / (1024 * 1024)

    print(f" ✅ Saved {gt_filename} ({gt_mb:.2f} MB)")
    print(f" ✅ Saved {pred_filename} ({pred_mb:.2f} MB)")

# ==============================================================================
# 4. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    quad_pulses = [
        {'t0': 0.05, 'theta0': np.pi/2, 'phi0': np.pi/2},
        {'t0': 0.05, 'theta0': 3*np.pi/2, 'phi0': np.pi/2},
        {'t0': 0.05, 'theta0': np.pi/2, 'phi0': 3*np.pi/2},
        {'t0': 0.05, 'theta0': 3*np.pi/2, 'phi0': 3*np.pi/2},
    ]
    bake_complex_scenario("complex_quad_symmetry", quad_pulses)

    cascade_pulses = [
        {'t0': 0.05, 'theta0': np.pi, 'phi0': np.pi},
        {'t0': 0.15, 'theta0': np.pi, 'phi0': np.pi},
        {'t0': 0.25, 'theta0': np.pi, 'phi0': np.pi},
    ]
    bake_complex_scenario("complex_cascade", cascade_pulses)

    chaos_pulses = [
        {'t0': 0.05, 'theta0': 1.0, 'phi0': 1.0, 'sigma_s': 0.2},
        {'t0': 0.07, 'theta0': 5.0, 'phi0': 2.0, 'sigma_s': 0.4},
        {'t0': 0.12, 'theta0': 3.0, 'phi0': 4.5, 'sigma_s': 0.15},
        {'t0': 0.08, 'theta0': 0.5, 'phi0': 5.5, 'sigma_s': 0.3},
        {'t0': 0.10, 'theta0': 4.0, 'phi0': 0.5, 'sigma_s': 0.25},
    ]
    bake_complex_scenario("complex_chaos", chaos_pulses)
