import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# =============================================================================
# NOMAD Report Figure Generator for Google Colab
# =============================================================================
# Run this script sequentially in Colab cells. 
# Functions are grouped by the sections in `recommended_figures.md`.
# Where actual model data is required, the script falls back to generating 
# realistic synthetic mock data so you can test the visualization pipeline 
# immediately. Replace the mock data arrays with your actual HDF5/Torch tensors.
# =============================================================================

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'figure.titlesize': 18,
    'figure.dpi': 150
})

R_default = 3.0
r_default = 1.0
N_grid = 128

# -----------------------------------------------------------------------------
# 1. Geometric Foundations
# -----------------------------------------------------------------------------
def fig1_geometry_and_metric(R=R_default, r=r_default, N=100):
    """Generates Figure 1: 3D Torus and Metric Determinant Heatmap."""
    fig = plt.figure(figsize=(14, 6))
    
    # --- Plot A: 3D Torus ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    theta = np.linspace(0, 2*np.pi, N)
    phi = np.linspace(0, 2*np.pi, N)
    THETA, PHI = np.meshgrid(theta, phi)
    
    X = (R + r * np.cos(THETA)) * np.cos(PHI)
    Y = (R + r * np.cos(THETA)) * np.sin(PHI)
    Z = r * np.sin(THETA)
    
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
    ax1.set_title("Plot A: 3D Torus Manifold $\mathbb{T}^2$")
    ax1.set_box_aspect([1, 1, r/R])
    ax1.axis('off')
    
    # --- Plot B: Metric Determinant Heatmap ---
    ax2 = fig.add_subplot(1, 2, 2)
    # The metric determinant \sqrt{|g|} = r(R + r\cos\theta)
    sqrt_g = r * (R + r * np.cos(THETA))
    
    c = ax2.contourf(THETA, PHI, sqrt_g, levels=50, cmap='plasma')
    plt.colorbar(c, ax=ax2, label=r'$\sqrt{|g|}$ (Area Element)')
    
    ax2.set_title("Plot B: Metric Determinant $\sqrt{|g|}$")
    ax2.set_xlabel(r"Poloidal Angle $\theta$")
    ax2.set_ylabel(r"Toroidal Angle $\phi$")
    ax2.set_xticks([0, np.pi, 2*np.pi])
    ax2.set_xticklabels(['$0$ (Outer)', '$\pi$ (Inner)', '$2\pi$ (Outer)'])
    ax2.set_yticks([0, np.pi, 2*np.pi])
    ax2.set_yticklabels(['0', '$\pi$', '$2\pi$'])
    
    plt.tight_layout()
    plt.savefig('fig1_torus_geometry.png')
    plt.show()

# -----------------------------------------------------------------------------
# 2. Ricker Wavelet Source
# -----------------------------------------------------------------------------
def fig2_ricker_wavelet(R=R_default, r=r_default, N=N_grid):
    """Generates Figure 2: The zero-mean Ricker Wavelet source in space and time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Plot A: Spatial Profile ---
    theta = np.linspace(0, 2*np.pi, N)
    phi = np.linspace(0, 2*np.pi, N)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    theta0, phi0 = np.pi, np.pi # Center at inner equator
    sigma_s = 0.5
    
    dtheta = (THETA - theta0 + np.pi) % (2*np.pi) - np.pi
    dphi = (PHI - phi0 + np.pi) % (2*np.pi) - np.pi
    
    # Chordal distance approximation on the manifold
    rho_sq = (r * dtheta)**2 + ((R + r * np.cos(theta0)) * dphi)**2
    spatial = (2.0 - rho_sq / sigma_s**2) * np.exp(-rho_sq / (2 * sigma_s**2))
    spatial -= spatial.mean() # Enforce zero-mean
    
    im = axes[0].imshow(spatial.T, extent=[0, 2*np.pi, 0, 2*np.pi], origin='lower', cmap='RdBu', aspect='auto')
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title("Plot A: Spatial Ricker Wavelet (Zero-Mean)")
    axes[0].set_xlabel(r"$\theta$")
    axes[0].set_ylabel(r"$\phi$")
    
    # --- Plot B: Temporal Profile ---
    t = np.linspace(0, 0.2, 500)
    t0 = 0.05
    sigma_t = 0.01
    temporal = np.exp(-(t - t0)**2 / (2 * sigma_t**2))
    
    axes[1].plot(t, temporal, 'k-', lw=2)
    axes[1].fill_between(t, temporal, alpha=0.2, color='gray')
    axes[1].set_title("Plot B: Temporal Gaussian Envelope")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('fig2_ricker_wavelet.png')
    plt.show()

# -----------------------------------------------------------------------------
# 3. Wave Propagation
# -----------------------------------------------------------------------------
def fig3_wave_propagation(P_sequence=None):
    """Generates Figure 3: Snapshots of wave propagation. 
    Accepts P_sequence of shape (T, N_theta, N_phi). Mocks data if None."""
    
    if P_sequence is None:
        print("[Notice] Using mock wave propagation data for Fig 3. Replace with your solver output.")
        # Create a mock expanding ring (anisotropic to simulate torus)
        theta = np.linspace(0, 2*np.pi, N_grid)
        phi = np.linspace(0, 2*np.pi, N_grid)
        THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
        
        P_sequence = []
        for t in [0.1, 0.5, 1.0]:
            # Mock dispersion: expanding radius, varying wave speed
            r_t = t * 3.0
            dist = np.sqrt(((THETA - np.pi)*1.5)**2 + (PHI - np.pi)**2) # Anisotropic
            wave = np.sin(10 * (dist - r_t)) * np.exp(-((dist - r_t)**2)/0.2)
            P_sequence.append(wave)
        P_sequence = np.array(P_sequence)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    times = ["t=10", "t=30", "t=50"] # Adjust to actual your simulation steps
    
    for i in range(3):
        im = axes[i].imshow(P_sequence[i].T, extent=[0, 2*np.pi, 0, 2*np.pi], 
                            origin='lower', cmap='seismic', vmin=-1, vmax=1, aspect='auto')
        axes[i].set_title(f"Snapshot at {times[i]}")
        axes[i].set_xlabel(r"$\theta$")
        if i == 0: axes[i].set_ylabel(r"$\phi$")
        
    plt.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.suptitle("Figure 3: Acoustic Wave Propagation on Torus (Notice Anisotropy)", y=1.05)
    plt.savefig('fig3_wave_propagation.png', bbox_inches='tight')
    plt.show()

# -----------------------------------------------------------------------------
# 4. Learned Latent Grid
# -----------------------------------------------------------------------------
def fig5_learned_latent_grid(deformation_field=None):
    """Generates Figure 5: The Diffeomorphism network mapping.
    deformation_field: array of shape (H, W, 2) containing (du, dv)."""
    
    if deformation_field is None:
        print("[Notice] Using mock deformation field. Pass your `geo_net` output to see actual learned grid.")
        N = 20
        y = np.linspace(-1, 1, N)
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, y)
        
        # Mock deformation simulating flattening the torus metric
        dX = 0.2 * np.sin(np.pi * Y) * np.cos(np.pi * X)
        dY = 0.2 * np.cos(np.pi * X) * np.sin(np.pi * Y)
    else:
        # We want roughly 20 grid lines, so we calculate a stride
        H, W = deformation_field.shape[:2]
        stride = max(1, H // 20)
        
        # Subsample high-res field
        dX = deformation_field[::stride, ::stride, 0]
        dY = deformation_field[::stride, ::stride, 1]
        
        # Recreate the exact base grid coordinates that match the subsampled shape
        y_full = np.linspace(-1, 1, H)
        x_full = np.linspace(-1, 1, W)
        
        y = y_full[::stride]
        x = x_full[::stride]
        X, Y = np.meshgrid(x, y)

    X_def = X + dX
    Y_def = Y + dY

    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Plot original base grid (faint)
    for i in range(X.shape[0]):
        ax.plot(X[i, :], Y[i, :], color='gray', alpha=0.3, linestyle='--')
    for j in range(X.shape[1]):
        ax.plot(X[:, j], Y[:, j], color='gray', alpha=0.3, linestyle='--')
        
    # Plot deformed latent grid
    for i in range(X_def.shape[0]):
        ax.plot(X_def[i, :], Y_def[i, :], color='blue', alpha=0.7, lw=1.5)
    for j in range(X_def.shape[1]):
        ax.plot(X_def[:, j], Y_def[:, j], color='blue', alpha=0.7, lw=1.5)

    ax.set_title("Figure 5: Learned Latent Grid via DiffeomorphismNet", pad=20)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_aspect('equal')
    ax.grid(False)
    plt.savefig('fig5_latent_grid.png')
    plt.show()

# -----------------------------------------------------------------------------
# 5. Quantitative Results
# -----------------------------------------------------------------------------
def fig6_loss_curves(history_unet=None, history_fno=None, history_geofno=None):
    """Generates Figure 6: Comparative MSE Loss Curves."""
    epochs = np.arange(1, 51)
    
    if history_geofno is None:
        print("[Notice] Using mock loss history. Replace with your `trainer.history['val_loss']`.")
        # Mock learning curves
        loss_unet_val = 0.1 * np.exp(-epochs/10) + 0.05
        loss_fno_train = 0.5 * np.exp(-epochs/5) + 0.0001 # FNO memorizes
        loss_fno_val = 0.2 * np.exp(-epochs/8) + 0.08     # FNO validation stays high
        loss_geofno_val = 0.4 * np.exp(-epochs/7) + 0.001 # GeoFNO generalizes
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, loss_unet_val, 'r--', label='Periodic U-Net (Val)')
    ax.plot(epochs, loss_fno_train, 'g:', label='Vanilla FNO (Train)')
    ax.plot(epochs, loss_fno_val, 'g-', label='Vanilla FNO (Val)')
    ax.plot(epochs, loss_geofno_val, 'b-', label='Geo-FNO (Val)', lw=2.5)
    
    ax.set_yscale('log')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title("Figure 6: Operator Learning Convergence (Train vs Validation)")
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc='upper right')
    
    plt.savefig('fig6_loss_curves.png')
    plt.show()

def fig7_spatial_error_heatmaps(P_gt, P_unet, P_fno, P_geofno):
    """Generates Figure 7: Spatial Absolute Error comparison across models.
    Inputs are 2D numpy arrays of shape (N_theta, N_phi)."""
    
    models = ["Periodic U-Net", "Vanilla FNO", "Geo-FNO"]
    preds = [P_unet, P_fno, P_geofno]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i, (name, pred) in enumerate(zip(models, preds)):
        err = np.abs(P_gt - pred)
        
        # Ground Truth
        im1 = axes[i, 0].imshow(P_gt.T, cmap='seismic', origin='lower', vmin=-1, vmax=1, aspect='auto')
        axes[i, 0].set_title(f"Ground Truth" if i==0 else "")
        axes[i, 0].set_ylabel(name, fontsize=16, fontweight='bold', labelpad=20)
        
        # Prediction
        im2 = axes[i, 1].imshow(pred.T, cmap='seismic', origin='lower', vmin=-1, vmax=1, aspect='auto')
        axes[i, 1].set_title(f"Prediction" if i==0 else "")
        
        # Absolute Error
        im3 = axes[i, 2].imshow(err.T, cmap='inferno', origin='lower', vmin=0, vmax=0.5, aspect='auto')
        axes[i, 2].set_title(f"Absolute Error $|P_{{pred}} - P_{{gt}}|$" if i==0 else "")
        
        # Formatting
        for j in range(3):
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])

    # Colorbars
    fig.colorbar(im2, ax=axes[:, 1], orientation='vertical', fraction=0.05, pad=0.04, label="Pressure")
    fig.colorbar(im3, ax=axes[:, 2], orientation='vertical', fraction=0.05, pad=0.04, label="Absolute Error")
    
    plt.suptitle("Figure 7: Spatial Generalization on Unseen Toroidal Initial Conditions", y=0.95)
    plt.savefig('fig7_error_heatmaps.png', bbox_inches='tight')
    plt.show()

# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    print("Generating Figure 1: Geometry & Metric...")
    fig1_geometry_and_metric()
    
    print("Generating Figure 2: Ricker Source...")
    fig2_ricker_wavelet()
    
    print("Generating Figure 3: Wave Propagation...")
    fig3_wave_propagation()
    
    print("Generating Figure 5: Learned Latent Grid...")
    fig5_learned_latent_grid()
    
    print("Generating Figure 6: Loss Curves...")
    fig6_loss_curves()
    
    print("Generating Figure 7: Error Heatmaps...")
    # Generate mock 2D fields for testing the layout
    mock_gt = np.sin(np.linspace(0, 10, 128)) * np.cos(np.linspace(0, 10, 128))[:, None]
    mock_unet = mock_gt + np.random.normal(0, 0.3, (128, 128)) # High uniform error
    mock_fno = mock_gt * 0.5 + 0.2                             # Misses amplitude/phase (memo fails)
    mock_geofno = mock_gt + np.random.normal(0, 0.05, (128, 128)) # Low error
    fig7_spatial_error_heatmaps(mock_gt, mock_unet, mock_fno, mock_geofno)
    
    print("All figures successfully generated and saved to the current directory.")
