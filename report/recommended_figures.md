# Recommended Figures & Results for the NOMAD Report

Based on the mathematical framework and the PyTorch implementation in `src_code.txt`, you should generate the following visualizations to provide empirical evidence for the theoretical claims in `nomad.md`.

## 1. Geometric Foundations
To ground the mathematics in physical intuition, visualize the manifold and its metric properties.

* **Figure 1: Toroidal Coordinate System and Metric Heterogeneity**
  * **Plot A:** A 3D wireframe or surface plot of the Torus ($\mathbb{T}^2$) clearly labeling the poloidal angle $\theta$ and toroidal angle $\phi$. Highlight the inner equator ($\theta=\pi$) and outer equator ($\theta=0$).
  * **Plot B:** A 2D heatmap of the metric determinant $\sqrt{|g|} = r(R + r\cos\theta)$ unrolled onto a flat $[\theta, \phi]$ grid. This visually demonstrates the spatial variance of the area element, proving why translation-invariant convolutions will fail.

## 2. High-Fidelity Spectral Simulation (Ground Truth)
Showcase the quality of the Fourier pseudospectral solver dataset.

* **Figure 2: Ricker Wavelet Source Injection**
  * **Plot A:** The 2D spatial profile (Mexican Hat) of the source on the unwrapped $[\theta, \phi]$ grid, showing the zero-mean property (positive center, negative ring).
  * **Plot B:** A 1D time-series plot of the temporal Gaussian envelope showing the excitation impulse.
* **Figure 3: Acoustic Wave Propagation on the Torus**
  * A sequence of snapshots ($t=t_1, t_2, t_3$) showing the pressure field $P(\theta, \phi)$ evolving over time.
  * *Crucial observation to highlight:* Show how the wavefront becomes anisotropic (non-circular) as it expands, due to the varying wave speed in the $(\theta, \phi)$ coordinate frame caused by the metric $g_{ij}$.

## 3. Architecture & Neural Operator Mechanisms
Help the reader understand the structural differences between the models.

* **Figure 4: Geo-FNO Pipeline Diagram**
  * A block diagram tracing the flow of data:
    1. Physical inputs ($P_{\text{in}}, S, g$) mapped to a deformed grid via the `DiffeomorphismNet`.
    2. The Pullback operation (`F.grid_sample`).
    3. The `BaseFNO2d` operating in the flat latent space.
    4. The Pushforward operation mapping back to the physical torus.

* **Figure 5: Learned Latent Grid (The Diffeomorphism)**
  * A 2D plot of the `latent_grid` outputted by the `DiffeomorphismNet` (i.e., `base_grid + deformation`).
  * Plot the original uniform Cartesian grid lines, and overlay the deformed grid lines. This is the "smoking gun" figure that proves the network has successfully learned a coordinate transformation that neutralizes the toroidal metric.

## 4. Quantitative Results & Comparative Analysis
Provide empirical evidence that Geo-FNO solves the generalization problem.

* **Figure 6: Training vs. Validation Loss Curves**
  * A standard line chart plotting the MSE loss over epochs for **Periodic U-Net**, **Vanilla FNO**, and **Geo-FNO**.
  * Expected outcome: Vanilla FNO will show training loss dropping to near zero (memorization) but validation loss plateauing. Geo-FNO should show both dropping concurrently.
* **Figure 7: Spatial Error Heatmaps (Rollout Evaluation)**
  * Take an unseen validation rollout at a late time-step (e.g., $t=t_{\text{out}}$).
  * Create a $3 \times 3$ grid of plots:
    * **Rows:** Periodic U-Net, Vanilla FNO, Geo-FNO.
    * **Columns:** Ground Truth Field, Predicted Field, Absolute Error Field ($|P_{\text{pred}} - P_{\text{true}}|$).
  * This visually proves that Vanilla FNO produces massive structural errors on unseen data, while Geo-FNO accurately captures the wave dispersion.

## 5. Performance / Implementation Details
* **Table 1: Computational Performance Matrix**
  * A table comparing the three architectures in terms of:
    * Parameter Count (Geo-FNO has slightly more due to `DiffeomorphismNet`).
    * Inference Time per step (in milliseconds, leveraging FP16/AMP).
    * Final Validation MSE.
  * This proves that the massive accuracy gain of Geo-FNO comes with a negligible computational overhead.
