# Acoustic Wave Torus Geometric Deep Learning Framework: Scientific Report

## 1. Introduction & Physical Formulation
This project implements a Geometric Deep Learning framework for acoustic wave propagation on a 2D Torus manifold. Simulating waves on complex geometries presents unique challenges for standard Convolutional Neural Networks (CNNs) due to periodic boundary conditions and non-Euclidean metric tensors. We formulate the problem using the Laplace-Beltrami operator to construct a physics-informed high-fidelity numerical simulator (using explicit RK4 and Spectral methods). These ground-truth datasets are then used to train autoregressive deep learning models.

## 2. Experimental Progression & Model Architectures

In our attempt to capture the wave dynamics accurately, we progressed through several data-driven discovery architectures.

### 2.1 Periodic UNet
The initial approach utilized a Periodic UNet. While UNets are powerful for standard grid-based image-to-image translation, their standard zero-padding boundary conditions severely disrupt the toroidal topology. By enforcing circular padding, the UNet learned localized wave propagation but struggled to capture long-range interactions effectively across the Torus.

### 2.2 FNO2D (Fourier Neural Operator)
To resolve global interactions, we applied the base 2D Fourier Neural Operator (FNO2D). FNOs perform convolutions in the spectral domain, making them naturally suited for periodic signals.
* **Challenges:** Initial training iterations revealed suboptimal results and significant overfitting on single-pulse datasets.
* **Physics-Informed Improvements:** We attempted to regularize the FNO by incorporating a scaled spectral loss and an energy-conservation physics-informed loss directly into the training objective. While this improved theoretical stability, the results still lacked sharp spatial resolution during rollout.

### 2.3 Fast Geo-FNO
Finally, we introduced Geo-FNO, an architecture specifically designed for complex topologies by utilizing a learned or prescribed diffeomorphism mapping the physical Torus grid to a latent uniform computational space.
* **Results:** Geo-FNO yielded surprisingly superior results compared to both the Periodic UNet and the baseline FNO2D. It maintained numerical stability over long-term rollouts without accumulating catastrophic phase errors.

## 3. High Grid Evaluation and Rigorous Analysis (Geo-FNO)
Given the performance of Geo-FNO, the subsequent detailed analysis focuses on this architecture. We evaluate the model's performance on high-resolution multi-pulse datasets, measuring the strict $L^2$ relative errors over time.

*(Note: The embedded figures from the iterative training loops and final Geo-FNO evaluations are referenced below. As further physical refinements are added to Geo-FNO, these evaluations will be updated to include strict energy conservation and manifold Laplacian validations.)*

## 4. Figures from Notebook Executions

The following figures correspond to the various evaluations (time-domain traces, error convergence, and manifold rollouts) extracted from the core Jupyter Notebook runs.

````carousel
![Notebook Output 1](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_001_cell_40.png)
<!-- slide -->
![Notebook Output 2](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_002_cell_41.png)
<!-- slide -->
![Notebook Output 3](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_003_cell_42.png)
<!-- slide -->
![Notebook Output 4](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_004_cell_43.png)
<!-- slide -->
![Notebook Output 5](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_005_cell_45.png)
<!-- slide -->
![Notebook Output 6](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_006_cell_47.png)
<!-- slide -->
![Notebook Output 7](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_007_cell_48.png)
<!-- slide -->
![Notebook Output 8](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_008_cell_49.png)
<!-- slide -->
![Notebook Output 9](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_009_cell_51.png)
<!-- slide -->
![Notebook Output 10](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_010_cell_53.png)
````

````carousel
![Notebook Output 11](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_011_cell_55.png)
<!-- slide -->
![Notebook Output 12](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_012_cell_57.png)
<!-- slide -->
![Notebook Output 13](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_013_cell_60.png)
<!-- slide -->
![Notebook Output 14](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_014_cell_61.png)
<!-- slide -->
![Notebook Output 15](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_015_cell_63.png)
<!-- slide -->
![Notebook Output 16](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_016_cell_64.png)
<!-- slide -->
![Notebook Output 17](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_017_cell_65.png)
<!-- slide -->
![Notebook Output 18](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_018_cell_66.png)
<!-- slide -->
![Notebook Output 19](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_019_cell_68.png)
<!-- slide -->
![Notebook Output 20](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_020_cell_69.png)
````

````carousel
![Notebook Output 21](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_021_cell_70.png)
<!-- slide -->
![Notebook Output 22](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_022_cell_72.png)
<!-- slide -->
![Notebook Output 23](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_023_cell_74.png)
<!-- slide -->
![Notebook Output 24](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_024_cell_75.png)
<!-- slide -->
![Notebook Output 25](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_025_cell_77.png)
<!-- slide -->
![Notebook Output 26](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_026_cell_78.png)
<!-- slide -->
![Notebook Output 27](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_027_cell_80.png)
<!-- slide -->
![Notebook Output 28](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_028_cell_81.png)
<!-- slide -->
![Notebook Output 29](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_029_cell_82.png)
<!-- slide -->
![Notebook Output 30](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_030_cell_84.png)
<!-- slide -->
![Notebook Output 31](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_031_cell_86.png)
<!-- slide -->
![Notebook Output 32](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_032_cell_87.png)
<!-- slide -->
![Notebook Output 33](file:///Users/raven/Projects/NOMAD/notebook/report_figures/fig_033_cell_88.png)
````

## 5. Next Steps
- Further refine the **Geo-FNO** formulation with physics information (Energy Conservation Loss and strict Spectral Damping mappings).
- Complete high-grid (128x128 and 256x256) multi-pulse evaluation metrics.
- Regenerate unified clean notebook to facilitate this rigorous final analysis.
