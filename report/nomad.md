# Operator Learning for Acoustic Wave Propagation on Toroidal Manifolds: A Formal Mathematical Approach

**Date**: June 2026

---

## Abstract

This paper presents a mathematically rigorous framework for simulating and predicting acoustic wave propagation on a non-Euclidean toroidal manifold ($\mathbb{T}^2$). We bridge the continuous differential geometry of curved surfaces with discrete pseudospectral collocation methods, ensuring highly accurate, energy-stable data generation. Furthermore, we deconstruct the theoretical limitations of translation-invariant neural architectures—such as Periodic U-Nets and vanilla Fourier Neural Operators (FNOs)—when applied to heterogeneous metric spaces. To resolve these mathematical failures, we formally define the Geometry-Aware Fourier Neural Operator (Geo-FNO). We demonstrate how learned diffeomorphic mappings can dynamically deform the intrinsic Riemannian metric, wrapping the curved space into a translationally-invariant latent domain to achieve robust, global operator generalization over highly chaotic initial conditions.

---

## 1. Differential Geometry of the Torus

To solve physical equations on a curved surface, we must first rigorously define the intrinsic geometry using Riemannian metrics.

### 1.1 Parameterization and Tangent Space

We embed a 2-torus into $\mathbb{R}^3$. Let $R$ be the major radius and $r$ the minor radius. The surface is parameterized by two angles: the poloidal angle $\theta \in [0, 2\pi)$ and the toroidal angle $\phi \in [0, 2\pi)$. The coordinate transformation $\mathbf{x}: \mathbb{T}^2 \to \mathbb{R}^3$ is defined as:

$$
\mathbf{x}(\theta, \phi) = 
\begin{bmatrix}
(R + r\cos\theta)\cos\phi \\
(R + r\cos\theta)\sin\phi \\
r\sin\theta
\end{bmatrix}
$$

The basis vectors of the local tangent space $\mathcal{T}_p\mathbb{T}^2$ are the partial derivatives of the parameterization with respect to the coordinates:

$$
\mathbf{e}_\theta = \frac{\partial \mathbf{x}}{\partial \theta} = 
\begin{bmatrix}
-r\sin\theta\cos\phi \\
-r\sin\theta\sin\phi \\
r\cos\theta
\end{bmatrix}, \quad
\mathbf{e}_\phi = \frac{\partial \mathbf{x}}{\partial \phi} = 
\begin{bmatrix}
-(R + r\cos\theta)\sin\phi \\
(R + r\cos\theta)\cos\phi \\
0
\end{bmatrix}
$$

### 1.2 The Metric Tensor

The first fundamental form determines the distance metric $ds^2$ on the manifold, captured by the metric tensor $g_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j$:

$$ g_{\theta\theta} = \mathbf{e}_\theta \cdot \mathbf{e}_\theta = r^2 $$
$$ g_{\phi\phi} = \mathbf{e}_\phi \cdot \mathbf{e}_\phi = (R + r\cos\theta)^2 $$
$$ g_{\theta\phi} = g_{\phi\theta} = \mathbf{e}_\theta \cdot \mathbf{e}_\phi = 0 $$

Because the off-diagonal elements are zero, the basis vectors are strictly orthogonal. The metric tensor $g$ and its inverse $g^{-1}$ (denoted in index notation as $g^{ij}$) are:

$$
g = \begin{pmatrix} r^2 & 0 \\ 0 & (R + r\cos\theta)^2 \end{pmatrix}, \quad 
g^{-1} = \begin{pmatrix} \frac{1}{r^2} & 0 \\ 0 & \frac{1}{(R + r\cos\theta)^2} \end{pmatrix}
$$

The determinant of the metric, $|g| = \det(g)$, dictates the local area volume element on the manifold:

$$ \sqrt{|g|} = r(R + r\cos\theta) $$

![Torus Geometry and Metric](/Users/raven/Projects/NOMAD/report/fig1_torus_geometry.png)
*Figure 1: (A) The 3D surface of the toroidal manifold. (B) The heterogeneous metric determinant $\sqrt{|g|}$ mapped across the $(\theta, \phi)$ plane. Notice the significant variation in the area element; the physical distance per radian is much smaller at the inner equator ($\theta = \pi$) than the outer equator ($\theta = 0, 2\pi$).*

---

## 2. Acoustic Wave Equation & Discretization

### 2.1 The Laplace–Beltrami Operator

The acoustic wave equation on a manifold $\mathcal{M}$ governs the evolution of the scalar pressure field $P$:

$$ \frac{\partial^2 P}{\partial t^2} = c^2 \Delta_{\mathcal{M}} P + S(t, \theta, \phi) $$

Where $c$ is the wave speed, and $\Delta_{\mathcal{M}}$ is the Laplace–Beltrami operator. For a scalar function $f \in C^\infty(\mathcal{M})$, the operator generalizes the standard Laplacian by incorporating the metric volume element:

$$ \Delta_{\mathcal{M}} f = \frac{1}{\sqrt{|g|}} \partial_i \left( \sqrt{|g|} \, g^{ij} \partial_j f \right) $$

Expanding this for our orthogonal toroidal metric yields the precise PDE we must discretize:

$$ \Delta_{\mathcal{M}} P = \frac{1}{r^2}\frac{\partial^2 P}{\partial\theta^2} - \frac{\sin\theta}{r(R+r\cos\theta)}\frac{\partial P}{\partial\theta} + \frac{1}{(R+r\cos\theta)^2}\frac{\partial^2 P}{\partial\phi^2} $$

### 2.2 Fourier Pseudospectral Collocation

To integrate the PDE with high precision, we evaluate the spatial derivatives using Fourier analysis, which achieves exponential accuracy (spectral convergence) for smooth periodic domains. Discretizing $P(\theta, \phi)$ onto an $N_\theta \times N_\phi$ grid, the 2D Discrete Fourier Transform (DFT) is:

$$ \hat{P}_{k_\theta, k_\phi} = \mathcal{F}[P] = \sum_{m=0}^{N_\theta-1} \sum_{n=0}^{N_\phi-1} P_{m,n} e^{-2\pi i (m k_\theta / N_\theta + n k_\phi / N_\phi)} $$

By the derivative theorem, a spatial derivative is a point-wise multiplication by the wavenumber $ik$:

$$ \mathcal{F}\left[ \frac{\partial^\alpha P}{\partial \theta^\alpha} \right] = (i k_\theta)^\alpha \hat{P}_{k_\theta, k_\phi} $$

The physical derivatives are recovered globally via the Inverse Fast Fourier Transform (IFFT) and mapped point-wise back onto the spatial grid (collocation):

$$ \frac{\partial P}{\partial \theta} = \text{Re}(\mathcal{F}^{-1}[ i k_\theta \hat{P} ]), \quad \frac{\partial^2 P}{\partial \theta^2} = \text{Re}(\mathcal{F}^{-1}[ -k_\theta^2 \hat{P} ]) $$

### 2.3 Leapfrog Time Integration and CFL Stability

For the temporal evolution, we use an explicit second-order central difference (Leapfrog) scheme. Approximating the second-order time derivative using a Taylor expansion yields the discrete time-step update:

$$ P^{(n+1)} = 2P^{(n)} - P^{(n-1)} + \Delta t^2 \left(c^2 \Delta_{\mathcal{M}} P^{(n)} + S^{(n)}\right) $$

**CFL Condition:** Stability dictates that the timestep $\Delta t$ must resolve the highest frequencies. Because the physical grid spacing $ds$ is inhomogeneous, the strict constraint occurs where grid points are physically closest—the inner equator ($\theta = \pi$):

$$ \Delta t \le \text{CFL} \cdot \frac{\min\left( r \Delta\theta, (R-r)\Delta\phi \right)}{c} $$

---

## 3. Excitation Source and Complex Chaos

To prevent unbounded energy accumulation on the closed manifold surface (which lacks absorbing boundaries), the acoustic source $S$ must integrate to strictly zero ($\int_{\mathbb{T}^2} S \, dA = 0$).

We utilize a spatial Mexican Hat (Ricker) wavelet, formalized as the negative normalized second derivative of a Gaussian:

$$ S_{\text{space}}(\rho) = \left( 1 - \frac{\rho^2}{\sigma_s^2} \right) \exp\left(-\frac{\rho^2}{2\sigma_s^2}\right) $$

Where $\rho^2 \approx (r \Delta\theta)^2 + ((R + r\cos\theta_0) \Delta\phi)^2$ approximates the localized manifold distance. 

![Ricker Wavelet](/Users/raven/Projects/NOMAD/report/fig2_ricker_wavelet.png)
*Figure 2: The Ricker Wavelet in space and time. Its zero-mean spatial property stabilizes the closed-manifold simulation.*

![Wave Propagation](/Users/raven/Projects/NOMAD/report/fig3_wave_propagation.png)
*Figure 3: Simulated wave propagation. The dispersion is highly anisotropic due to the changing metric tensor, traveling faster along the compressed inner equator geometry.*

### 3.1 Complex Chaos Dataset

To ensure the neural operators learn the true physical PDE operator mapping $\mathcal{G}: \mathcal{A} \to \mathcal{U}$ rather than overfitting to a single stationary point source, we generate initial conditions characterized by randomized superpositions of continuous spatial frequencies.

![Complex Chaos Dataset](/Users/raven/Projects/NOMAD/report/plot_complex_chaos.png)
*Figure 4: A representative sample from the Complex Chaos dataset, illustrating highly non-linear, multi-modal initial wave states.*

---

## 4. Operator Learning Formalism

We aim to learn an operator mapping the input sequence $P(t \in [0, T_{in}])$ to the future state $P(t \in (T_{in}, T_{out}])$.

### 4.1 Periodic U-Net

A fully convolutional UNet operates on local receptive fields. On a torus, standard zero-padding destroys the topological boundaries, necessitating strictly circular modulo padding:

$$ (K * P)_{i,j} = \sum_{u} \sum_{v} K_{u,v} P_{(i-u) \bmod N_\theta, (j-v) \bmod N_\phi} $$

**Failure Mode:** While topologically correct, the CNN kernels $K$ are strictly translation-invariant. The local physical wave-speed $v(\theta) = c\sqrt{g^{ij}}$ is inhomogeneous. A standard U-Net fundamentally cannot apply different functional mapping logic to different geographic areas of the manifold.

### 4.2 Vanilla Fourier Neural Operator (FNO)

The FNO learns a continuous integral operator update $v^{(l+1)}(\mathbf{x}) = \sigma\left( W v^{(l)}(\mathbf{x}) + \int_{\mathcal{D}} \kappa(\mathbf{x}, \mathbf{y}) v^{(l)}(\mathbf{y}) d\mathbf{y} \right)$. By assuming a translation-invariant kernel $\kappa(\mathbf{x}, \mathbf{y}) = \kappa(\mathbf{x} - \mathbf{y})$, the integral simplifies to a convolution in Fourier space:

$$ \int_{\mathcal{D}} \kappa(\mathbf{x}-\mathbf{y}) v^{(l)}(\mathbf{y}) d\mathbf{y} = \mathcal{F}^{-1} \left[ R_\phi \cdot \mathcal{F}[v^{(l)}] \right](\mathbf{x}) $$

Where $R_\phi$ is a learned complex weight tensor truncating the Fourier series to a finite mode limit. 

**Failure Mode:** Because the true physical propagation kernel on a Torus is structurally dependent on the heterogeneous distance metric $\sqrt{|g|}$, the assumption $\kappa(\mathbf{x}, \mathbf{y}) = \kappa(\mathbf{x} - \mathbf{y})$ is violently violated. The FNO memorizes Euclidean topologies but generalizes extremely poorly to metric deformations.

### 4.3 Tensor Contraction and ONNX Compatibility

To support ONNX optimization without native complex tensors (`torch.cfloat`), the Fourier multiplications must be mapped to real-valued index contractions. Let the feature $\hat{v} = x_r + i x_i$ and weights $R_\phi = w_r + i w_i$. The implementation utilizes Einstein summation (`torch.einsum`) to compute the resulting tensor cross-terms explicitly:

$$ \mathcal{Y}_{r} = X_{r} W_{r} - X_{i} W_{i}, \quad \mathcal{Y}_{i} = X_{i} W_{r} + X_{r} W_{i} $$

---

## 5. Geometry-Aware Fourier Neural Operator (Geo-FNO)

To resolve the theoretical limits of translation-invariant architectures on curved manifolds, the Geo-FNO maps the physical domain $\Omega_{\text{phys}} = \mathbb{T}^2$ to a flat latent domain $\Omega_{\text{latent}} = [-1, 1]^2$ using a learned diffeomorphism $\varphi: \Omega_{\text{phys}} \to \Omega_{\text{latent}}$.

![Diffeomorphism Network](/Users/raven/Projects/NOMAD/report/diffeomorphic_mapping_net.png)
*Figure 5: The architectural pipeline. A lightweight fully convolutional DiffeomorphismNet learns a deformation field $\delta(\mathbf{x})$ directly from the normalized metric tensor, determining the optimal latent mapping.*

The Geo-FNO mechanism proceeds in four stages:

1. **Latent Mapping:** $\varphi(\mathbf{x}) = \mathbf{x}_{\text{uniform}} + \delta(\mathbf{x})$.
2. **Pullback Operator ($\varphi^*$):** The physical pressure tensor $P_{\text{phys}}$ is pulled back into the flat latent space via differentiable bilinear interpolation (`F.grid_sample`).
   $$ P_{\text{latent}}(\mathbf{y}) = (\varphi^* P_{\text{phys}})(\mathbf{y}) = P_{\text{phys}}(\varphi^{-1}(\mathbf{y})) $$
3. **Latent FNO:** The FNO operates on $P_{\text{latent}}$. Because $\varphi$ has absorbed the metric heterogeneity—such that the pullback metric $(\varphi^* g)_{ij} \approx \delta_{ij}$—the effective kernel $\tilde{\kappa}$ is now approximately translation-invariant, perfectly aligning with the foundational assumptions of the FNO.
4. **Pushforward:** The predicted future state is pushed back to the physical torus.

![Latent Grid](/Users/raven/Projects/NOMAD/report/fig5_latent_grid.png)
*Figure 6: The learned latent coordinate grid $\varphi(\mathbf{x})$. The network learns to "unwrap" and distort the parameterization space specifically to flatten the variations caused by the toroidal metric determinant.*

---

## 6. Quantitative Results

The integration of the diffeomorphic mapping yields radical improvements in generalization capabilities over structurally complex wave propagation states.

![Loss Convergence](/Users/raven/Projects/NOMAD/report/geofno128x128_loss.png)
*Figure 7: Training and validation loss curves on the $128 \times 128$ resolution grid. Notice how the Periodic U-Net plateau's early, and the Vanilla FNO severely overfits (huge validation gap). The Geo-FNO remains stable, successfully modeling the unseen data.*

![Spatial Generalization Error](/Users/raven/Projects/NOMAD/report/geofno128x128.png)
*Figure 8: Spatial absolute error across unseen test initial conditions. The Geo-FNO predicts the future acoustic state almost perfectly, while the translation-invariant architectures accumulate massive error at the inner equator (where metric deformation is maximum).*

---

## Conclusion

By unifying explicit differential geometry with discrete spectral collocation methods, we have established a rigorously stable pipeline for generating acoustic propagation data on toroidal manifolds. Furthermore, our mathematical analysis demonstrates exactly why naive CNN architectures and rigid global spectral models fail on non-Euclidean spaces. The Geometry-Aware FNO mathematically bridges this gap, employing data-driven diffeomorphisms to pull back curved state spaces into translation-invariant domains. This formalizes a highly accurate, robust, and computationally scalable framework for neural operator learning on complex geometric manifolds.