# Acoustic Wave Equation on a Torus: WebGL Real-Time Simulation Architecture

## 1. Introduction and Physics Model

The physics of this application models the propagation of acoustic pressure waves on the curved 2-dimensional Riemannian manifold of a Torus. Unlike a flat Cartesian grid where waves propagate uniformly in all directions, a Torus has non-uniform curvature. The "inside" ring of the Torus has negative Gaussian curvature, while the "outside" ring has positive Gaussian curvature. This causes sound waves to stretch, squeeze, and diffract dynamically based on their coordinate position.

### The Governing Partial Differential Equation (PDE)
The propagation of an acoustic wave $u(\theta, \phi, t)$ on an arbitrary curved manifold is governed by the Laplace-Beltrami wave equation:

$$ \frac{\partial^2 u}{\partial t^2} = c^2 \Delta_g u + S(\theta, \phi, t) $$

Where:
- $c$ is the wave speed.
- $S(\theta, \phi, t)$ is the external source injection (the interaction taps).
- $\Delta_g$ is the Laplace-Beltrami operator, which generalizes the standard Cartesian Laplacian $\nabla^2$ to curved surfaces.

For a Torus defined by major radius $R$ and minor radius $r$, parameterized by poloidal angle $\theta$ and toroidal angle $\phi$, the metric tensor determinant is $g = r^2(R + r \cos\theta)^2$.
The Laplace-Beltrami operator thus expands analytically to:
$$ \Delta_g u = \frac{1}{r(R + r \cos\theta)} \left[ \frac{\partial}{\partial \theta} \left( \frac{R + r \cos\theta}{r} \frac{\partial u}{\partial \theta} \right) + \frac{\partial}{\partial \phi} \left( \frac{r}{R + r \cos\theta} \frac{\partial u}{\partial \phi} \right) \right] $$

---

## 2. Theoretical Considerations & Challenges

### The Conservation of Mass Theorem
By strictly defining the PDE as perfectly lossless (i.e., lacking a linear damping coefficient $-k \frac{\partial u}{\partial t}$ or a spring coefficient $-k u$), the equation perfectly conserves physical momentum.

However, this introduces a severe numerical challenge: If an interaction source $S(\theta, \phi)$ injects a strictly positive pressure wave (e.g., a standard Gaussian pulse) into a bounded, lossless manifold, the global integral of the pressure increases continuously. Over time, this raises the mathematical baseline voltage of the entire Torus straight to infinity.

### The 2D Mexican Hat (Ricker Wavelet) Solution
To resolve baseline shift without artificially dampening the physics, the injected source must have an exact spatial mean of zero. While a 1D Mexican Hat wavelet drops off according to $(1 - x^2/\sigma^2)$, the proper exact 2D projection (the Laplacian of a Gaussian function) utilizes the following geometry:
$$ S(r) = \left(2 - \frac{r^2}{\sigma^2}\right) \exp\left(-\frac{r^2}{2\sigma^2}\right) $$
By explicitly enforcing the discrete mathematical grid to sum this shape to exactly `0.000000`, we inject massive local energy that physically models a perfectly pure disturbance that conserves system mass forever. 

---

## 3. Time Dynamics & Discretization

To solve the continuous PDE on a discrete computer, we utilize the **Explicit Finite-Difference Time-Domain (FDTD)** method. 

### The Staggered Grid Leapfrog Method
We discretize time into discrete frames $t_n = n \Delta t$. The second derivative of time can be approximated using the central leapfrog operator:
$$ \frac{\partial^2 u}{\partial t^2} \approx \frac{u^{n+1} - 2u^n + u^{n-1}}{\Delta t^2} $$

By cleanly isolating the future state $u^{n+1}$, we get the fundamental simulation rule computed every frame:
$$ u^{n+1} = 2u^n - u^{n-1} + \Delta t^2 \left( c^2 \Delta_g u^n + S \right) $$

**CFL Stability Criteria**
To prevent the finite difference algorithm from violating the speed of light and instantly blowing up to mathematical `NaN`s, the time step $\Delta t$ and spatial subdivision $\Delta x$ must adhere to the Courant-Friedrichs-Lewy (CFL) condition:
$$ c \frac{\Delta t}{\Delta x} \le \frac{1}{\sqrt{2}} $$
The simulation uses $dt = 0.01$ and $c = 1.0$ on a $128 \times 128$ angular grid.

---

## 4. Practical Engineering: The FBO Ping-Pong WebGL Engine

Performing $128 \times 128$ physical calculus operations $\sim 60$ times per second in JavaScript falls victim to CPU bottlenecking. To achieve absolute real-time performance, the entire differential algorithm is outsourced directly to the system Graphics Card (GPU) using WebGL.

### WebGL Framebuffer Objects (FBO)
Instead of executing math on arrays, the state of the Torus is painted mathematically onto off-screen HTML5 Canvases called Framebuffers.
1. `texPrev`: Holds the values representing $u^{n-1}$.
2. `texCurr`: Holds the values representing $u^n$.
3. `texNext`: The render target for $u^{n+1}$.

### The Ping-Pong Loop
Every frame in `useFrame`:
1. The discrete geometry array is generated in JavaScript (Raycast mouse hits form the Mexican Hat matrix) and asynchronously uploaded to `texSource`.
2. A custom mathematical Graphic Shader (`waveFragmentShader`) binds `texPrev`, `texCurr`, and `texSource`.
3. The GPU processes all `16,384` grid points entirely in parallel utilizing the specialized FDTD physics algorithm, dropping the answer into `texNext`.
4. The JavaScript pointers are computationally "swapped" (`prev = curr`, `curr = next`, `next = prev`) to avoid performing expensive memory allocations.

---

## 5. Final Output Format & Objects

### 1. `WaveEngine.tsx` (The Brain)
- **Input:** 3D cursor Raycast hits transformed into Toroidal topological coordinates ($0 \to 1$ bounding UV space), tracking `pointerUV` and `isPointerDown`.
- **Compute:** Employs the `OrthographicCamera` rendering pipeline strictly as a GPGPU math node.
- **Output:** Returns a continuous stream of live `THREE.Texture` objects representing the pure physics state back to the parent React context.

### 2. `TorusVisualizer.tsx` (The Eyes)
- **Input:** A raw physics `THREE.Texture` passed by event hook.
- **Compute:** Employs a custom procedural Fragment Shader material wrapped over the Three.js `<torusGeometry>`. The code samples the spatial normals of the torus and interpolates physical Lambertian Diffuse, Phong Specular lighting, and cinematic Rim lights. 
- **Output:** The visual manifestation dynamically mapping the physical pressure scale: High pressure peaks are shaded Rose Gold, crushing low pressure troughs emit Bioluminescent Mint, mapped over an Obsidian baseline void.
