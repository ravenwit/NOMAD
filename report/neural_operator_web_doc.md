# PeriodicUNet (Toroidal Operator Net) Web Implementation Documentation

This document provides a comprehensive overview of the methodology and technical details regarding the implementation of the `PeriodicUNet` (`toroidal_operator_net.pt`) deep learning model within the web-based Acoustic Wave Torus simulation.

## 1. Methodology

The core philosophy of this implementation is to bypass traditional, computationally heavy numerical PDE solvers (such as finite-difference or spectral methods) by replacing them with a fast, autoregressive neural operator running entirely on the client side. 

### 1.1 The Neural Operator Architecture
The model is a **PeriodicUNet**, trained to learn the transition dynamics of the wave equation mapped to a toroidal manifold. 
Instead of operating in standard Euclidean space, the network receives geometric priors to understand the curvature of the torus.

- **Inputs (`1, 4, 64, 64` Tensor):**
  - **`P_curr`**: Current pressure state (normalized).
  - **`P_prev`**: Previous pressure state (normalized) — provides momentum/velocity context necessary for second-order PDEs.
  - **`S_curr`**: The external source/impulse tensor (normalized), allowing user interaction.
  - **`M_static`**: A static metric embedding tensor representing the Riemannian metric $g = \det(g)$ of the Torus across the grid, allowing the convolution kernels to implicitly adapt to the curvature.

- **Output (`1, 1, 64, 64` Tensor):**
  - **`P_next`**: The predicted pressure state at the next timestep (normalized).

### 1.2 Model Export and Conversion (`export_onnx.py`)
To run inside a browser, the PyTorch model (`toroidal_operator_net.pt`) is exported to the Open Neural Network Exchange (ONNX) format. 
- The export script uses ONNX **Opset 17** (with fallbacks to 16, 14, or 11). Opset 16+ is specifically preferred because it supports explicit `Pad(mode='wrap')`, which natively handles the PyTorch `padding_mode='circular'` necessary for periodic boundary conditions on the Torus.
- Along with the `.onnx` model, the export script extracts the dataset's normalization statistics (`p_mean`, `p_std`, `s_mean`, `s_std`) and saves them to `norm_stats.json`. This ensures the inputs and outputs of the model perfectly match the distributions seen during training.

---

## 2. Technical Details of the Web Implementation

The web client runs the neural operator directly in the browser via WebGPU and WebAssembly.

### 2.1 Inference Engine (`NeuralInference.ts`)
The `NeuralInference` class manages the state loop and the interaction with `onnxruntime-web`.

1. **Initialization:**
   - The engine precomputes the `M_static` metric tensor based on the Torus parameters $R=3.0$ and $r=1.0$. The metric $r(R + r \cos(\theta))$ is calculated for each grid point and normalized to `[0, 1]`.
   - The `norm_stats.json` file is loaded to configure the normalization parameters.
   - The ONNX session is created, requesting the `webgpu` execution provider with a fallback to `wasm`. Multithreading is intentionally locked to 1 thread (`ort.env.wasm.numThreads = 1`) to bypass strict `SharedArrayBuffer` cross-origin isolation requirements in some browsers.

2. **Inference Loop (`predict`):**
   - **Normalization:** The inputs (`P_curr`, `P_prev`, `S_curr`) are packed into a single 4-channel tensor. The `S_curr` tensor is dynamically normalized using `(val - sMean) / sStd`.
   - **Execution:** The ONNX Runtime evaluates the network asynchronously. 
   - **Autoregression:** The states are rolled forward: `P_prev` becomes `P_curr`, and `P_curr` becomes the newly predicted `P_next`.
   - **Denormalization & Upsampling:** The output `P_next` is denormalized (`val * pStd + pMean`) back into the physical scale. Because the model operates on a low-resolution $64 \times 64$ grid (for performance), the output is immediately **bilinearly upsampled** to a crisp $256 \times 256$ texture array in JavaScript before being handed to the renderer.

3. **User Interaction (`injectPulse`):**
   - When a user clicks, the engine receives the UV coordinates ($u \rightarrow \phi, v \rightarrow \theta$).
   - A **Mexican Hat (Ricker) Wavelet** is injected into `S_curr` at the mapped location. 
   - *Crucial Detail:* A zero-mean constraint is enforced on the injected impulse by subtracting the mean of the perturbation across the entire grid. This prevents the closed toroidal system from artificially drifting or inflating with infinite energy over time.

### 2.2 Rendering and React Three Fiber (`WaveEngine.tsx` & `TorusVisualizer.tsx`)
1. **Texture Pipeline:**
   - The $256 \times 256$ upsampled output array is written directly into a `THREE.DataTexture` named `hostTexture`.
   - In the `useFrame` hook, the `WaveEngine` monitors a `neuralBusy` flag. If the ONNX runtime is free, it triggers the next inference step. Once complete, it sets `hostTexture.needsUpdate = true` to push the new data to the GPU.
   
2. **Visualization Shader (`TorusVisualizer.tsx`):**
   - The `TorusVisualizer` maps the `hostTexture` to the Torus geometry using a custom `ShaderMaterial`.
   - The **Fragment Shader** maps the pressure values to a premium color palette: positive peaks are styled in *Rose Gold* (`colorPos`), and negative troughs in *Bioluminescent Mint* (`colorNeg`). 
   - The shader implements procedural lighting calculations (diffuse, specular, rim lighting) utilizing the view direction and the normal vectors. This makes the wave propagation visually pop and react to the camera angle, resulting in a highly aesthetic, interactive 3D topology.
