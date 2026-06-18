<div align="center">
  
# 🌊 NOMAD 
**Neural Operator Mapping for Acoustic Dynamics**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](#)
[![React](https://img.shields.io/badge/React-UI-61dafb?logo=react&logoColor=black)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

*A state-of-the-art SciML framework bridging differential geometry and deep learning to simulate acoustic wave propagation on non-Euclidean manifolds.*

</div>

---

## 📖 Overview

NOMAD is a physics-informed Scientific Machine Learning (SciML) pipeline designed to formally simulate and predict acoustic wave propagation on a closed toroidal manifold ($\mathbb{T}^2$) utilizing a **Geometry-Aware Fourier Neural Operator (Geo-FNO)**. 

Traditional translation-invariant networks (like CNNs or vanilla FNOs) fail mathematically on curved surfaces due to heterogeneous metric tensors. NOMAD solves this by learning a **Diffeomorphic Pullback**, mapping the physical curved manifold into a flat latent space where the PDE operator is translationally invariant.

### ✨ Key Features
- 📐 **Rigorous Physical Simulation**: High-fidelity ground truth PDE solver via Pseudospectral Fourier Collocation natively on the Torus.
- 🧠 **Geometry-Aware Deep Learning**: A highly optimized PyTorch training harness featuring **Pushforward Training** (autoregressive stabilization) and memory-safe lazy-loading (HDF5).
- 📊 **Comprehensive Evaluation**: Generates rigorous spatial prediction plots, temporal L2 error accumulation curves, and energy conservation metrics.
- 🌐 **Interactive 3D Web UI**: A React application for real-time 3D wave visualization. Currently, it compares the high-fidelity ground truth against the Geo-FNO predictions using heavily optimized, statically-baked binary (`.bin`) sequences.

---

## 📂 Directory Structure

```text
NOMAD/
├── 📁 src/                  # Core Python Pipeline
│   ├── 📄 data/             # memory-safe h5py lazy-loading (ChunkedTorusDataset)
│   ├── 📄 models/           # Architecture definitions (GeoFNO, DiffeomorphismNet)
│   ├── 📄 numerical/        # Ground-truth pseudospectral wave solver
│   └── 📄 training/         # FastTrainer, MLflow tracking, Pushforward logic
├── 📁 web/                  # React + Vite frontend for 3D Torus visualizations
├── 📁 report/               # Formal mathematical documentation and figures
├── 📁 scripts/              # Utility scripts (ONNX compression, binary baking)
├── 📁 tests/               # pytest suite ensuring geometric stability
├── 📄 config.yaml           # Unified configuration for generation, training, and evaluation
└── 📄 main.py               # Main CLI entry point
```

---

## 🛠️ Installation

### 1. Python Environment (Backend)
Clone the repository and install the strict dependencies:
```bash
git clone https://github.com/your-username/NOMAD.git
cd NOMAD
pip install -r requirements.txt
```

### 2. Node Environment (Frontend)
Navigate to the web directory and install the UI dependencies:
```bash
cd web
npm install
npm run dev
```

---

## ⚙️ Configuration (`config.yaml`)

The entire ML and simulation pipeline is unified under a single `config.yaml` file:
- **`generation`**: Control the physical geometry ($R$, $r$), spectral grid resolution, total rollouts, and time-step scaling.
- **`training`**: Define the Geo-FNO hyperparameters (`modes`, `width`), `batch_size`, `epochs`, and crucially, the **`unroll_steps`** to control Pushforward training depth.
- **`evaluation`**: Determine independent evaluation sets, testing resolution, and maximum autoregressive evaluation steps.

---

## 🚀 How to Run the Pipeline

### 1. 🌊 Generate the Ground-Truth Dataset
This uses the discrete Fourier collocation method and leapfrog time-stepping to simulate wave propagation physically on the manifold surface. The simulation automatically logs the physical clocks (`dt`, `dt_macro`, `c`).
```bash
python main.py --mode generate
```

### 2. 🧠 Train the Geo-FNO
Train the neural operator to map the initial conditions to the future acoustic field. The code tracks validation MSE and learning rates in `mlflow`.
```bash
python main.py --mode train
```
> **💡 Tip**: To leverage autoregressive Pushforward training stabilization, simply increase `unroll_steps` in the `config.yaml` to `> 1`.

### 3. 📈 Evaluate the Model
Evaluate the trained Geo-FNO on unseen data. This step strictly focuses on quantitative validation: it runs autoregressive inference over the prediction sequence and generates rigorous evaluation plots, including the compounding L2 error curves and spatial prediction visualizations.
```bash
python main.py --mode evaluate
```

### 4. 🎬 Bake the Web App Binary Sequences
To visualize the Geo-FNO side-by-side with the ground truth simulator in the React app, you must bake the test scenarios into contiguous flat memory binary arrays (`.bin`). We utilize the `CHORUS BAKING STUDIO` script to generate high-speed `Float32` exports of complex physical scenarios (like `quad_symmetry`, `cascade`, and `chaos` pulses). These `.bin` files are routed automatically to the web app for browser rendering.
```bash
python scripts/bake_web_binaries.py
```

### 5. 🌐 Run the 3D Web UI
Launch the interactive React application. The frontend loads the baked `.bin` data to render interactive predictions on a 3D Torus.
```bash
cd web
npm run dev
```

---

## 🔮 Future Work: ONNX Compression & In-Browser Inference

Currently, the web UI relies on pre-baked binary outputs. Future iterations of this repository will push the entire inference engine directly into the browser utilizing **ONNX Web Runtime**.

Our architecture is fully compatible with ONNX quantization frameworks to compress the massive 151M parameter models (`modes=24, width=128`) into lightweight web-ready files using Half Precision (FP16) or Integer Quantization (INT8). 

You can run the full conversion suite (which automatically routes the compressed models to `web/public/models`) using our dedicated ONNX Compression Engine:
```bash
python scripts/onnx_compression_engine.py
```

This reduces a massive 1.7 GB PyTorch checkpoint into a fraction of its size, enabling real-time neural operator inference entirely on edge clients!

---

## 🧪 Running the Test Suite
A strict `pytest` suite guarantees the core stability of the geometric parameters and dynamic slicing algebra:
```bash
python3 -m pytest tests/
```

---

## 📚 Mathematical Documentation

For an elaborate theoretical breakdown of:
- The Torus 3D parameterization and Riemannian metric tensor formulation.
- The Laplace-Beltrami operator discretization.
- Why standard Periodic U-Nets and vanilla FNOs fail mathematically on curved spaces.
- How the Geo-FNO resolves these topological limitations.

Please read the formal paper and empirical results located in [report/nomad.md](report/nomad.md).
