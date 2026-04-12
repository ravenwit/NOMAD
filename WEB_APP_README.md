# 3D Interactive Acoustic Wave Torus - Web Application

This document outlines the architecture, technology stack, and implementation steps required to build a highly interactive, browser-based 3D simulation of our Acoustic Wave Torus. It allows users to interact with the manifold in real-time using their mouse, using a pre-trained PyTorch deep learning model running inference in the browser.

## Overall Architecture

Instead of running a heavy numerical pseudospectral wave simulator in the browser, the web application relies entirely on our **Autoregressive Geometric ConvLSTM** to predict wave propagation. The inference loop runs natively using client-side execution, bypassing the need for a persistent Python backend connection.

The flow is as follows:
`Mouse Input (Raycast UV)` -> `Source Tensor S(t)` -> `ONNX Inference Engine` -> `Pressure Field P(t+1)` -> `Three.js Shader Material`

---

## Technical Stack

1. **Model Deployment**: PyTorch $ \to $ ONNX $ \to $ **ONNX Runtime Web** (Execution via WebGL / WebGPU WASM)
2. **3D Visualizer**: **Three.js** (or React Three Fiber if using React)
3. **Materials/Shaders**: Custom **GLSL Vertex and Fragment Shaders**
4. **Frontend Framework**: Vanilla JS, React, or Next.js.

---

## Implementation Steps

### Phase 1: Train and Export the Model (e.g., on Kaggle)

You must first train your Physics-Informed Autoregressive Model elsewhere utilizing GPUs. 

1. **Train your model**: Use `src/train.py` connected to Kaggle's A100/T4 GPUs.
2. **Export to ONNX**: Once the generator is trained, serialize it into an interoperable format (`.onnx`).
   ```python
   import torch
   
   # Load trained ConvLSTM generator
   model = AutoregressiveGenerator(input_channels=3)
   model.load_state_dict(torch.load("best_generator.pth"))
   model.eval()

   # Create dummy inputs corresponding to your sequence variables
   dummy_P = torch.randn(1, 3, 64, 64)
   dummy_S = torch.randn(1, 3, 64, 64)
   dummy_h = model.init_hidden(1, 64, 64, device='cpu')
   dummy_z = torch.randn(1, 1, 1, 1)

   # Export 
   torch.onnx.export(model, 
                     (dummy_P, dummy_S, dummy_h, dummy_z), 
                     "acoustic_torus.onnx",
                     opset_version=14,
                     input_names=['P_curr', 'S_curr', 'h_curr', 'z'],
                     output_names=['P_next', 'h_next'])
   ```

### Phase 2: Building the 3D Web Environment

Set up a standard web project and implement a Three.js scene containing a Torus.

1. **Initialize Three.js**: Setup WebGL renderer, scene, and camera.
2. **Create the Torus**: Use `new THREE.TorusGeometry(R, r, radialSegments, tubularSegments)`. 
   - *Ensure the segments match your model's grid resolution (e.g., $N_\theta = 64$, $N_\phi = 64$)*.
3. **Custom Shader Material**: Don't use a standard material. Write a custom `ShaderMaterial` that takes a `DataTexture` as a uniform. This texture will carry the neural network's pressure outputs. The Vertex Shader will displace the mesh vertices based on the pressure, and the Fragment Shader will color map the waves (e.g., red for high pressure, blue for low).

### Phase 3: The Interaction System (Raycasting)

To allow the user to inject source disturbances:
1. Attach a **Raycaster** to mouse move and click events.
2. Cast a ray into the scene against the Torus mesh.
3. If an intersection occurs, extract the `uv` coordinates (`intersection.uv`).
4. Map the `u` and `v` coordinates to your discrete $64 \times 64$ grid.
5. Create a Javascript `Float32Array` representing your $S_{\mu\nu}$ source tensor. Add a Gaussian blob (energy spike) at the mapped UV index whenever the user clicks or drags.

### Phase 4: Integrating ONNX Runtime Web (Inference Loop)

Load `onnxruntime-web` into your project and structure the render loop:

1. **Load the Model**:
   ```javascript
   import * as ort from 'onnxruntime-web';
   const session = await ort.InferenceSession.create('./acoustic_torus.onnx', { executionProviders: ['webgl'] });
   ```
2. **Recursive Inference Hook**: Inside your `requestAnimationFrame(animate)` loop:
   - Prepare the input tensors from your JS arrays (current pressure, mouse source disturbance, hidden states).
   - Await the inference step (`session.run(feeds)`).
   - Extract the new `P_next` array.
   - Overwrite the old pressure state and hidden state buffers with the new output buffers.
3. **Update the UI**: Look at the output pressure array from the model, copy it over to a `THREE.DataTexture`, and push this texture to the Torus slice's `ShaderMaterial` uniform via `.needsUpdate = true`. 

### Performance Considerations for the Web
- Due to the nature of recurrent architectures, hidden states must be carefully managed in JavaScript memory to avoid garbage collection stuttering.
- The `webgpu` execution provider for ONNX Runtime Web is modern and highly recommended for tensor convolution over `webgl`. Ensure your layers rely on optimized matrix multiplication ops!
