4. The async predict blocks every render frame
In 

WaveEngine.tsx:293-311
, neural.predict() is called every useFrame. While it's async, the WASM computation itself is CPU-bound and blocks. New frames only dispatch after the previous prediction resolves (neuralBusy guard), so you get at most ~1 inference per ~50-100ms instead of the spectral solver's ~3-5ms per worker response.

5. The bilinear upsample adds further overhead
The 

64→256 bilinear upsampling loop
 in JavaScript adds another 256² = 65K iterations of per-pixel interpolation after each inference.

How to Actually Make It Faster
Here are the options, ranked by impact:

Switch to WebGPU execution provider — executionProviders: ['webgpu'] would run the Conv2d ops on the GPU. This is the single biggest win (10-100×).

Use the FNO instead of UNet — your 

FNO2d
 at modes=12, width=32 is dramatically lighter (~200K params, ~50M FLOPs). Export that to ONNX instead.

Enable multi-threading — set ort.env.wasm.numThreads = navigator.hardwareConcurrency and configure proper SharedArrayBuffer headers (COOP/COEP).

Run inference in a WebWorker — like you already do for the spectral solver. This frees the main thread for rendering.

Reduce model size — the UNet bottleneck at 1024 channels on a 4×4 spatial grid is overkill. You could halve the channels throughout or use bilinear=True to cut decoder params in half.

Would you like me to implement any of these optimizations?

ONNX model location finished
The background search confirms the ONNX model is 66 MB (both in web/public/ and web/dist/), matching the PyTorch checkpoint exactly. This validates the analysis — you're shipping the full 26M-parameter UNet through single-threaded WASM.

The most impactful fix would be exporting the much lighter FNO2d (~200K params) to ONNX instead, or switching to the webgpu execution provider. Let me know which direction you'd like to go.