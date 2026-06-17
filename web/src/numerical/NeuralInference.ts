import * as ort from 'onnxruntime-web';

export class NeuralInference {
  private session: ort.InferenceSession | null = null;
  private metricEmbed: Float32Array; // precomputed (64*64)
  private pMean: number = 0;
  private pStd: number = 1;
  private sMean: number = 0;
  private sStd: number = 1;
  private P_curr: Float32Array; // (64*64) - current pressure state (normalized)
  private P_prev: Float32Array; // (64*64) - previous pressure state (normalized)
  private ready: boolean = false;
  private lastInferenceMs: number = 0;

  constructor() {
    this.P_curr = new Float32Array(64 * 64);
    this.P_prev = new Float32Array(64 * 64);
    this.metricEmbed = new Float32Array(64 * 64);

    // Compute metric embed:
    // theta_grid = linspace(0, 2π, 64)
    // metric = r*(R + r*cos(theta)), normalized to [0,1]
    const R = 3.0;
    const r = 1.0;
    const maxMetric = r * (R + r);
    const minMetric = r * (R - r);

    for (let j = 0; j < 64; j++) {
      const theta = (j / 64) * 2 * Math.PI;
      const metric = r * (R + r * Math.cos(theta));
      const m_norm = (metric - minMetric) / (maxMetric - minMetric);

      for (let i = 0; i < 64; i++) {
        // Data layout: [phi, theta] i.e., [i, j]
        this.metricEmbed[j * 64 + i] = m_norm;
      }
    }
  }

  async init(modelUrl: string, statsUrl: string): Promise<void> {
    try {
      // Load stats
      const statsRes = await fetch(statsUrl);
      const stats = await statsRes.json();
      this.pMean = stats.p_mean;
      this.pStd = stats.p_std;
      this.sMean = stats.s_mean;
      this.sStd = stats.s_std;

      // Disable multithreading to avoid SharedArrayBuffer requirements
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = true; // Run WASM in a background worker to prevent UI freezing

      // Load model with WebGPU backend
      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['webgpu', 'webgl', 'wasm'] // Fallback to webgl/wasm if webgpu is unavailable
      });

      this.ready = true;
      console.log("Neural Inference initialized successfully.");
    } catch (e) {
      console.error("Failed to init NeuralInference:", e);
    }
  }

  async predict(sourceTensor: Float32Array | null): Promise<Float32Array> {
    if (!this.ready || !this.session) {
      return new Float32Array(256 * 256);
    }

    const t0 = performance.now();

    // Prepare S_curr (64x64)
    const S_curr = new Float32Array(64 * 64);
    if (sourceTensor) {
      for (let i = 0; i < 64 * 64; i++) {
        S_curr[i] = (sourceTensor[i] - this.sMean) / this.sStd;
      }
    } else {
      for (let i = 0; i < 64 * 64; i++) {
        S_curr[i] = (0 - this.sMean) / this.sStd;
      }
    }

    // Pack inputs into (1, 4, 64, 64) tensor
    const inputData = new Float32Array(4 * 64 * 64);
    inputData.set(this.P_curr, 0);                 // Channel 0: P_curr
    inputData.set(this.P_prev, 64 * 64);           // Channel 1: P_prev
    inputData.set(S_curr, 2 * 64 * 64);            // Channel 2: S_curr
    inputData.set(this.metricEmbed, 3 * 64 * 64);  // Channel 3: M_static

    const tensor = new ort.Tensor('float32', inputData, [1, 4, 64, 64]);

    // Run inference
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.session.inputNames[0]] = tensor;
    
    const results = await this.session.run(feeds);
    const outputTensor = results[this.session.outputNames[0]];
    const P_next = outputTensor.data as Float32Array;

    // Roll state
    this.P_prev.set(this.P_curr);
    this.P_curr.set(P_next);

    // Denormalize
    const P_phys = new Float32Array(64 * 64);
    for (let i = 0; i < 64 * 64; i++) {
      P_phys[i] = P_next[i] * this.pStd + this.pMean;
    }

    // Upsample 64x64 -> 256x256 bilinearly
    const out = new Float32Array(256 * 256);
    const scale = 64 / 256;
    for (let oy = 0; oy < 256; oy++) {
      const sy = oy * scale;
      const j0 = Math.floor(sy);
      const j1 = (j0 + 1) % 64;
      const beta = sy - j0;

      for (let ox = 0; ox < 256; ox++) {
        const sx = ox * scale;
        const i0 = Math.floor(sx);
        const i1 = (i0 + 1) % 64;
        const alpha = sx - i0;

        const p00 = P_phys[j0 * 64 + i0];
        const p10 = P_phys[j0 * 64 + i1];
        const p01 = P_phys[j1 * 64 + i0];
        const p11 = P_phys[j1 * 64 + i1];

        out[oy * 256 + ox] = (1 - alpha) * (1 - beta) * p00
                           + alpha * (1 - beta) * p10
                           + (1 - alpha) * beta * p01
                           + alpha * beta * p11;
      }
    }

    const t1 = performance.now();
    this.lastInferenceMs = t1 - t0;
    
    return out;
  }

  injectPulse(theta0: number, phi0: number, impulse: number): Float32Array {
    const S = new Float32Array(64 * 64);
    
    // Mexican Hat (Ricker) wavelet logic
    const radius = 6;
    const sigma = 2.0;
    
    // Translate 0-1 uv space into 0-64 space
    // Assuming theta0, phi0 are in [0, 1] range based on WaveEngine logic
    const cx = phi0 * 64; // ox
    const cy = theta0 * 64; // oy
    
    let sum = 0;
    for(let j = 0; j < 64; j++) {
      for(let i = 0; i < 64; i++) {
        // distance with periodic wrapping
        let dx = Math.abs(i - cx);
        if (dx > 32) dx = 64 - dx;
        
        let dy = Math.abs(j - cy);
        if (dy > 32) dy = 64 - dy;
        
        const r2 = dx*dx + dy*dy;
        if (r2 <= radius*radius) {
            const v = (2 - r2/(sigma*sigma)) * Math.exp(-r2 / (2 * sigma * sigma));
            S[j * 64 + i] = v * impulse;
            sum += S[j * 64 + i];
        }
      }
    }
    
    // zero mean constraint
    const mean = sum / (64 * 64);
    for(let k = 0; k < 64 * 64; k++) {
      S[k] -= mean;
    }
    
    return S;
  }

  reset(): void {
    this.P_curr.fill(0);
    this.P_prev.fill(0);
  }

  getLastInferenceMs(): number {
    return this.lastInferenceMs;
  }

  isReady(): boolean {
    return this.ready;
  }

  async release(): Promise<void> {
    if (this.session) {
      try {
        await this.session.release();
        this.session = null;
        this.ready = false;
        console.log("Neural Inference session released.");
      } catch (e) {
        console.error("Error releasing Neural Inference session:", e);
      }
    }
  }
}
