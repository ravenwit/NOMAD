import * as ort from 'onnxruntime-web';

// --- Float16 Utilities ---
function float32ToFloat16(val: number): number {
    const floatView = new Float32Array(1);
    const int32View = new Int32Array(floatView.buffer);
    
    floatView[0] = val;
    const x = int32View[0];
    
    const sign = (x >> 16) & 0x8000;
    let exp = (x >> 23) & 0xff;
    let mantissa = x & 0x007fffff;
    
    if (exp === 0) {
        return sign;
    } else if (exp === 255) {
        return sign | 0x7c00 | (mantissa ? 1 : 0);
    } else {
        exp = exp - 127 + 15;
        if (exp >= 31) {
            return sign | 0x7c00;
        } else if (exp <= 0) {
            mantissa = (mantissa | 0x00800000) >> (1 - exp);
            return sign | (mantissa >> 13);
        } else {
            return sign | (exp << 10) | (mantissa >> 13);
        }
    }
}

function float16ToFloat32(val: number): number {
    const floatView = new Float32Array(1);
    const int32View = new Int32Array(floatView.buffer);
    
    const sign = (val & 0x8000) << 16;
    const exp = (val & 0x7c00) >> 10;
    const mantissa = val & 0x03ff;
    
    if (exp === 0) {
        if (mantissa === 0) {
            int32View[0] = sign;
        } else {
            let m = mantissa;
            let e = 0;
            while ((m & 0x0400) === 0) {
                m <<= 1;
                e++;
            }
            int32View[0] = sign | ((127 - 15 - e + 1) << 23) | ((m & 0x03ff) << 13);
        }
    } else if (exp === 31) {
        int32View[0] = sign | 0x7f800000 | (mantissa << 13);
    } else {
        int32View[0] = sign | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }
    
    return floatView[0];
}

function toHalf(f32Array: Float32Array): Uint16Array {
    const u16 = new Uint16Array(f32Array.length);
    for (let i = 0; i < f32Array.length; i++) {
        u16[i] = float32ToFloat16(f32Array[i]);
    }
    return u16;
}

function fromHalf(u16Array: Uint16Array): Float32Array {
    const f32 = new Float32Array(u16Array.length);
    for (let i = 0; i < u16Array.length; i++) {
        f32[i] = float16ToFloat32(u16Array[i]);
    }
    return f32;
}
// -------------------------


export class GeoFNOInference {
  private session: ort.InferenceSession | null = null;
  
  private geomFeatures: Float32Array; // (3, 256, 256)
  private p_in: Float32Array; // (3, 256, 256)
  private s_in: Float32Array; // (3, 256, 256)
  
  private P_out_buffer: Float32Array; // (30, 256, 256)
  private playbackIndex: number = 0;
  
  private ready: boolean = false;
  private isPredicting: boolean = false;
  private nextBatchReady: boolean = false;
  private next_P_out_buffer: Float32Array | null = null;
  
  private lastInferenceMs: number = 0;

  // Assuming p_scale and s_scale are close to 1.0 based on the python script.
  private pScale: number = 1.0;
  private sScale: number = 1.0;

  constructor() {
    this.p_in = new Float32Array(3 * 256 * 256);
    this.s_in = new Float32Array(3 * 256 * 256);
    this.geomFeatures = new Float32Array(3 * 256 * 256);
    this.P_out_buffer = new Float32Array(30 * 256 * 256);
    this.next_P_out_buffer = new Float32Array(30 * 256 * 256);

    const R = 3.0;
    const r = 1.0;
    const maxMetric = r * (R + r);
    const minMetric = r * (R - r);

    // Populate geomFeatures: [m_norm, THETA/2pi, PHI/2pi]
    for (let j = 0; j < 256; j++) {
      const theta = (j / 256) * 2 * Math.PI;
      const metric = r * (R + r * Math.cos(theta));
      const m_norm = (metric - minMetric) / (maxMetric - minMetric);

      for (let i = 0; i < 256; i++) {
        const phi = (i / 256) * 2 * Math.PI;
        
        const idx = j * 256 + i;
        // Channel 0: m_norm
        this.geomFeatures[idx] = m_norm;
        // Channel 1: THETA / 2pi
        this.geomFeatures[256 * 256 + idx] = theta / (2 * Math.PI);
        // Channel 2: PHI / 2pi
        this.geomFeatures[2 * 256 * 256 + idx] = phi / (2 * Math.PI);
      }
    }
  }

  async init(modelUrl: string): Promise<void> {
    try {
      ort.env.wasm.numThreads = 1;

      this.session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm']
      });

      this.ready = true;
      console.log("GeoFNO Neural Inference initialized successfully.");
      
      // Kick off the first prediction
      await this.runInference(this.p_in, this.s_in, this.P_out_buffer);
      this.playbackIndex = 0;
      
      // Start background fetch for the next batch
      this.triggerNextBatch();

    } catch (e) {
      console.error("Failed to init GeoFNOInference:", e);
    }
  }

  private async runInference(p_in_arr: Float32Array, s_in_arr: Float32Array, out_buffer: Float32Array): Promise<void> {
    if (!this.session) return;
    this.isPredicting = true;
    const t0 = performance.now();

    // Scale inputs
    const p_in_scaled = new Float32Array(p_in_arr.length);
    const s_in_scaled = new Float32Array(s_in_arr.length);
    for (let i = 0; i < p_in_arr.length; i++) {
      p_in_scaled[i] = p_in_arr[i] / this.pScale;
      s_in_scaled[i] = s_in_arr[i] / this.sScale;
    }

    // Convert to Float16 (Uint16Array)
    const p_in_f16 = toHalf(p_in_scaled);
    const s_in_f16 = toHalf(s_in_scaled);
    const geom_f16 = toHalf(this.geomFeatures);

    const t_p_in = new ort.Tensor('float16', p_in_f16, [1, 3, 256, 256]);
    const t_s_in = new ort.Tensor('float16', s_in_f16, [1, 3, 256, 256]);
    const t_geom = new ort.Tensor('float16', geom_f16, [1, 3, 256, 256]);

    const feeds: Record<string, ort.Tensor> = {
      p_in: t_p_in,
      s_in: t_s_in,
      geom_features: t_geom
    };

    try {
      const results = await this.session.run(feeds);
      let p_out_data = results['p_out'].data;
      
      // Convert back to Float32 if output is Float16 (Uint16Array)
      let p_out: Float32Array;
      if (p_out_data instanceof Uint16Array) {
          p_out = fromHalf(p_out_data);
      } else {
          p_out = p_out_data as Float32Array;
      }

      // De-scale and copy to out_buffer
      for (let i = 0; i < p_out.length; i++) {
        out_buffer[i] = p_out[i] * this.pScale;
      }
      
      this.lastInferenceMs = performance.now() - t0;
    } catch (e) {
      console.error("GeoFNO inference failed:", e);
    } finally {
      this.isPredicting = false;
    }
  }

  private triggerNextBatch() {
    if (this.isPredicting) return;
    
    // Extract last 3 frames from P_out_buffer to form the next p_in
    const next_p_in = new Float32Array(3 * 256 * 256);
    // Frames 27, 28, 29
    next_p_in.set(this.P_out_buffer.subarray(27 * 256 * 256, 30 * 256 * 256));
    
    // Next s_in is just zeros (unless user interacts, handled later)
    const next_s_in = new Float32Array(3 * 256 * 256);

    this.nextBatchReady = false;
    this.runInference(next_p_in, next_s_in, this.next_P_out_buffer!).then(() => {
      this.nextBatchReady = true;
    });
  }

  /**
   * Called every frame by the WebGL renderer.
   * Returns the current 256x256 frame and advances the playback index.
   */
  tick(): Float32Array | null {
    if (!this.ready) return null;

    // If we reached the end of the buffer
    if (this.playbackIndex >= 30) {
      if (this.nextBatchReady) {
        // Swap buffers
        const temp = this.P_out_buffer;
        this.P_out_buffer = this.next_P_out_buffer!;
        this.next_P_out_buffer = temp;
        
        this.playbackIndex = 0;
        
        // Immediately start fetching the next batch
        this.triggerNextBatch();
      } else {
        // Stall on the last frame until next batch is ready
        this.playbackIndex = 29;
      }
    }

    const frameSize = 256 * 256;
    const start = this.playbackIndex * frameSize;
    const currentFrame = this.P_out_buffer.subarray(start, start + frameSize);
    
    this.playbackIndex++;
    return currentFrame;
  }

  /**
   * If the user interacts, we instantly halt the normal playback queue,
   * reconstruct p_in from the last 3 visible frames, inject the pulse into s_in,
   * and compute a fresh 30-frame sequence to ensure low-latency responsiveness.
   */
  async injectPulse(theta0: number, phi0: number, impulse: number): Promise<void> {
    if (!this.ready) return;
    
    // Construct new p_in based on what the user actually saw right before clicking
    const new_p_in = new Float32Array(3 * 256 * 256);
    const frameSize = 256 * 256;
    const idx = this.playbackIndex - 1; // currently visible frame index

    for (let t = 0; t < 3; t++) {
      const historicalIdx = idx - (2 - t);
      if (historicalIdx < 0) {
        // Fallback to initial p_in if we just started
        new_p_in.set(this.p_in.subarray(t * frameSize, (t + 1) * frameSize), t * frameSize);
      } else {
        new_p_in.set(this.P_out_buffer.subarray(historicalIdx * frameSize, (historicalIdx + 1) * frameSize), t * frameSize);
      }
    }

    // Construct new s_in (zeros, with impulse in the last frame t=2)
    const new_s_in = new Float32Array(3 * 256 * 256);
    
    const radius = 6;
    const sigma = 2.0;
    const cx = phi0 * 256;
    const cy = theta0 * 256;
    
    let sum = 0;
    for (let j = 0; j < 256; j++) {
      for (let i = 0; i < 256; i++) {
        let dx = Math.abs(i - cx);
        if (dx > 128) dx = 256 - dx;
        
        let dy = Math.abs(j - cy);
        if (dy > 128) dy = 256 - dy;
        
        const r2 = dx * dx + dy * dy;
        if (r2 <= radius * radius) {
          const v = (2 - r2 / (sigma * sigma)) * Math.exp(-r2 / (2 * sigma * sigma));
          const val = v * impulse;
          new_s_in[2 * frameSize + j * 256 + i] = val; // Inject in t=2
          sum += val;
        }
      }
    }
    
    // Zero mean constraint
    const mean = sum / (256 * 256);
    for (let k = 2 * frameSize; k < 3 * frameSize; k++) {
      new_s_in[k] -= mean;
    }

    // Save active state
    this.p_in.set(new_p_in);
    this.s_in.set(new_s_in);

    // Compute fresh batch immediately
    await this.runInference(this.p_in, this.s_in, this.P_out_buffer);
    this.playbackIndex = 0;
    this.triggerNextBatch(); // Precompute next batch
  }

  reset(): void {
    this.p_in.fill(0);
    this.s_in.fill(0);
    this.P_out_buffer.fill(0);
    if (this.next_P_out_buffer) this.next_P_out_buffer.fill(0);
    this.playbackIndex = 0;
  }

  getLastInferenceMs(): number {
    return this.lastInferenceMs;
  }

  isReady(): boolean {
    return this.ready;
  }
}
