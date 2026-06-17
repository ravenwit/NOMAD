import { Client } from "@gradio/client";

export class HuggingFaceInference {
  private client: any = null;
  private P_out_buffer: Float32Array; // (30, 256, 256)
  private playbackIndex: number = 0;
  
  private ready: boolean = false;
  private isPredicting: boolean = false;
  private lastInferenceMs: number = 0;

  constructor() {
    this.P_out_buffer = new Float32Array(30 * 256 * 256);
  }

  async init(): Promise<void> {
    try {
      this.client = await Client.connect("raven-shakir/nomad");
      this.ready = true;
      console.log("HuggingFace Inference initialized successfully.");
      
      // Fire an initial pulse so the torus isn't blank
      await this.injectPulse(0.5, 0.5);
    } catch (e) {
      console.error("Failed to init HuggingFaceInference:", e);
    }
  }

  async injectPulse(theta0_uv: number, phi0_uv: number): Promise<void> {
    if (!this.ready || !this.client) return;
    
    const theta0 = theta0_uv * 2.0 * Math.PI;
    const phi0 = phi0_uv * 2.0 * Math.PI;

    this.isPredicting = true;
    const t0 = performance.now();
    try {
      const result = await this.client.predict("/run_simulation", {
        theta: theta0,
        phi: phi0,
      });
      const data = result.data[0] as number[];
      
      // Update buffer
      this.P_out_buffer.set(data);
      this.playbackIndex = 0;
      this.lastInferenceMs = performance.now() - t0;
    } catch (e) {
      console.error("HuggingFace inference failed:", e);
    } finally {
      this.isPredicting = false;
    }
  }

  tick(): Float32Array | null {
    if (!this.ready) return null;

    if (this.playbackIndex >= 30) {
      // Hold on the last frame
      this.playbackIndex = 29;
    }

    const frameSize = 256 * 256;
    const start = this.playbackIndex * frameSize;
    const currentFrame = this.P_out_buffer.subarray(start, start + frameSize);
    
    // Always advance frames, regardless of prediction state
    if (this.playbackIndex < 29) {
      this.playbackIndex++;
    }
    
    return currentFrame;
  }

  reset(): void {
    this.P_out_buffer.fill(0);
    this.playbackIndex = 0;
  }

  getLastInferenceMs(): number {
    return this.lastInferenceMs;
  }

  isReady(): boolean {
    return this.ready;
  }
}
