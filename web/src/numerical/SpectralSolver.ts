import { ComplexArray, fft2d } from "./fft";

export class TSSolver {
  N_theta: number;
  N_phi: number;
  R: number;
  r: number;
  c: number;
  dt: number;

  P_curr: Float64Array;
  P_prev: Float64Array;

  k_theta: Float64Array;
  k_phi: Float64Array;

  g_inv_tt: Float64Array;
  g_inv_pp: Float64Array;
  gamma_term: Float64Array;

  complexBuffer1: ComplexArray;
  
  // Pre-allocated derivative buffers (Frequency Domain)
  d1_re: Float64Array;
  d1_im: Float64Array;
  d2t_re: Float64Array;
  d2t_im: Float64Array;
  d2p_re: Float64Array;
  d2p_im: Float64Array;

  // Pre-allocated spatial derivative buffers
  dP_dtheta: Float64Array;
  d2P_dtheta2: Float64Array;
  d2P_dphi2: Float64Array;
  
  source: Float64Array;

  constructor(R: number = 1.5, r: number = 0.5, c: number = 1.0, N_theta: number = 256, N_phi: number = 256, CFL: number = 0.1) {
    this.R = R;
    this.r = r;
    this.c = c;
    this.N_theta = N_theta;
    this.N_phi = N_phi;

    const N = N_theta * N_phi;
    const d_theta = 2 * Math.PI / N_theta;
    const d_phi = 2 * Math.PI / N_phi;
    const min_dx = Math.min(r * d_theta, (R - r) * d_phi);
    this.dt = CFL * min_dx / c;

    this.P_curr = new Float64Array(N);
    this.P_prev = new Float64Array(N);
    this.source = new Float64Array(N);

    this.k_theta = new Float64Array(N_theta);
    for (let i = 0; i < N_theta; i++) {
      let freq = i < N_theta / 2 ? i : i - N_theta;
      this.k_theta[i] = freq;
    }

    this.k_phi = new Float64Array(N_phi);
    for (let i = 0; i < N_phi; i++) {
        let freq = i < N_phi / 2 ? i : i - N_phi;
        this.k_phi[i] = freq;
    }

    this.g_inv_tt = new Float64Array(N);
    this.g_inv_pp = new Float64Array(N);
    this.gamma_term = new Float64Array(N);

    for (let i = 0; i < N_theta; i++) {
      const theta = i * d_theta; 
      const g_tt = 1.0 / (r * r);
      for (let j = 0; j < N_phi; j++) {
        const idx = i * N_phi + j;
        this.g_inv_tt[idx] = g_tt;
        this.g_inv_pp[idx] = 1.0 / Math.pow(R + r * Math.cos(theta), 2);
        this.gamma_term[idx] = -Math.sin(theta) / (r * (R + r * Math.cos(theta)));
      }
    }

    this.complexBuffer1 = new ComplexArray(N);
    
    // Init derivative buffers
    this.d1_re = new Float64Array(N);
    this.d1_im = new Float64Array(N);
    this.d2t_re = new Float64Array(N);
    this.d2t_im = new Float64Array(N);
    this.d2p_re = new Float64Array(N);
    this.d2p_im = new Float64Array(N);

    this.dP_dtheta = new Float64Array(N);
    this.d2P_dtheta2 = new Float64Array(N);
    this.d2P_dphi2 = new Float64Array(N);
  }

  computeLaplaceBeltrami(P: Float64Array, target: Float64Array) {
    const N = this.N_theta * this.N_phi;
    const buf = this.complexBuffer1;

    // Load P into buffer for FFT
    for (let i = 0; i < N; i++) {
      buf.real[i] = P[i];
      buf.imag[i] = 0;
    }

    fft2d(buf, this.N_phi, this.N_theta, false);

    // Compute Derivatives in Frequency Domain (Zero-allocation)
    for (let i = 0; i < this.N_theta; i++) {
      const kt = this.k_theta[i];
      for (let j = 0; j < this.N_phi; j++) {
        const idx = i * this.N_phi + j;
        const re = buf.real[idx];
        const im = buf.imag[idx];
        const kp = this.k_phi[j];

        // d/dtheta = i * k_theta * F
        this.d1_re[idx] = -kt * im;
        this.d1_im[idx] = kt * re;

        // d2/dtheta2 = -k_theta^2 * F
        this.d2t_re[idx] = -kt * kt * re;
        this.d2t_im[idx] = -kt * kt * im;

        // d2/dphi2 = -k_phi^2 * F
        this.d2p_re[idx] = -kp * kp * re;
        this.d2p_im[idx] = -kp * kp * im;
      }
    }

    // IFFT 1: First derivative
    for(let i=0; i<N; i++) { buf.real[i] = this.d1_re[i]; buf.imag[i] = this.d1_im[i]; }
    fft2d(buf, this.N_phi, this.N_theta, true);
    for(let i=0; i<N; i++) this.dP_dtheta[i] = buf.real[i];

    // IFFT 2: Second theta derivative
    for(let i=0; i<N; i++) { buf.real[i] = this.d2t_re[i]; buf.imag[i] = this.d2t_im[i]; }
    fft2d(buf, this.N_phi, this.N_theta, true);
    for(let i=0; i<N; i++) this.d2P_dtheta2[i] = buf.real[i];

    // IFFT 3: Second phi derivative
    for(let i=0; i<N; i++) { buf.real[i] = this.d2p_re[i]; buf.imag[i] = this.d2p_im[i]; }
    fft2d(buf, this.N_phi, this.N_theta, true);
    for(let i=0; i<N; i++) this.d2P_dphi2[i] = buf.real[i];

    // Assemble laplacian
    for (let i = 0; i < N; i++) {
      target[i] = (this.g_inv_tt[i] * this.d2P_dtheta2[i]) +
                  (this.gamma_term[i] * this.dP_dtheta[i]) +
                  (this.g_inv_pp[i] * this.d2P_dphi2[i]);
    }
  }

  // Inject a zero-mean Mexican Hat (Ricker wavelet) pulse
  injectPulse(theta0: number, phi0: number, impulse: number = 10000.0) {
    const d_theta = 2 * Math.PI / this.N_theta;
    const d_phi = 2 * Math.PI / this.N_phi;
    
    const radius = 6;
    const sigma = 2.0;
    const j0 = Math.floor(theta0 / d_theta);
    const i0 = Math.floor(phi0 / d_phi);

    let sum = 0.0;
    const points = [];
    for (let dj = -radius; dj <= radius; dj++) {
      for (let di = -radius; di <= radius; di++) {
        const distSq = di * di + dj * dj;
        if (distSq <= radius * radius) {
          const r_sq_over_sigma_sq = distSq / (sigma * sigma);
          const value = (2.0 - r_sq_over_sigma_sq) * Math.exp(-distSq / (2.0 * sigma * sigma));
          points.push({ di, dj, value });
          sum += value;
        }
      }
    }
    const correction = sum / points.length;

    for (const p of points) {
      const j = (j0 + p.dj + this.N_theta) % this.N_theta;
      const i = (i0 + p.di + this.N_phi) % this.N_phi;
      const idx = j * this.N_phi + i;
      this.source[idx] += (p.value - correction) * impulse;
    }
  }

  step(steps: number) {
    const N = this.N_theta * this.N_phi;
    const laplacian = new Float64Array(N);

    for (let s = 0; s < steps; s++) {
      this.computeLaplaceBeltrami(this.P_curr, laplacian);
      for (let i = 0; i < N; i++) {
        // P_next = 2*P_curr - P_prev + dt^2 * (c^2 * L + S)
        const accel = (this.c * this.c * laplacian[i]) + this.source[i];
        const P_next = 2 * this.P_curr[i] - this.P_prev[i] + this.dt * this.dt * accel;
        this.P_prev[i] = this.P_curr[i];
        this.P_curr[i] = P_next;
      }
      // Source is impulsive per step (cleared like in FDTD)
      this.source.fill(0);
    }
  }

  reset() {
    this.P_curr.fill(0);
    this.P_prev.fill(0);
    this.source.fill(0);
  }

  getFloat32Array(): Float32Array {
    return new Float32Array(this.P_curr);
  }
}
