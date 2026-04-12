export class ComplexArray {
  real: Float64Array;
  imag: Float64Array;
  length: number;

  constructor(length: number) {
    this.length = length;
    this.real = new Float64Array(length);
    this.imag = new Float64Array(length);
  }
}

// In-place Radix-2 Cooley-Tukey FFT
// Optimized with precomputed twiddle factors and shared scratch buffers

const MAX_SIZE = 1024;
const twiddleCache = new Map<number, {re: Float64Array, im: Float64Array}>();

function getTwiddles(n: number, invert: boolean) {
  const key = invert ? -n : n;
  if (!twiddleCache.has(key)) {
    const re = new Float64Array(n / 2);
    const im = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
        const angle = (invert ? 2 : -2) * Math.PI * i / n;
        re[i] = Math.cos(angle);
        im[i] = Math.sin(angle);
    }
    twiddleCache.set(key, {re, im});
  }
  return twiddleCache.get(key)!;
}

export function fft1d(x_re: Float64Array, x_im: Float64Array, invert: boolean = false) {
  const n = x_re.length;
  if (n <= 1) return;

  const {re: tRe, im: tIm} = getTwiddles(n, invert);

  // Bit-reversal permutation (Standard)
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let tr = x_re[i]; x_re[i] = x_re[j]; x_re[j] = tr;
      let ti = x_im[i]; x_im[i] = x_im[j]; x_im[j] = ti;
    }
  }

  // Cooley-Tukey butterflies
  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const step = n / len;
    
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < halfLen; k++) {
        const tIdx = k * step;
        const w_re = tRe[tIdx];
        const w_im = tIm[tIdx];
        
        const idx = i + k + halfLen;
        const v_re = x_re[idx] * w_re - x_im[idx] * w_im;
        const v_im = x_re[idx] * w_im + x_im[idx] * w_re;

        x_re[idx] = x_re[i + k] - v_re;
        x_im[idx] = x_im[i + k] - v_im;
        x_re[i + k] += v_re;
        x_im[i + k] += v_im;
      }
    }
  }

  if (invert) {
    for (let i = 0; i < n; i++) {
      x_re[i] /= n;
      x_im[i] /= n;
    }
  }
}

// Reusable scratch buffers for 2D FFT to avoid GC pressure
const rowRe = new Float64Array(MAX_SIZE);
const rowIm = new Float64Array(MAX_SIZE);

export function fft2d(mat: ComplexArray, width: number, height: number, invert: boolean = false) {
  // Transpose-based 2D FFT is often faster, but we'll stick to Row-Column for now with zero allocations
  
  // Rows
  for (let y = 0; y < height; y++) {
    const offset = y * width;
    const sliceRe = mat.real.subarray(offset, offset + width);
    const sliceIm = mat.imag.subarray(offset, offset + width);
    fft1d(sliceRe, sliceIm, invert);
  }

  // Columns (This is the slowest part due to cache misses and manual copy)
  // To avoid allocation, we use the global rowRe/rowIm scratch buffers
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < height; y++) {
      rowRe[y] = mat.real[y * width + x];
      rowIm[y] = mat.imag[y * width + x];
    }
    
    fft1d(rowRe.subarray(0, height), rowIm.subarray(0, height), invert);
    
    for (let y = 0; y < height; y++) {
      mat.real[y * width + x] = rowRe[y];
      mat.imag[y * width + x] = rowIm[y];
    }
  }
}
