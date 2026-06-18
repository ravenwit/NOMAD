import React, { useMemo, useRef, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { HuggingFaceInference } from '../numerical/HuggingFaceInference';

const W = 256; // Grid resolution
const H = 256;

const waveFragmentShader = `
uniform sampler2D texCurr;
uniform sampler2D texPrev;
uniform sampler2D texSource;

uniform vec2 texelSize;
uniform float dt;
uniform float c;
uniform float R;
uniform float r;

varying vec2 vUv;

const float PI = 3.1415926535897932384626433832795;

float get_sqrt_g(float theta) {
    return r * (R + r * cos(theta));
}

float get_A(float theta) {
    return (R + r * cos(theta)) / r;
}

float get_B(float theta) {
    return r / (R + r * cos(theta));
}

void main() {
    float phi = vUv.x * 2.0 * PI;
    float theta = vUv.y * 2.0 * PI;

    float dx = 2.0 * PI * texelSize.x;
    float dy = 2.0 * PI * texelSize.y;

    float u_c = texture2D(texCurr, vUv).r;
    float u_p = texture2D(texPrev, vUv).r;

    float u_right = texture2D(texCurr, fract(vUv + vec2(texelSize.x, 0.0))).r;
    float u_left  = texture2D(texCurr, fract(vUv - vec2(texelSize.x, 0.0))).r;
    float u_up    = texture2D(texCurr, fract(vUv + vec2(0.0, texelSize.y))).r;
    float u_down  = texture2D(texCurr, fract(vUv - vec2(0.0, texelSize.y))).r;

    float theta_j = theta;
    float theta_j_plus_half = theta + dy * 0.5;
    float theta_j_minus_half = theta - dy * 0.5;

    float A_plus  = get_A(theta_j_plus_half);
    float A_minus = get_A(theta_j_minus_half);
    float B_j     = get_B(theta_j);

    float d_theta_term = (A_plus * (u_up - u_c) - A_minus * (u_c - u_down)) / (dy * dy);
    float d_phi_term   = B_j * (u_right - 2.0 * u_c + u_left) / (dx * dx);

    float laplacian = (1.0 / get_sqrt_g(theta_j)) * (d_theta_term + d_phi_term);

    float source = texture2D(texSource, vUv).r;
    float next_u = (2.0 * u_c) - u_p + (dt * dt * ((c * c * laplacian) + source));

    gl_FragColor = vec4(next_u, 0.0, 0.0, 1.0);
}
`;

const quadVertexShader = `
varying vec2 vUv;
void main() {
    vUv = uv;
    gl_Position = vec4(position, 1.0);
}
`;

export type SolverMode = 'webgl' | 'local_spectral' | 'remote_spectral' | 'huggingface';

export interface WaveEngineProps {
  onOutputTextureReady: (tex: THREE.Texture) => void;
  pointerUV: THREE.Vector2 | null;
  isPointerDown: boolean;
  R?: number;
  r?: number;
  solverMode?: SolverMode;
  resetTrigger?: number;
  onInferenceTime?: (ms: number) => void;
}

export const WaveEngine: React.FC<WaveEngineProps> = ({
  onOutputTextureReady,
  pointerUV,
  isPointerDown,
  R = 1.5,
  r = 0.5,
  solverMode = 'webgl',
  resetTrigger = 0,
  onInferenceTime
}) => {

  const { gl } = useThree();

  const targets = useMemo(() => {
    const options = {
      type: THREE.FloatType,
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      wrapS: THREE.RepeatWrapping,
      wrapT: THREE.RepeatWrapping,
      format: THREE.RedFormat,
      depthBuffer: false,
    };
    return [
      new THREE.WebGLRenderTarget(W, H, options),
      new THREE.WebGLRenderTarget(W, H, options),
      new THREE.WebGLRenderTarget(W, H, options),
    ];
  }, []);

  const rtPointers = useRef({
    prev: targets[0],
    curr: targets[1],
    next: targets[2],
  });

  const sourceTextureRef = useRef<THREE.DataTexture | null>(null);
  if (!sourceTextureRef.current) {
    const tex = new THREE.DataTexture(new Float32Array(W * H), W, H, THREE.RedFormat, THREE.FloatType);
    tex.needsUpdate = true;
    sourceTextureRef.current = tex;
  }
  const sourceTexture = sourceTextureRef.current;

  const scene = useMemo(() => new THREE.Scene(), []);
  const camera = useMemo(() => new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1), []);
  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      vertexShader: quadVertexShader,
      fragmentShader: waveFragmentShader,
      uniforms: {
        texCurr: { value: null },
        texPrev: { value: null },
        texSource: { value: sourceTexture },
        texelSize: { value: new THREE.Vector2(1.0 / W, 1.0 / H) },
        dt: { value: 0.01 },
        c: { value: 1.0 },
        R: { value: R },
        r: { value: r },
      },
    });
  }, [R, r, sourceTexture]);

  useEffect(() => {
    const geometry = new THREE.PlaneGeometry(2, 2);
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
    return () => {
      scene.remove(mesh);
      geometry.dispose();
    };
  }, [material, scene]);

  const wasPointerDown = useRef(false);
  const fetching = useRef(false);
  const remoteClickQueue = useRef<{theta: number, phi: number} | null>(null);

  const hfRef = useRef<HuggingFaceInference | null>(null);
  const workerRef = useRef<Worker | null>(null);
  const workerBusy = useRef(false);

  useEffect(() => {
    const worker = new Worker(new URL('../numerical/WaveWorker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    worker.postMessage({ type: 'init', R, r, width: W, height: H, c: 1.0, CFL: 0.1 });
    worker.onmessage = (e) => {
        if (e.data.type === 'frame') {
            const data = e.data.data as Float32Array;
            const hostData = hostTexture.image.data as unknown as Float32Array;
            hostData.set(data);
            hostTexture.needsUpdate = true;
            workerBusy.current = false;
        }
    };
    return () => worker.terminate();
  }, [R, r]);

  useEffect(() => {
    const hf = new HuggingFaceInference();
    hfRef.current = hf;
    hf.init().catch(e => console.warn('[WaveEngine] HF init failed:', e));
  }, []);

  useEffect(() => {
    if (resetTrigger > 0) {
      if (solverMode === 'webgl') {
        const { prev, curr, next } = rtPointers.current;
        const currentTarget = gl.getRenderTarget();
        [prev, curr, next].forEach(rt => {
          gl.setRenderTarget(rt);
          gl.clear();
        });
        gl.setRenderTarget(currentTarget);
      } else if (solverMode === 'local_spectral') {
        workerRef.current?.postMessage({ type: 'reset' });
      } else if (solverMode === 'remote_spectral') {
        fetch('http://localhost:8000/reset').catch(() => {});
      } else if (solverMode === 'huggingface') {
        hfRef.current?.reset();
      }
    }
  }, [resetTrigger, solverMode, gl, targets]);
  
  const hostTextureRef = useRef<THREE.DataTexture | null>(null);
  if (!hostTextureRef.current) {
    const tex = new THREE.DataTexture(new Float32Array(W * H), W, H, THREE.RedFormat, THREE.FloatType);
    tex.needsUpdate = true;
    hostTextureRef.current = tex;
  }
  const hostTexture = hostTextureRef.current;

  const syncWithPython = async () => {
    if (fetching.current) return;
    fetching.current = true;
    try {
      let url = `http://localhost:8000/step?steps=10`;
      if (remoteClickQueue.current) {
        url += `&theta0=${remoteClickQueue.current.theta}&phi0=${remoteClickQueue.current.phi}`;
        remoteClickQueue.current = null;
      }
      const resp = await fetch(url);
      const buffer = await resp.arrayBuffer();
      const data = new Float32Array(buffer);
      
      const hostData = hostTexture.image.data as unknown as Float32Array;
      hostData.set(data);
      hostTexture.needsUpdate = true;
    } catch (e) {
    } finally {
      fetching.current = false;
    }
  };

  useEffect(() => {
    return () => {
      targets.forEach(t => t.dispose());
      sourceTexture.dispose();
      hostTexture.dispose();
      material.dispose();
    };
  }, [targets, sourceTexture, hostTexture, material]);

  useFrame(() => {
    const { prev, curr, next } = rtPointers.current;

    if (solverMode === 'huggingface') {
        const hf = hfRef.current;
        if (hf && hf.isReady()) {
            if (isPointerDown && !wasPointerDown.current && pointerUV) {
                hf.injectPulse(pointerUV.y, pointerUV.x).catch(e => console.error(e));
            }
            const frameData = hf.tick();
            if (frameData) {
                const hostData = hostTexture.image.data as unknown as Float32Array;
                hostData.set(frameData);
                hostTexture.needsUpdate = true;
            }
            if (onInferenceTime) onInferenceTime(hf.getLastInferenceMs());
        }
        onOutputTextureReady(hostTexture);
        wasPointerDown.current = isPointerDown;
        return;
    }

    if (solverMode === 'remote_spectral') {
        if (isPointerDown && !wasPointerDown.current && pointerUV) {
            remoteClickQueue.current = { theta: pointerUV.y * 2 * Math.PI, phi: pointerUV.x * 2 * Math.PI };
        }
        if (!fetching.current) syncWithPython();
        onOutputTextureReady(hostTexture);
        wasPointerDown.current = isPointerDown;
        return;
    }

    if (solverMode === 'local_spectral') {
        if (isPointerDown && !wasPointerDown.current && pointerUV) {
            workerRef.current?.postMessage({
                type: 'pulse', theta0: pointerUV.y * 2 * Math.PI, phi0: pointerUV.x * 2 * Math.PI, impulse: 10000.0
            });
        }
        if (workerRef.current && !workerBusy.current) {
            workerBusy.current = true;
            workerRef.current.postMessage({ type: 'step', steps: 1 });
        }
        onOutputTextureReady(hostTexture);
        wasPointerDown.current = isPointerDown;
        return;
    }

    // WebGL Default
    if (isPointerDown || wasPointerDown.current) {
      const srcData = sourceTexture.image.data as unknown as Float32Array;
      srcData.fill(0.0);
      if (isPointerDown && pointerUV) {
        const u_idx = Math.floor(pointerUV.x * W);
        const v_idx = Math.floor(pointerUV.y * H);
        const radius = 6, sigma = 2.0, impulse = 10000.0;
        let sum = 0.0;
        const weights = [];
        for (let j = -radius; j <= radius; j++) {
          for (let i = -radius; i <= radius; i++) {
            const distSq = i * i + j * j;
            if (distSq <= radius * radius) {
              const r_sq_over_sigma_sq = distSq / (sigma * sigma);
              const value = (2.0 - r_sq_over_sigma_sq) * Math.exp(-distSq / (2.0 * sigma * sigma));
              weights.push({ i, j, value });
              sum += value;
            }
          }
        }
        const correction = sum / weights.length;
        for (const w of weights) {
          const u_curr = (u_idx + w.i + W) % W;
          const v_curr = (v_idx + w.j + H) % H;
          srcData[v_curr * W + u_curr] += (w.value - correction) * impulse;
        }
      }
      sourceTexture.needsUpdate = true;
    }
    wasPointerDown.current = isPointerDown;

    material.uniforms.texCurr.value = curr.texture;
    material.uniforms.texPrev.value = prev.texture;

    gl.setRenderTarget(next);
    gl.clear();
    gl.render(scene, camera);
    gl.setRenderTarget(null);

    onOutputTextureReady(next.texture);

    rtPointers.current = { prev: curr, curr: next, next: prev };
  });

  return null;
};
