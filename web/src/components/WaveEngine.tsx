import React, { useMemo, useRef, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';

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

// g \equiv det(g), here sqrt(g) = r(R + r * cos(theta))
float get_sqrt_g(float theta) {
    return r * (R + r * cos(theta));
}

// A(theta) = sqrt(g) * g^{theta, theta} = r(R + r * cos(theta)) / r^2 = (R + r * cos(theta)) / r
float get_A(float theta) {
    return (R + r * cos(theta)) / r;
}

// B(theta) = sqrt(g) * g^{phi, phi} = r(R + r * cos(theta)) / (R + r * cos(theta))^2 = r / (R + r * cos(theta))
float get_B(float theta) {
    return r / (R + r * cos(theta));
}

void main() {
    float phi = vUv.x * 2.0 * PI;
    float theta = vUv.y * 2.0 * PI;

    float dx = 2.0 * PI * texelSize.x;
    float dy = 2.0 * PI * texelSize.y;

    // Current and previous values (only reading Red channel assuming Grayscale/Float textures)
    float u_c = texture2D(texCurr, vUv).r;
    float u_p = texture2D(texPrev, vUv).r;

    // Periodic Boundary Conditions using fract() for UV space wrapping
    float u_right = texture2D(texCurr, fract(vUv + vec2(texelSize.x, 0.0))).r;
    float u_left  = texture2D(texCurr, fract(vUv - vec2(texelSize.x, 0.0))).r;
    float u_up    = texture2D(texCurr, fract(vUv + vec2(0.0, texelSize.y))).r;
    float u_down  = texture2D(texCurr, fract(vUv - vec2(0.0, texelSize.y))).r;

    // Evaluate coefficients at staggered grid points for stability
    float theta_j = theta;
    float theta_j_plus_half = theta + dy * 0.5;
    float theta_j_minus_half = theta - dy * 0.5;

    float A_plus  = get_A(theta_j_plus_half);
    float A_minus = get_A(theta_j_minus_half);
    float B_j     = get_B(theta_j);

    float d_theta_term = (A_plus * (u_up - u_c) - A_minus * (u_c - u_down)) / (dy * dy);
    float d_phi_term   = B_j * (u_right - 2.0 * u_c + u_left) / (dx * dx);

    float laplacian = (1.0 / get_sqrt_g(theta_j)) * (d_theta_term + d_phi_term);

    // Explicit FDTD Update Step (pure lossless wave equation)
    // u^{n+1} = 2u^n - u^{n-1} + dt^2 * (c^2 * laplacian + S)
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

export interface WaveEngineProps {
  onOutputTextureReady: (tex: THREE.Texture) => void;
  pointerUV: THREE.Vector2 | null;
  isPointerDown: boolean;
  R?: number;
  r?: number;
  isRemote?: boolean;
}

export const WaveEngine: React.FC<WaveEngineProps> = ({
  onOutputTextureReady,
  pointerUV,
  isPointerDown,
  R = 1.5,
  r = 0.5,
  isRemote = false
}) => {
  const { gl } = useThree();

  // Create ping-pong FBOs (with internal float format for math limits)
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

  // Source Texture corresponding to physical "taps"
  const sourceTexture = useMemo(() => {
    const tex = new THREE.DataTexture(new Float32Array(W * H), W, H, THREE.RedFormat, THREE.FloatType);
    tex.needsUpdate = true;
    return tex;
  }, []);

  // Compute Scene elements
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
        dt: { value: 0.01 }, // Time step
        c: { value: 1.0 },   // True theoretical wave speed
        R: { value: R },
        r: { value: r },
      },
    });
  }, [R, r, sourceTexture]);

  useEffect(() => {
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material);
    scene.add(mesh);
    return () => {
      scene.remove(mesh);
    };
  }, [material, scene]);

  const wasPointerDown = useRef(false);
  const fetching = useRef(false);
  const remoteClickQueue = useRef<{theta: number, phi: number} | null>(null);

  // Dedicated data texture to hold the frames from Python
  const pythonTexture = useMemo(() => {
    const tex = new THREE.DataTexture(new Float32Array(W * H), W, H, THREE.RedFormat, THREE.FloatType);
    tex.needsUpdate = true;
    return tex;
  }, []);

  // Function to sync state from Python Spectral Solver
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
      
      const pyData = pythonTexture.image.data as Float32Array;
      pyData.set(data);
      pythonTexture.needsUpdate = true;
    } catch (e) {
      console.error("Failed to sync with Python solver:", e);
    } finally {
      fetching.current = false;
    }
  };


  // Execute Simulation Step on every frame
  useFrame(() => {
    const { prev, curr, next } = rtPointers.current;

    // IF REMOTE: We continuously poll Python to provide the field state
    if (isRemote) {
        if (isPointerDown && !wasPointerDown.current && pointerUV) {
            remoteClickQueue.current = {
                theta: pointerUV.y * 2 * Math.PI,
                phi: pointerUV.x * 2 * Math.PI
            };
        }
        if (!fetching.current) {
            syncWithPython();
        }
        onOutputTextureReady(pythonTexture);
        wasPointerDown.current = isPointerDown;
        return;
    }

    // 1. Update Source Term (Raycast hit coordinates)
    if (isPointerDown || wasPointerDown.current) {
      const srcData = sourceTexture.image.data as unknown as Float32Array;
      srcData.fill(0.0); // Clear source every frame

      if (isPointerDown && pointerUV) {
        // Create a zero-mean Mexican Hat (Ricker wavelet) pulse 
        const u_idx = Math.floor(pointerUV.x * W);
        const v_idx = Math.floor(pointerUV.y * H);
        const radius = 6;
        const sigma = 2.0;
        const impulse = 10000.0;

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

    // 2. Setup uniform state for Shader computation
    material.uniforms.texCurr.value = curr.texture;
    material.uniforms.texPrev.value = prev.texture;

    // 3. Render next step to 'next' render target
    gl.setRenderTarget(next);
    gl.clear();
    gl.render(scene, camera);
    gl.setRenderTarget(null); // Restore default

    // 4. Send output back to the Visualizer component
    onOutputTextureReady(next.texture);

    // 5. Swap pointers
    rtPointers.current = {
      prev: curr,
      curr: next,
      next: prev,
    };
  });

  return null;
};
