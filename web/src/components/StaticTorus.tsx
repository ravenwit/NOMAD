import React, { useRef, useMemo, useEffect } from 'react';
import * as THREE from 'three';
import { useFrame } from '@react-three/fiber';
import { vertexShader, fragmentShader } from './TorusVisualizer';

interface StaticTorusProps {
  R: number;
  r: number;
  radialSegments: number;
  tubularSegments: number;
  frameData: Float32Array | null;
  res: number; // e.g. 128 or 256
  colorMap?: 'default' | 'error';
  intensityMultiplier?: number;
}

export const StaticTorus: React.FC<StaticTorusProps> = ({ 
  R, r, radialSegments, tubularSegments, frameData, res, colorMap = 'default', intensityMultiplier = 1.5 
}) => {
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  
  // Create a persistent DataTexture
  const textureRef = useRef<THREE.DataTexture | null>(null);
  if (!textureRef.current) {
      const tex = new THREE.DataTexture(new Float32Array(res * res), res, res, THREE.RedFormat, THREE.FloatType);
      tex.needsUpdate = true;
      textureRef.current = tex;
  }
  const texture = textureRef.current;

  // Re-allocate if resolution changes
  useEffect(() => {
      if (texture.image.width !== res) {
          texture.image.width = res;
          texture.image.height = res;
          texture.image.data = new Float32Array(res * res) as any;
          texture.needsUpdate = true;
      }
  }, [res, texture]);

  const uniforms = useMemo(() => ({
    dataTexture: { value: texture },
    intensity: { value: intensityMultiplier },
    gain: { value: 1.0 }, // We will update this dynamically
    isErrorMode: { value: colorMap === 'error' ? 1.0 : 0.0 }
  }), [texture, intensityMultiplier, colorMap]);

  // Update texture data when frameData changes
  useFrame(() => {
      if (frameData && frameData.length === res * res) {
          const hostData = texture.image.data as unknown as Float32Array;
          hostData.set(frameData);
          texture.needsUpdate = true;

          // Auto-calculate dynamic gain based on max absolute value
          let maxAbs = 0;
          for (let i = 0; i < frameData.length; i++) {
              const absVal = Math.abs(frameData[i]);
              if (absVal > maxAbs) maxAbs = absVal;
          }
          if (maxAbs > 0) {
              uniforms.gain.value = 1.0 / maxAbs; // Normalize colors to the current frame's max amplitude
          }
      }
  });

  return (
    <mesh castShadow receiveShadow>
      <torusGeometry args={[R, r, radialSegments, tubularSegments]} />
      <shaderMaterial
        ref={materialRef}
        vertexShader={vertexShader}
        fragmentShader={
            // Inject error mode color logic into fragment shader
            'uniform float isErrorMode;\nuniform float gain;\n' +
            `
            vec3 heatmap(float t) {
                // Brighter c0 so the torus is never completely invisible (empty)
                vec3 c0 = vec3(0.05, 0.05, 0.15); 
                vec3 c1 = vec3(0.8, 0.1, 0.2);
                vec3 c2 = vec3(1.0, 0.8, 0.1);
                vec3 c3 = vec3(1.0, 1.0, 1.0);
                
                if (t < 0.33) return mix(c0, c1, t * 3.0);
                if (t < 0.66) return mix(c1, c2, (t - 0.33) * 3.0);
                return mix(c2, c3, (t - 0.66) * 3.0);
            }
            ` +
            fragmentShader.replace(
                'float gain = 30.0;',
                '' // Remove hardcoded gain, use the uniform
            ).replace(
                'gl_FragColor = vec4(finalColor * intensity, 1.0);',
                `
                if (isErrorMode > 0.5) {
                    float t = clamp(abs(wave) * gain, 0.0, 1.0);
                    vec3 errColor = heatmap(t);
                    vec3 finalErr = errColor * (ambient + diffuse) + specular + rimLight;
                    gl_FragColor = vec4(finalErr * intensity, 1.0);
                } else {
                    gl_FragColor = vec4(finalColor * intensity, 1.0);
                }
                `
            )
        }
        uniforms={uniforms}
        wireframe={false}
      />
    </mesh>
  );
};
