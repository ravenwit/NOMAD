import React, { useMemo, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import { WaveEngine } from './WaveEngine';

interface TorusVisualizerProps {
  R: number;
  r: number;
  radialSegments: number; // N_theta
  tubularSegments: number; // N_phi
  intensityMultiplier?: number;
}

const vertexShader = `
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewPosition;

  void main() {
    vUv = uv;
    // Compute normal and view position for fragment shader lighting
    vNormal = normalize(normalMatrix * normal);
    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    vViewPosition = -mvPosition.xyz;
    
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = `
  uniform float intensity;
  uniform sampler2D dataTexture;
  varying vec2 vUv;
  varying vec3 vNormal;
  varying vec3 vViewPosition;

  // Premium aesthetic palette
  vec3 colorZero = vec3(0.04, 0.05, 0.08); // Obsidian dark base
  vec3 colorPos  = vec3(1.0, 0.45, 0.35);  // Rose gold for positive peaks
  vec3 colorNeg  = vec3(0.1, 0.95, 0.7);   // Bioluminescent mint for negative troughs

  void main() {
    float wave = texture2D(dataTexture, vUv).r;
    float gain = 30.0;
    
    vec3 materialColor;
    if (wave > 0.0) {
        float t = clamp(wave * gain, 0.0, 1.0);
        materialColor = mix(colorZero, colorPos, smoothstep(0.0, 1.0, t));
    } else {
        float t = clamp(-wave * gain, 0.0, 1.0);
        materialColor = mix(colorZero, colorNeg, smoothstep(0.0, 1.0, t));
    }
    
    // Procedural Lighting setup
    vec3 normal = normalize(vNormal);
    vec3 viewDir = normalize(vViewPosition);

    vec3 lightDir = normalize(vec3(1.0, 1.5, 1.0)); // Top-right
    vec3 lightColor = vec3(1.0, 0.98, 0.95);
    vec3 ambient = vec3(0.2, 0.25, 0.3); // Soft ambient
    
    // Diffuse
    float diff = max(dot(normal, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;
    
    // Specular Highlight
    vec3 halfVector = normalize(lightDir + viewDir);
    float spec = pow(max(dot(normal, halfVector), 0.0), 64.0);
    vec3 specular = 0.6 * spec * lightColor;
    
    // Rim Light
    float rimDot = 1.0 - max(dot(viewDir, normal), 0.0);
    float rimAmount = smoothstep(0.6, 1.0, rimDot);
    vec3 rimLight = vec3(0.4, 0.5, 0.8) * rimAmount * 0.4;

    vec3 finalColor = materialColor * (ambient + diffuse) + specular + rimLight;
    
    gl_FragColor = vec4(finalColor * intensity, 1.0);
  }
`;

export const TorusVisualizer: React.FC<TorusVisualizerProps> = ({ 
  R, r, radialSegments, tubularSegments, intensityMultiplier = 1.5 
}) => {
  const materialRef = useRef<THREE.ShaderMaterial>(null);
  
  // Interaction states for WaveEngine
  const [pointerUV, setPointerUV] = useState<THREE.Vector2 | null>(null);
  const [isPointerDown, setIsPointerDown] = useState(false);
  const [activeTexture, setActiveTexture] = useState<THREE.Texture | null>(null);

  const uniforms = useMemo(() => ({
    dataTexture: { value: null as THREE.Texture | null },
    intensity: { value: intensityMultiplier },
  }), [intensityMultiplier]);

  // Update uniforms without forcing a react re-render when texture arrives
  const handleTextureReady = useCallback((tex: THREE.Texture) => {
    setActiveTexture(tex);
    if (materialRef.current) {
        materialRef.current.uniforms.dataTexture.value = tex;
    }
  }, []);


  return (
    <>
      <WaveEngine 
        onOutputTextureReady={handleTextureReady}
        pointerUV={pointerUV}
        isPointerDown={isPointerDown}
        R={R}
        r={r}
      />
      
      <mesh 
        castShadow 
        receiveShadow
        onPointerDown={(e) => {
          e.stopPropagation();
          setIsPointerDown(true);
          if (e.uv) setPointerUV(e.uv);
        }}
        onPointerUp={() => setIsPointerDown(false)}
        onPointerLeave={() => setIsPointerDown(false)}
        onPointerMove={(e) => {
          if (isPointerDown && e.uv) setPointerUV(e.uv);
        }}
      >
        <torusGeometry args={[R, r, radialSegments, tubularSegments]} />
        {activeTexture && (
          <shaderMaterial
            ref={materialRef}
            vertexShader={vertexShader}
            fragmentShader={fragmentShader}
            uniforms={uniforms}
            wireframe={false}
          />
        )}
      </mesh>
    </>
  );
};
