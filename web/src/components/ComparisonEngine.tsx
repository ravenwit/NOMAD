import React, { useMemo, useRef, useState, useCallback, useEffect } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import { WaveEngine, SolverMode } from './WaveEngine';

// Import shaders
import { vertexShader, fragmentShader } from './TorusVisualizer';

const SingleTorus: React.FC<{
  position: [number, number, number],
  texture: THREE.Texture | null,
  R: number, r: number,
  onInteract?: (uv: THREE.Vector2) => void,
  onPointerDown?: () => void,
  onPointerUp?: () => void
}> = ({ position, texture, R, r, onInteract, onPointerDown, onPointerUp }) => {
    const materialRef = useRef<THREE.ShaderMaterial>(null);

    const uniforms = useMemo(() => ({
        dataTexture: { value: null as THREE.Texture | null },
        intensity: { value: 1.5 },
    }), []);

    useEffect(() => {
        if (materialRef.current && texture) {
            materialRef.current.uniforms.dataTexture.value = texture;
        }
    }, [texture]);

    return (
        <group position={position}>
            <mesh 
                castShadow receiveShadow
                onPointerDown={(e) => {
                    e.stopPropagation();
                    if (onPointerDown) onPointerDown();
                    if (e.uv && onInteract) onInteract(e.uv);
                }}
                onPointerUp={() => {
                    if (onPointerUp) onPointerUp();
                }}
                onPointerLeave={() => {
                    if (onPointerUp) onPointerUp();
                }}
                onPointerMove={(e) => {
                    if (e.buttons > 0 && e.uv && onInteract) {
                        onInteract(e.uv);
                    }
                }}
            >
                <torusGeometry args={[R, r, 128, 128]} />
                {texture && (
                    <shaderMaterial
                        ref={materialRef}
                        vertexShader={vertexShader}
                        fragmentShader={fragmentShader}
                        uniforms={uniforms}
                    />
                )}
            </mesh>
        </group>
    );
};

export const ComparisonEngine: React.FC<{
  solverModeA: SolverMode,
  solverModeB: SolverMode,
  titleA: string,
  subtitleA: string,
  titleB: string,
  subtitleB: string,
  resetTrigger: number,
}> = ({ solverModeA, solverModeB, titleA, subtitleA, titleB, subtitleB, resetTrigger }) => {
    
    const [texA, setTexA] = useState<THREE.Texture | null>(null);
    const [texB, setTexB] = useState<THREE.Texture | null>(null);

    const [pointerUV, setPointerUV] = useState<THREE.Vector2 | null>(null);
    const [isPointerDown, setIsPointerDown] = useState(false);
    
    const [timeA, setTimeA] = useState(0);
    const [timeB, setTimeB] = useState(0);

    const handleInteract = useCallback((uv: THREE.Vector2) => {
        setPointerUV(uv.clone());
    }, []);

    const R = 1.5;
    const r = 0.5;

    return (
        <div className="w-full h-full relative">
            {/* Main Canvas for Side-by-Side Simulation */}
            <Canvas camera={{ position: [0, 0, 6], fov: 45 }}>
                <ambientLight intensity={0.2} />
                <pointLight position={[10, 10, 10]} intensity={1} />
                <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
                
                {/* Hidden simulation engines */}
                <WaveEngine 
                    solverMode={solverModeA} 
                    onOutputTextureReady={setTexA} 
                    pointerUV={pointerUV}
                    isPointerDown={isPointerDown}
                    resetTrigger={resetTrigger}
                    onInferenceTime={setTimeA}
                />
                <WaveEngine 
                    solverMode={solverModeB} 
                    onOutputTextureReady={setTexB} 
                    pointerUV={pointerUV}
                    isPointerDown={isPointerDown}
                    resetTrigger={resetTrigger}
                    onInferenceTime={setTimeB}
                />
                
                <SingleTorus 
                    position={[-2.2, 0, 0]} 
                    texture={texA} 
                    R={R} r={r} 
                    onInteract={handleInteract}
                    onPointerDown={() => setIsPointerDown(true)}
                    onPointerUp={() => setIsPointerDown(false)}
                />
                
                <SingleTorus 
                    position={[2.2, 0, 0]} 
                    texture={texB} 
                    R={R} r={r} 
                    onInteract={handleInteract}
                    onPointerDown={() => setIsPointerDown(true)}
                    onPointerUp={() => setIsPointerDown(false)}
                />
                
                <OrbitControls enablePan={false} maxDistance={12} minDistance={3} autoRotate={false} />
            </Canvas>

            {/* Overlay Labels */}
            <div className="absolute top-8 left-[20%] -translate-x-1/2 pointer-events-none">
                <div className="bg-black/40 backdrop-blur border border-zinc-800 rounded-lg px-4 py-2 text-center shadow-2xl">
                    <h3 className="text-white font-semibold text-lg drop-shadow-md">{titleA}</h3>
                    <p className="text-zinc-400 text-xs mt-1">{subtitleA}</p>
                    {timeA > 0 && <p className="text-cyan-400 text-xs mt-1">{timeA.toFixed(1)}ms</p>}
                </div>
            </div>

            <div className="absolute top-8 left-[80%] -translate-x-1/2 pointer-events-none">
                <div className="bg-black/40 backdrop-blur border border-zinc-800 rounded-lg px-4 py-2 text-center shadow-2xl">
                    <h3 className="text-white font-semibold text-lg drop-shadow-md">{titleB}</h3>
                    <p className="text-orange-400 text-xs mt-1">{subtitleB}</p>
                    {timeB > 0 && <p className="text-orange-400 text-xs mt-1">{timeB.toFixed(1)}ms</p>}
                </div>
            </div>
        </div>
    );
};
