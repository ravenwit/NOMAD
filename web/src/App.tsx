import { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars, Stats } from '@react-three/drei';
import { TorusVisualizer } from './components/TorusVisualizer';
import { SolverMode } from './components/WaveEngine';
import { PerformanceBadge } from './components/PerformanceBadge';

function App() {
  const [solverMode, setSolverMode] = useState<SolverMode>('webgl');
  const [resetTrigger, setResetTrigger] = useState(0);
  const [inferenceTime, setInferenceTime] = useState<number>(0);

  return (
    <div className="w-screen h-screen bg-black flex flex-col font-sans">
      <div className="absolute top-0 left-0 w-full p-6 z-10 pointer-events-none text-white">
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-300">
          Acoustic Wave Topology
        </h1>
        <p className="opacity-80 text-lg mt-2 max-w-xl">
          Real-time visualization of the 2D Acoustic Wave Equation explicitly mapped onto a Toroidal manifold using a custom Laplace-Beltrami geometric shader.
        </p>
        
        <div className="mt-8 pointer-events-auto flex gap-2 items-center">
          <div className="flex gap-2 bg-zinc-900/50 backdrop-blur-md p-1 rounded-lg border border-zinc-800 w-fit">
            <button 
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${solverMode === 'webgl' ? 'bg-blue-600 text-white shadow-lg' : 'text-zinc-400 hover:text-white hover:bg-zinc-800/50'}`}
              onClick={() => setSolverMode('webgl')}
            >
              FDTD (WebGL)
            </button>
            <button 
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${solverMode === 'local_spectral' ? 'bg-blue-600 text-white shadow-lg' : 'text-zinc-400 hover:text-white hover:bg-zinc-800/50'}`}
              onClick={() => setSolverMode('local_spectral')}
            >
              Spectral (JS Native)
            </button>
            <button 
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${solverMode === 'remote_spectral' ? 'bg-blue-600 text-white shadow-lg' : 'text-zinc-400 hover:text-white hover:bg-zinc-800/50'}`}
              onClick={() => setSolverMode('remote_spectral')}
            >
              Spectral (Python GPU)
            </button>
            <button 
              className={`px-4 py-2 rounded-md text-sm font-medium transition-all ${solverMode === 'neural_operator' ? 'bg-violet-600 text-white shadow-lg shadow-violet-600/30' : 'text-zinc-400 hover:text-white hover:bg-zinc-800/50'}`}
              onClick={() => setSolverMode('neural_operator')}
            >
              Neural FNO (ONNX)
            </button>
          </div>
          
          <button 
            className="px-4 py-2 bg-rose-600/20 hover:bg-rose-600/40 text-rose-300 border border-rose-500/30 rounded-lg backdrop-blur-md text-sm font-medium transition-all shadow-lg ml-4"
            onClick={() => setResetTrigger(r => r + 1)}
          >
            Reset Field
          </button>
        </div>
      </div>

      <div className="flex-1 relative">
        <Canvas camera={{ position: [0, -3, 4], fov: 45 }}>
          <ambientLight intensity={0.2} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
          
          <TorusVisualizer 
            R={1.5} r={0.5} 
            radialSegments={128} tubularSegments={128} 
            solverMode={solverMode}
            resetTrigger={resetTrigger}
            onInferenceTime={setInferenceTime}
          />
          
          <OrbitControls 
            enablePan={false}
            autoRotate={solverMode === 'webgl'}
            autoRotateSpeed={0.5}
            maxDistance={10}
            minDistance={2}
          />
          <Stats />
        </Canvas>
        <PerformanceBadge inferenceMs={inferenceTime} visible={solverMode === 'neural_operator'} />
      </div>
      
      <div className="p-4 bg-zinc-900/80 backdrop-blur border-t border-zinc-800 flex justify-between items-center text-sm text-zinc-400">
        <div>
          <span className="font-semibold text-zinc-200">Mode:</span> {solverMode === 'webgl' ? 'WebGL-Harmonized FDTD' : solverMode === 'local_spectral' ? 'TypeScript Spectral (FFT)' : solverMode === 'remote_spectral' ? 'Python GPU Spectral (FFT)' : 'Neural Operator PeriodicUNet (ONNX Runtime Web)'}
        </div>
        <div className="flex gap-4">
          <span>Grid: {solverMode === 'neural_operator' ? '64×64 → 256×256 (upsampled)' : '256x256'}</span>
          <span>Integrator: {solverMode === 'webgl' ? 'Shader Harmonized Leapfrog' : solverMode === 'local_spectral' ? 'JS Native Spectral Math' : solverMode === 'remote_spectral' ? 'Torch Accelerated Spectral' : 'Autoregressive Neural Rollout'}</span>
          <span>Metric: g_tt, g_pp</span>
        </div>
      </div>
    </div>
  );
}

export default App;
