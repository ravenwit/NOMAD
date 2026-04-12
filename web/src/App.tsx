import { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars, Stats } from '@react-three/drei';
import { TorusVisualizer } from './components/TorusVisualizer';

function App() {
  const [isRemote, setIsRemote] = useState(false);

  return (
    <div className="w-screen h-screen bg-black flex flex-col font-sans">
      <div className="absolute top-0 left-0 w-full p-6 z-10 pointer-events-none text-white">
        <h1 className="text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-300">
          Acoustic Wave Topology
        </h1>
        <p className="opacity-80 text-lg mt-2 max-w-xl">
          Real-time visualization of the 2D Acoustic Wave Equation explicitly mapped onto a Toroidal manifold using a custom Laplace-Beltrami geometric shader.
        </p>
        
        <div className="mt-8 pointer-events-auto flex items-center gap-3 bg-zinc-900/50 backdrop-blur-md p-3 rounded-lg border border-zinc-700 w-fit">
          <label className="relative inline-flex items-center cursor-pointer">
            <input 
              type="checkbox" 
              className="sr-only peer" 
              checked={isRemote}
              onChange={() => setIsRemote(!isRemote)}
            />
            <div className="w-11 h-6 bg-zinc-700 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-zinc-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            <span className="ml-3 text-sm font-medium text-zinc-200">
              Remote Spectral Solver (Python)
            </span>
          </label>
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
            isRemote={isRemote}
          />
          
          <OrbitControls 
            enablePan={false}
            autoRotate={!isRemote}
            autoRotateSpeed={0.5}
            maxDistance={10}
            minDistance={2}
          />
          <Stats />
        </Canvas>
      </div>
      
      <div className="p-4 bg-zinc-900/80 backdrop-blur border-t border-zinc-800 flex justify-between items-center text-sm text-zinc-400">
        <div>
          <span className="font-semibold text-zinc-200">Phase 1:</span> {isRemote ? 'G-Deep Spectral (Python)' : 'WebGL-Harmonized FDTD'}
        </div>
        <div className="flex gap-4">
          <span>Grid: 128x128</span>
          <span>Integrator: {isRemote ? 'Python-Spectral' : 'Shader Harmonized'}</span>
          <span>Metric: g_tt, g_pp</span>
        </div>
      </div>
    </div>
  );
}

export default App;
