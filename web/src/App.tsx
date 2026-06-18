import { useState } from 'react';
import { ComparisonEngine } from './components/ComparisonEngine';
import { ChaosAnalysis } from './components/ChaosAnalysis';

function App() {
  const [currentView, setCurrentView] = useState<'simulation' | 'huggingface' | 'analysis'>('simulation');
  const [resetTrigger, setResetTrigger] = useState(0);

  return (
    <div className="w-screen h-screen bg-black flex flex-col font-sans">
      {/* Top App Navigation */}
      <div className="w-full bg-zinc-950/90 backdrop-blur border-b border-zinc-800 flex items-center justify-between px-6 py-3 z-50">
        <div className="flex items-center gap-4">
            <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-300 mr-4 tracking-tight">
              Acoustic Topology
            </h1>
            <div className="flex bg-zinc-900 rounded-lg p-1 border border-zinc-800">
                <button 
                    onClick={() => setCurrentView('simulation')}
                    className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${currentView === 'simulation' ? 'bg-zinc-800 text-white shadow' : 'text-zinc-500 hover:text-zinc-300'}`}
                >
                    Simulation Dashboard
                </button>
                <button 
                    onClick={() => setCurrentView('huggingface')}
                    className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${currentView === 'huggingface' ? 'bg-zinc-800 text-white shadow' : 'text-zinc-500 hover:text-zinc-300'}`}
                >
                    HuggingFace API
                </button>
                <button 
                    onClick={() => setCurrentView('analysis')}
                    className={`px-4 py-1.5 rounded-md text-sm font-medium transition-colors ${currentView === 'analysis' ? 'bg-zinc-800 text-white shadow' : 'text-zinc-500 hover:text-zinc-300'}`}
                >
                    Geo FNO Comparison
                </button>
            </div>
        </div>
        <div className="text-xs text-zinc-500 font-mono">NOMAD Physics Engine v2.1</div>
      </div>

      {currentView === 'analysis' ? (
          <ChaosAnalysis />
      ) : (
        <div className="flex-1 relative flex flex-col">
          <div className="absolute top-0 left-0 w-full p-6 z-10 pointer-events-none text-white">
            <h2 className="text-3xl font-bold mb-2">
              {currentView === 'simulation' ? 'FDTD vs Spectral Simulation' : 'Simulation vs AI Model'}
            </h2>
            <p className="opacity-80 text-sm max-w-xl">
              {currentView === 'simulation' 
                ? 'Real-time synchronization between WebGL FDTD physics and Python Spectral engines.'
                : 'Real-time synchronization between fundamental geometric physics and neural proxy.'}
            </p>
            
            <div className="mt-6 pointer-events-auto flex gap-2 items-center flex-wrap max-w-4xl">
              <button 
                className="px-4 py-2 bg-rose-600/20 hover:bg-rose-600/40 text-rose-300 border border-rose-500/30 rounded-lg backdrop-blur-md text-sm font-medium transition-all shadow-lg"
                onClick={() => setResetTrigger(r => r + 1)}
              >
                Reset Fields
              </button>
            </div>
          </div>

          <div className="flex-1 w-full h-full">
             {currentView === 'simulation' ? (
                <ComparisonEngine 
                  solverModeA="webgl" 
                  solverModeB="remote_spectral" 
                  titleA="FDTD (WebGL)" subtitleA="Browser Physics Engine"
                  titleB="Python Spectral" subtitleB="Remote Server Engine"
                  resetTrigger={resetTrigger} 
                />
             ) : (
                <ComparisonEngine 
                  solverModeA="webgl" 
                  solverModeB="huggingface" 
                  titleA="Ground Truth" subtitleA="FDTD (WebGL)"
                  titleB="GeoFNO Prediction" subtitleB="Hugging Face API"
                  resetTrigger={resetTrigger} 
                />
             )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
