import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import { StaticTorus } from './StaticTorus';

interface AnalysisData {
    buffer: Float32Array;
    framesLoaded: number;
    totalFrames: number;
    height: number;
    width: number;
}

export const ChaosAnalysis: React.FC = () => {
    const [gtData, setGtData] = useState<AnalysisData | null>(null);
    const [predData, setPredData] = useState<AnalysisData | null>(null);
    const [loadingGt, setLoadingGt] = useState({ progress: 0 });
    const [loadingPred, setLoadingPred] = useState({ progress: 0 });
    
    const [currentFrame, setCurrentFrame] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [speed, setSpeed] = useState(1);
    
    const [errorBuffer, setErrorBuffer] = useState<Float32Array | null>(null);

    // Stream binary data
    useEffect(() => {
        let isMounted = true;
        
        const streamData = async (url: string, setData: React.Dispatch<React.SetStateAction<AnalysisData | null>>, setLoading: React.Dispatch<React.SetStateAction<{ progress: number }>>) => {
            try {
                const response = await fetch(url);
                if (!response.body) throw new Error("ReadableStream not supported.");
                
                const contentLength = +(response.headers.get('Content-Length') || 0);
                
                // If content length is unknown, assume ~58MB (897 frames of 128x128 float32)
                const totalBytes = contentLength || (897 * 128 * 128 * 4);
                const totalFrames = Math.floor(totalBytes / (128 * 128 * 4));
                
                const buffer = new Float32Array(totalBytes / 4);
                let loadedBytes = 0;
                
                const reader = response.body.getReader();
                
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    if (!isMounted) return;
                    
                    // value is a Uint8Array
                    const chunkFloats = new Float32Array(value.buffer, value.byteOffset, value.byteLength / 4);
                    buffer.set(chunkFloats, loadedBytes / 4);
                    loadedBytes += value.byteLength;
                    
                    const framesLoaded = Math.floor(loadedBytes / (128 * 128 * 4));
                    const progress = Math.round((loadedBytes / totalBytes) * 100);
                    
                    setLoading({ progress });
                    setData({
                        buffer,
                        framesLoaded,
                        totalFrames,
                        height: 128,
                        width: 128
                    });
                }
            } catch (err) {
                console.error("Stream failed:", err);
            }
        };

        streamData('/data/complex_chaos_gt.bin', setGtData, setLoadingGt);
        streamData('/data/complex_chaos_pred.bin', setPredData, setLoadingPred);

        return () => { isMounted = false; };
    }, []);

    // Playback loop
    useEffect(() => {
        let req: number;
        let lastTime = performance.now();
        const fps = 30 * speed;
        const interval = 1000 / fps;

        const loop = (time: number) => {
            if (isPlaying && gtData && predData) {
                if (time - lastTime >= interval) {
                    const maxLoaded = Math.min(gtData.framesLoaded, predData.framesLoaded);
                    if (maxLoaded > 0) {
                        setCurrentFrame(prev => (prev + 1) % maxLoaded);
                    }
                    lastTime = time;
                }
            }
            req = requestAnimationFrame(loop);
        };
        req = requestAnimationFrame(loop);
        return () => cancelAnimationFrame(req);
    }, [isPlaying, gtData, predData, speed]);

    // Compute error buffer always so PIP window has data
    useEffect(() => {
        if (gtData && predData) {
            const maxLoaded = Math.min(gtData.framesLoaded, predData.framesLoaded);
            if (currentFrame < maxLoaded) {
                const frameSize = gtData.width * gtData.height;
                const start = currentFrame * frameSize;
                const gtFrame = gtData.buffer.subarray(start, start + frameSize);
                const predFrame = predData.buffer.subarray(start, start + frameSize);
                
                const err = new Float32Array(frameSize);
                for (let i = 0; i < frameSize; i++) {
                    err[i] = Math.abs(predFrame[i] - gtFrame[i]); // Made it absolute error to look like heat map
                }
                setErrorBuffer(err);
            }
        }
    }, [currentFrame, gtData, predData]);

    const getFrameBuffer = (data: AnalysisData | null) => {
        if (!data || data.framesLoaded === 0) return null;
        // Clamp to loaded frames
        const safeFrame = Math.min(currentFrame, data.framesLoaded - 1);
        const frameSize = data.width * data.height;
        const start = safeFrame * frameSize;
        return data.buffer.subarray(start, start + frameSize);
    };

    // Calculate overall max loaded frames between both streams
    const maxLoadedFrames = Math.min(gtData?.framesLoaded || 0, predData?.framesLoaded || 0);
    const targetFrames = gtData?.totalFrames || 897;

    return (
        <div className="w-full h-full flex flex-col relative bg-black font-sans">
            
            {/* Streaming Overlay (Doesn't block UI playback!) */}
            {(loadingGt.progress < 100 || loadingPred.progress < 100) && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 z-50 bg-zinc-950/80 backdrop-blur-md border border-zinc-800 rounded-2xl p-4 flex gap-8 shadow-2xl">
                    <div className="w-32">
                        <div className="text-xs text-zinc-400 flex justify-between mb-1">
                            <span>GT Stream</span>
                            <span>{loadingGt.progress}%</span>
                        </div>
                        <div className="w-full h-1.5 bg-zinc-800 rounded-full">
                            <div className="h-full bg-blue-500 rounded-full transition-all duration-300" style={{ width: `${loadingGt.progress}%` }}></div>
                        </div>
                    </div>
                    <div className="w-32">
                        <div className="text-xs text-zinc-400 flex justify-between mb-1">
                            <span>Pred Stream</span>
                            <span>{loadingPred.progress}%</span>
                        </div>
                        <div className="w-full h-1.5 bg-zinc-800 rounded-full">
                            <div className="h-full bg-emerald-500 rounded-full transition-all duration-300" style={{ width: `${loadingPred.progress}%` }}></div>
                        </div>
                    </div>
                </div>
            )}

            {/* Main Canvas for GT and Prediction Side-by-Side */}
            <div className="flex-1 flex w-full relative">
                <Canvas camera={{ position: [0, 0, 6], fov: 45 }}>
                    <ambientLight intensity={0.2} />
                    <pointLight position={[10, 10, 10]} intensity={1} />
                    <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />
                    
                    <group position={[-2.2, 0, 0]}>
                        <StaticTorus R={1.5} r={0.5} radialSegments={128} tubularSegments={128} res={128} frameData={getFrameBuffer(gtData)} />
                    </group>
                    
                    <group position={[2.2, 0, 0]}>
                        <StaticTorus R={1.5} r={0.5} radialSegments={128} tubularSegments={128} res={128} frameData={getFrameBuffer(predData)} />
                    </group>
                    
                    <OrbitControls autoRotate={isPlaying} autoRotateSpeed={0.2} enablePan={false} maxDistance={12} minDistance={3} />
                </Canvas>

                {/* Overlay Labels */}
                <div className="absolute top-8 left-[20%] -translate-x-1/2 pointer-events-none">
                    <div className="bg-black/40 backdrop-blur border border-zinc-800 rounded-lg px-4 py-2 text-center shadow-2xl">
                        <h3 className="text-white font-semibold text-lg drop-shadow-md">Ground Truth</h3>
                        <p className="text-blue-400 text-xs mt-1">Target Trajectory</p>
                    </div>
                </div>

                <div className="absolute top-8 left-[80%] -translate-x-1/2 pointer-events-none">
                    <div className="bg-black/40 backdrop-blur border border-zinc-800 rounded-lg px-4 py-2 text-center shadow-2xl">
                        <h3 className="text-white font-semibold text-lg drop-shadow-md">GeoFNO Prediction</h3>
                        <p className="text-emerald-400 text-xs mt-1">Inferred Trajectory</p>
                    </div>
                </div>

                {/* Error Inset Picture-in-Picture */}
                <div className="absolute bottom-8 right-8 w-64 h-64 bg-zinc-950/80 backdrop-blur-xl border border-zinc-700 rounded-2xl shadow-[0_0_40px_rgba(255,0,0,0.1)] overflow-hidden flex flex-col z-30">
                    <div className="bg-zinc-900/90 border-b border-zinc-700 px-3 py-2 flex justify-between items-center z-10 pointer-events-none">
                        <span className="text-xs font-semibold text-rose-300 uppercase tracking-wider">Absolute Error</span>
                        <div className="flex gap-1">
                            <div className="w-2 h-2 rounded-full bg-rose-500 animate-pulse"></div>
                        </div>
                    </div>
                    <div className="flex-1 relative">
                        <Canvas camera={{ position: [0, -3, 3], fov: 45 }}>
                            <ambientLight intensity={0.2} />
                            <pointLight position={[10, 10, 10]} intensity={1} />
                            <StaticTorus R={1.5} r={0.5} radialSegments={128} tubularSegments={128} res={128} frameData={errorBuffer} colorMap="error" intensityMultiplier={2.0} />
                            <OrbitControls enablePan={false} enableZoom={false} autoRotate autoRotateSpeed={2.0} />
                        </Canvas>
                    </div>
                </div>
            </div>

            {/* Timeline UI Panel */}
            <div className="h-32 bg-zinc-950/80 backdrop-blur-xl border-t border-zinc-800 p-6 flex flex-col justify-center z-20">
                <div className="flex items-center gap-4 mb-4">
                    <button 
                        onClick={() => setIsPlaying(!isPlaying)}
                        disabled={maxLoadedFrames === 0}
                        className="w-12 h-12 flex items-center justify-center bg-blue-600 hover:bg-blue-500 text-white rounded-full shadow-lg shadow-blue-600/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                    >
                        {isPlaying ? (
                            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" /></svg>
                        ) : (
                            <svg className="w-5 h-5 ml-1" fill="currentColor" viewBox="0 0 20 20"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" /></svg>
                        )}
                    </button>
                    
                    <div className="flex-1 relative">
                        {/* Buffer progress bar behind the thumb */}
                        <div className="absolute top-1/2 -translate-y-1/2 left-0 h-2 bg-zinc-700/50 rounded-lg pointer-events-none" style={{ width: `${(maxLoadedFrames / targetFrames) * 100}%` }}></div>
                        
                        <input 
                            type="range" 
                            min="0" 
                            max={Math.max(0, maxLoadedFrames - 1)} 
                            value={Math.min(currentFrame, maxLoadedFrames - 1)}
                            onChange={(e) => {
                                setIsPlaying(false);
                                setCurrentFrame(parseInt(e.target.value));
                            }}
                            className="w-full h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer accent-blue-500 relative z-10 bg-transparent"
                            disabled={maxLoadedFrames === 0}
                        />
                        <div className="flex justify-between text-xs text-zinc-500 mt-2 font-mono">
                            <span>Frame {currentFrame}</span>
                            <span>{targetFrames} total</span>
                        </div>
                    </div>

                    <div className="flex gap-2 bg-zinc-900 rounded-lg p-1 border border-zinc-800">
                        {[0.5, 1, 2, 4].map(s => (
                            <button 
                                key={s}
                                onClick={() => setSpeed(s)}
                                className={`px-2 py-1 rounded text-xs font-medium transition-colors ${speed === s ? 'bg-zinc-700 text-white' : 'text-zinc-500 hover:text-zinc-300'}`}
                            >
                                {s}x
                            </button>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};
