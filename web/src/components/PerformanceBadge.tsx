import React from 'react';

interface PerformanceBadgeProps {
  inferenceMs: number;
  visible: boolean;
}

export const PerformanceBadge: React.FC<PerformanceBadgeProps> = ({ inferenceMs, visible }) => {
  if (!visible) return null;

  const stepsPerSec = inferenceMs > 0 ? Math.round(1000 / inferenceMs) : 0;
  // Assume spectral baseline is ~200ms
  const speedup = inferenceMs > 0 ? (200 / inferenceMs).toFixed(1) : 0;

  let colorClass = 'text-green-400';
  let dotClass = 'bg-green-500';
  let barClass = 'bg-green-500';
  if (inferenceMs > 50) {
    colorClass = 'text-red-400';
    dotClass = 'bg-red-500';
    barClass = 'bg-red-500';
  } else if (inferenceMs > 20) {
    colorClass = 'text-amber-400';
    dotClass = 'bg-amber-500';
    barClass = 'bg-amber-500';
  }

  // Bar width maxes out at 100ms
  const barWidth = Math.min(100, Math.max(0, (inferenceMs / 100) * 100));

  return (
    <div className="absolute bottom-6 right-6 z-50 pointer-events-none">
      <div className="bg-zinc-900/70 backdrop-blur-xl border border-zinc-700/50 rounded-xl p-4 shadow-2xl min-w-[200px]">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold text-zinc-300 uppercase tracking-wider">Neural Inference</span>
          <span className={`w-2 h-2 rounded-full ${dotClass} animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]`}></span>
        </div>
        
        <div className={`text-3xl font-bold font-mono tracking-tight ${colorClass} drop-shadow-sm`}>
          {inferenceMs.toFixed(1)}<span className="text-sm text-zinc-500 ml-1">ms</span>
        </div>

        <div className="flex flex-col gap-1 mt-3">
          <div className="flex justify-between text-xs">
            <span className="text-zinc-400">Throughput:</span>
            <span className="text-zinc-200 font-medium">~{stepsPerSec} steps/s</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-zinc-400">vs Spectral:</span>
            <span className="text-zinc-200 font-medium">{speedup}× faster</span>
          </div>
        </div>

        <div className="w-full bg-zinc-800 rounded-full h-1 mt-3 overflow-hidden">
          <div className={`h-1 rounded-full ${barClass} transition-all duration-100 ease-out`} style={{ width: `${barWidth}%` }}></div>
        </div>
      </div>
    </div>
  );
};
