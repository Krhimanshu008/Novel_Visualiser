import React from 'react';
import { GpuStats } from '../types';
import { Cpu } from 'lucide-react';

interface GpuBadgeProps {
  gpu: GpuStats;
}

export const GpuBadge: React.FC<GpuBadgeProps> = ({ gpu }) => {
  return (
    <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg bg-studio-900 border border-studio-700/60 shadow-sm hover:border-studio-500/50 transition-all cursor-pointer">
      <div className="relative">
        <Cpu className="w-4 h-4 text-studio-400" />
        <span
          className={`absolute -bottom-0.5 -right-0.5 w-2 h-2 rounded-full ${
            gpu.online ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'
          }`}
        />
      </div>
      <div className="flex flex-col text-left">
        <div className="flex items-center gap-1.5">
          <span className="text-xs font-semibold text-slate-200">
            {gpu.id.toUpperCase()}
          </span>
          <span className="text-[10px] font-mono text-emerald-400 bg-emerald-950/60 px-1 py-0.2 rounded border border-emerald-800/40">
            {gpu.latencyMs}ms
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-14 bg-studio-800 rounded-full h-1.5 overflow-hidden">
            <div
              className={`h-full rounded-full ${
                gpu.vramPercent > 85
                  ? 'bg-rose-500'
                  : gpu.vramPercent > 60
                  ? 'bg-amber-400'
                  : 'bg-emerald-400'
              }`}
              style={{ width: `${Math.max(8, gpu.vramPercent)}%` }}
            />
          </div>
          <span className="text-[10px] text-slate-400 font-mono">
            {gpu.vramFreeGb.toFixed(1)}GB free
          </span>
        </div>
      </div>
    </div>
  );
};
