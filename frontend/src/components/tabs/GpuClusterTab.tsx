import React, { useState } from 'react';
import { GpuStats } from '../../types';
import { Cpu, Activity, RefreshCw, Zap, Server, CheckCircle2, ShieldCheck } from 'lucide-react';

interface GpuClusterTabProps {
  gpus: GpuStats[];
  onRefresh: () => void;
}

export const GpuClusterTab: React.FC<GpuClusterTabProps> = ({ gpus, onRefresh }) => {
  const [isRefreshing, setIsRefreshing] = useState(false);

  const handleRefreshClick = () => {
    setIsRefreshing(true);
    onRefresh();
    setTimeout(() => setIsRefreshing(false), 800);
  };

  const workflows = [
    { name: 'WF-FOUNDRY-LOC', desc: 'Master plate + Depth & Canny maps', worker: 'Colab (gcgpu)' },
    { name: 'WF-FOUNDRY-CHAR', desc: '4-angle turnaround + 6 expressions', worker: 'GPU 0 (gpu0)' },
    { name: 'WF-SHOT-SOLO', desc: 'LoRA + IP-Adapter face + FaceDetailer', worker: 'GPU 0 (gpu0)' },
    { name: 'WF-SHOT-GRP', desc: 'Regional spatial cross-attention masks', worker: 'GPU 1 (gpu1)' },
    { name: 'WF-EDIT-INPAINT', desc: 'Differential diffusion defect fix', worker: 'GPU 1 (gpu1)' },
    { name: 'WF-MASTER-UPSCALE', desc: 'Tiled VAE 4K upscale pass', worker: 'Colab (gcgpu)' }
  ];

  return (
    <div className="p-6 max-w-[1920px] mx-auto space-y-6">
      {/* Cluster Summary Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 bg-studio-900 border border-studio-800 rounded-xl p-5 shadow-xl">
        <div>
          <div className="flex items-center gap-3">
            <h2 className="text-base font-bold text-white tracking-tight">
              3-GPU High-Throughput Production Cluster
            </h2>
            <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800">
              All 3 Online (HTTP 200)
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Aggregated Total VRAM: <span className="font-mono text-indigo-300 font-semibold">43.68 GB GDDR6</span> • Cloudflare Permanent Tunnels Active
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleRefreshClick}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-studio-800 hover:bg-studio-700 text-slate-200 text-xs font-medium border border-studio-700 transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 text-indigo-400 ${isRefreshing ? 'animate-spin' : ''}`} />
            <span>Probe Health</span>
          </button>
        </div>
      </div>

      {/* 3 GPU Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {gpus.map((gpu) => (
          <div
            key={gpu.id}
            className="bg-studio-900 border border-studio-800 rounded-xl p-5 shadow-xl flex flex-col justify-between"
          >
            <div>
              {/* Card Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-xl bg-studio-950 border border-studio-800 flex items-center justify-center">
                    <Cpu className="w-5 h-5 text-indigo-400" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white">{gpu.name}</h3>
                    <span className="text-[11px] font-mono text-slate-400 block truncate max-w-[200px]">
                      {gpu.url}
                    </span>
                  </div>
                </div>

                <span className="text-xs font-mono px-2 py-0.5 rounded-full bg-emerald-950/80 text-emerald-400 border border-emerald-800 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                  {gpu.latencyMs}ms
                </span>
              </div>

              {/* Role Badge */}
              <div className="mb-4 p-2.5 rounded-lg bg-studio-950 border border-studio-800 text-[11px] text-slate-300 flex items-center gap-2">
                <Zap className="w-3.5 h-3.5 text-amber-400 shrink-0" />
                <span className="font-medium">{gpu.role}</span>
              </div>

              {/* VRAM Allocation Gauge */}
              <div className="space-y-2 mb-4">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400 font-medium">VRAM Allocation</span>
                  <span className="font-mono text-slate-200">
                    {gpu.vramUsedGb.toFixed(2)} / {gpu.vramTotalGb.toFixed(2)} GB ({gpu.vramPercent.toFixed(1)}%)
                  </span>
                </div>
                <div className="w-full bg-studio-950 rounded-full h-2.5 overflow-hidden p-0.5 border border-studio-800">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 transition-all duration-500"
                    style={{ width: `${Math.max(6, gpu.vramPercent)}%` }}
                  />
                </div>
                <div className="flex justify-between text-[10px] font-mono text-slate-500">
                  <span>Free: {gpu.vramFreeGb.toFixed(2)} GB</span>
                  <span>Tesla T4 16GB</span>
                </div>
              </div>
            </div>

            {/* Queue & Job Status Footer */}
            <div className="pt-3 border-t border-studio-800 flex items-center justify-between text-xs font-mono">
              <span className="text-slate-400">Queue Remaining:</span>
              <span className="text-emerald-400 font-semibold">{gpu.queueRemaining} jobs</span>
            </div>
          </div>
        ))}
      </div>

      {/* Workflow Dispatch Matrix */}
      <div className="bg-studio-900 border border-studio-800 rounded-xl p-5 shadow-xl">
        <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
          <Server className="w-4 h-4 text-indigo-400" />
          <span>Workflow Dispatch Routing Rules</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {workflows.map((wf) => (
            <div key={wf.name} className="p-3 bg-studio-950 rounded-lg border border-studio-800 flex items-center justify-between">
              <div>
                <span className="text-xs font-mono font-bold text-indigo-300 block">{wf.name}</span>
                <span className="text-[11px] text-slate-400">{wf.desc}</span>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-studio-900 text-slate-200 border border-studio-700 whitespace-nowrap">
                {wf.worker}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
