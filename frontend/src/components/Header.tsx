import React from 'react';
import { TabType, GpuStats } from '../types';
import { GpuBadge } from './GpuBadge';
import { 
  Sparkles, 
  Layers, 
  Film, 
  Cpu, 
  ShieldCheck, 
  Clapperboard, 
  Play,
  Bookmark
} from 'lucide-react';

interface HeaderProps {
  currentTab: TabType;
  onTabChange: (tab: TabType) => void;
  gpus: GpuStats[];
  projectTitle: string;
  chapterTitle: string;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onTabChange,
  gpus,
  projectTitle,
  chapterTitle
}) => {
  const tabs: { id: TabType; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: 'research', label: '1. Research Hub', icon: <Sparkles className="w-4 h-4" /> },
    { id: 'foundry', label: '2. Asset Foundry', icon: <Layers className="w-4 h-4" /> },
    { id: 'director', label: '3. Scene Director', icon: <Film className="w-4 h-4" /> },
    { id: 'cluster', label: '4. 3-GPU Cluster', icon: <Cpu className="w-4 h-4" />, badge: '3 Active' },
    { id: 'qa', label: '5. Multi-Gate QA', icon: <ShieldCheck className="w-4 h-4" /> },
    { id: 'cinema', label: '6. Cinema Timeline', icon: <Clapperboard className="w-4 h-4" /> },
  ];

  return (
    <header className="sticky top-0 z-50 bg-studio-950/90 backdrop-blur-md border-b border-studio-800">
      {/* Top Bar: Studio Branding & Hardware Cluster Status */}
      <div className="max-w-[1920px] mx-auto px-5 py-2.5 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-500 flex items-center justify-center shadow-md shadow-indigo-500/20">
              <Film className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold text-white tracking-tight">Novel Visualiser</h1>
                <span className="text-[10px] font-mono font-medium px-2 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                  STUDIO v2.0
                </span>
              </div>
              <p className="text-xs text-slate-400 flex items-center gap-1.5">
                <Bookmark className="w-3 h-3 text-studio-400" />
                <span className="text-slate-300 font-medium">{projectTitle}</span>
                <span className="text-studio-600">•</span>
                <span className="text-indigo-300">{chapterTitle}</span>
              </p>
            </div>
          </div>
        </div>

        {/* 3-GPU Live Cluster Badges */}
        <div className="hidden lg:flex items-center gap-3">
          {gpus.map((gpu) => (
            <GpuBadge key={gpu.id} gpu={gpu} />
          ))}
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3">
          <button className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs shadow-md shadow-emerald-600/20 transition-all">
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>Render Batch</span>
          </button>
        </div>
      </div>

      {/* Navigation Tabs Bar */}
      <div className="max-w-[1920px] mx-auto px-5 flex items-center gap-1 border-t border-studio-800/60 overflow-x-auto scrollbar-none">
        {tabs.map((tab) => {
          const isActive = currentTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`flex items-center gap-2 px-4 py-2.5 text-xs font-medium border-b-2 transition-all whitespace-nowrap ${
                isActive
                  ? 'border-indigo-500 text-indigo-300 bg-indigo-500/5'
                  : 'border-transparent text-slate-400 hover:text-slate-200 hover:bg-studio-900/40'
              }`}
            >
              <span className={isActive ? 'text-indigo-400' : 'text-slate-400'}>
                {tab.icon}
              </span>
              <span>{tab.label}</span>
              {tab.badge && (
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded-full bg-emerald-950 text-emerald-400 border border-emerald-800/40">
                  {tab.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </header>
  );
};
