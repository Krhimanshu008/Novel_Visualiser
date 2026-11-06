import React, { useState } from 'react';
import { Play, Pause, SkipBack, SkipForward, Volume2, Film, Mic, Music, Download, Settings } from 'lucide-react';

export const CinemaSequencerTab: React.FC = () => {
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playheadTime, setPlayheadTime] = useState<number>(1.8);
  const totalDuration = 7.5; // seconds for scene 1

  return (
    <div className="p-6 max-w-[1920px] mx-auto space-y-6">
      {/* Top Section: Video Preview Player & Master Controls */}
      <div className="grid grid-cols-12 gap-5">
        {/* Cinema Video Viewport */}
        <div className="col-span-12 lg:col-span-8 bg-studio-900 border border-studio-800 rounded-xl p-4 shadow-xl">
          <div className="relative aspect-[16/9] bg-black rounded-lg overflow-hidden border border-studio-800 flex items-center justify-center group">
            <img
              src="https://images.unsplash.com/photo-1518709268805-4e9042af9f23?q=80&w=1200&auto=format&fit=crop"
              alt="Cinema Playback"
              className="w-full h-full object-cover"
            />

            {/* Subtitle / Dialogue Overlay */}
            <div className="absolute bottom-6 left-8 right-8 text-center pointer-events-none">
              <span className="bg-black/75 px-4 py-1.5 rounded-md text-amber-200 font-medium text-sm tracking-wide shadow-lg border border-white/10 backdrop-blur-sm">
                “Another ordinary day in the outskirts... or so I hoped.”
              </span>
            </div>

            {/* Center Play Indicator */}
            {!isPlaying && (
              <div 
                onClick={() => setIsPlaying(true)}
                className="absolute w-14 h-14 rounded-full bg-indigo-600/90 text-white flex items-center justify-center cursor-pointer shadow-xl hover:scale-110 transition-transform backdrop-blur"
              >
                <Play className="w-6 h-6 fill-current ml-0.5" />
              </div>
            )}
          </div>

          {/* Scrubber & Player Bar */}
          <div className="mt-3 flex items-center justify-between gap-4 px-2">
            <div className="flex items-center gap-3">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="w-8 h-8 rounded-lg bg-studio-800 hover:bg-studio-700 text-white flex items-center justify-center transition-colors"
              >
                {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
              </button>
              <button className="text-slate-400 hover:text-white">
                <SkipBack className="w-4 h-4" />
              </button>
              <button className="text-slate-400 hover:text-white">
                <SkipForward className="w-4 h-4" />
              </button>
              <span className="font-mono text-xs text-slate-300 ml-2">
                00:0{playheadTime.toFixed(1)} / 00:0{totalDuration.toFixed(1)}
              </span>
            </div>

            <div className="flex items-center gap-3">
              <span className="text-xs font-mono text-slate-400">24 FPS • 1080p H.264</span>
            </div>
          </div>
        </div>

        {/* Export & Audio Settings */}
        <div className="col-span-12 lg:col-span-4 bg-studio-900 border border-studio-800 rounded-xl p-5 shadow-xl flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-bold text-white mb-1 flex items-center gap-2">
              <Film className="w-4 h-4 text-indigo-400" />
              <span>Cinema Assembly Master</span>
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Multi-track NLE timeline compiler with forced alignment audio.
            </p>

            <div className="space-y-3 text-xs">
              <div className="p-3 bg-studio-950 rounded-lg border border-studio-800">
                <span className="text-slate-400 block mb-1">TTS Voice Profile</span>
                <span className="text-slate-200 font-semibold font-mono">XTTS-v2: Sunny Cynical (Pitched -1.2)</span>
              </div>
              <div className="p-3 bg-studio-950 rounded-lg border border-studio-800">
                <span className="text-slate-400 block mb-1">BGM Soundtrack</span>
                <span className="text-slate-200 font-semibold font-mono">Forgotten_Shore_Desolation_Ambience.wav</span>
              </div>
              <div className="p-3 bg-studio-950 rounded-lg border border-studio-800">
                <span className="text-slate-400 block mb-1">Interpolation / Camera Pan</span>
                <span className="text-slate-200 font-semibold font-mono">Subtle Slow Push-In (+3% Zoom)</span>
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-studio-800">
            <button className="w-full py-3 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-bold text-xs shadow-lg shadow-indigo-600/30 flex items-center justify-center gap-2 transition-all">
              <Download className="w-4 h-4" />
              <span>Export Complete Chapter MP4 (1080p)</span>
            </button>
          </div>
        </div>
      </div>

      {/* Multi-Track Timeline */}
      <div className="bg-studio-900 border border-studio-800 rounded-xl p-5 shadow-xl space-y-3">
        <div className="flex items-center justify-between border-b border-studio-800 pb-2">
          <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">
            Multi-Track Timeline Sequencer
          </span>
          <span className="text-xs font-mono text-indigo-400">Total Duration: 7.5s (2 Cuts)</span>
        </div>

        {/* Track 1: Visual Video Shots */}
        <div className="flex items-center gap-3">
          <div className="w-28 text-xs font-mono text-slate-400 flex items-center gap-1.5 shrink-0">
            <Film className="w-3.5 h-3.5 text-indigo-400" />
            <span>V1 Shots</span>
          </div>
          <div className="flex-1 flex gap-2 h-12 bg-studio-950 rounded-lg p-1 border border-studio-800">
            <div className="w-[46%] bg-indigo-600/40 border border-indigo-500 rounded p-1.5 flex flex-col justify-between">
              <span className="text-[10px] font-bold text-white truncate">Shot #1: Outskirts Transit Bench</span>
              <span className="text-[9px] font-mono text-indigo-300">3.5s (Medium Angle)</span>
            </div>
            <div className="w-[54%] bg-purple-600/40 border border-purple-500 rounded p-1.5 flex flex-col justify-between">
              <span className="text-[10px] font-bold text-white truncate">Shot #2: Scavenger Girder Stalk</span>
              <span className="text-[9px] font-mono text-purple-300">4.0s (Wide Angle)</span>
            </div>
          </div>
        </div>

        {/* Track 2: Dialogue & Narration */}
        <div className="flex items-center gap-3">
          <div className="w-28 text-xs font-mono text-slate-400 flex items-center gap-1.5 shrink-0">
            <Mic className="w-3.5 h-3.5 text-emerald-400" />
            <span>A1 Dialogue</span>
          </div>
          <div className="flex-1 flex gap-2 h-10 bg-studio-950 rounded-lg p-1 border border-studio-800">
            <div className="w-[42%] bg-emerald-600/30 border border-emerald-500/60 rounded px-2 flex items-center">
              <span className="text-[10px] font-mono text-emerald-300 truncate">“Another ordinary day in the outskirts...”</span>
            </div>
          </div>
        </div>

        {/* Track 3: BGM Soundtrack */}
        <div className="flex items-center gap-3">
          <div className="w-28 text-xs font-mono text-slate-400 flex items-center gap-1.5 shrink-0">
            <Music className="w-3.5 h-3.5 text-amber-400" />
            <span>A2 BGM</span>
          </div>
          <div className="flex-1 h-10 bg-studio-950 rounded-lg p-1 border border-studio-800 flex">
            <div className="w-full bg-amber-600/20 border border-amber-500/40 rounded px-3 flex items-center justify-between">
              <span className="text-[10px] font-mono text-amber-300">Ambient Wasteland Drone (Volume: -14dB)</span>
              <span className="text-[9px] font-mono text-slate-500">Crossfade Loop</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
