import React, { useState, useRef } from 'react';
import { BoundingBox } from '../types';
import { Move, Maximize2, Sparkles, User, Bug } from 'lucide-react';

interface SpatialFramingCanvasProps {
  backgroundImageUrl?: string;
  boxes: BoundingBox[];
  onBoxesChange: (boxes: BoundingBox[]) => void;
  selectedBoxId?: string;
  onSelectBox: (id: string) => void;
}

export const SpatialFramingCanvas: React.FC<SpatialFramingCanvasProps> = ({
  backgroundImageUrl,
  boxes,
  onBoxesChange,
  selectedBoxId,
  onSelectBox,
}) => {
  const [activeLayer, setActiveLayer] = useState<'rgb' | 'depth' | 'lineart'>('rgb');
  const [showGrid, setShowGrid] = useState(true);
  const containerRef = useRef<HTMLDivElement>(null);
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const handleMouseDown = (e: React.MouseEvent, box: BoundingBox) => {
    e.stopPropagation();
    onSelectBox(box.id);
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const mouseXNorm = (e.clientX - rect.left) / rect.width;
    const mouseYNorm = (e.clientY - rect.top) / rect.height;
    setDraggingId(box.id);
    setDragOffset({
      x: mouseXNorm - box.x,
      y: mouseYNorm - box.y,
    });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!draggingId || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const mouseXNorm = (e.clientX - rect.left) / rect.width;
    const mouseYNorm = (e.clientY - rect.top) / rect.height;

    const newX = Math.max(0, Math.min(1 - 0.1, mouseXNorm - dragOffset.x));
    const newY = Math.max(0, Math.min(1 - 0.1, mouseYNorm - dragOffset.y));

    onBoxesChange(
      boxes.map((b) =>
        b.id === draggingId
          ? {
              ...b,
              x: Number(newX.toFixed(3)),
              y: Number(newY.toFixed(3)),
            }
          : b
      )
    );
  };

  const handleMouseUp = () => {
    setDraggingId(null);
  };

  return (
    <div className="flex flex-col gap-2">
      {/* Controls & Layer Switches */}
      <div className="flex items-center justify-between px-3 py-1.5 bg-studio-900 border border-studio-800 rounded-lg text-xs">
        <div className="flex items-center gap-2">
          <span className="text-slate-400 font-medium">ControlNet Layer:</span>
          <div className="flex bg-studio-950 rounded p-0.5 border border-studio-800">
            {(['rgb', 'depth', 'lineart'] as const).map((l) => (
              <button
                key={l}
                onClick={() => setActiveLayer(l)}
                className={`px-2.5 py-1 rounded text-[11px] font-medium uppercase transition-colors ${
                  activeLayer === l
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {l}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1.5 text-slate-400 hover:text-slate-200 cursor-pointer">
            <input
              type="checkbox"
              checked={showGrid}
              onChange={(e) => setShowGrid(e.target.checked)}
              className="rounded bg-studio-950 border-studio-700 text-indigo-600 focus:ring-0"
            />
            <span className="text-[11px]">Rule of Thirds</span>
          </label>
          <span className="text-slate-500">•</span>
          <span className="text-[11px] font-mono text-slate-400">Aspect 16:9 (1216×832)</span>
        </div>
      </div>

      {/* Interactive Canvas Viewport */}
      <div
        ref={containerRef}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        className="relative w-full aspect-[1216/832] bg-studio-950 border border-studio-800 rounded-xl overflow-hidden shadow-2xl select-none"
        style={{
          backgroundImage:
            activeLayer === 'rgb'
              ? `url(${backgroundImageUrl || 'https://images.unsplash.com/photo-1518709268805-4e9042af9f23?q=80&w=1200&auto=format&fit=crop'})`
              : activeLayer === 'depth'
              ? 'radial-gradient(ellipse at 40% 60%, #1e1b4b 0%, #030712 100%)'
              : 'linear-gradient(to bottom, #111827, #030712)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
      >
        {/* Synthetic Depth or Lineart Simulation Filter */}
        {activeLayer === 'depth' && (
          <div className="absolute inset-0 bg-gradient-to-t from-indigo-900/40 to-transparent mix-blend-screen pointer-events-none" />
        )}

        {/* Rule of Thirds Overlay */}
        {showGrid && (
          <div className="absolute inset-0 pointer-events-none grid grid-cols-3 grid-rows-3 border border-indigo-500/10">
            <div className="border-r border-b border-indigo-400/20" />
            <div className="border-r border-b border-indigo-400/20" />
            <div className="border-b border-indigo-400/20" />
            <div className="border-r border-b border-indigo-400/20" />
            <div className="border-r border-b border-indigo-400/20" />
            <div className="border-b border-indigo-400/20" />
            <div className="border-r border-indigo-400/20" />
            <div className="border-r border-indigo-400/20" />
            <div />
          </div>
        )}

        {/* Bounding Boxes for Regional Conditioning */}
        {boxes.map((box) => {
          const isSelected = box.id === selectedBoxId;
          return (
            <div
              key={box.id}
              onMouseDown={(e) => handleMouseDown(e, box)}
              className={`absolute cursor-move transition-shadow rounded border-2 flex flex-col justify-between p-1.5 ${
                isSelected
                  ? 'border-indigo-400 bg-indigo-500/20 shadow-lg shadow-indigo-500/30'
                  : 'border-emerald-400/80 bg-emerald-500/10 hover:border-emerald-300'
              }`}
              style={{
                left: `${box.x * 100}%`,
                top: `${box.y * 100}%`,
                width: `${box.width * 100}%`,
                height: `${box.height * 100}%`,
              }}
            >
              {/* Header Pill */}
              <div className="flex items-center justify-between text-[11px] font-mono text-white bg-studio-950/80 px-1.5 py-0.5 rounded backdrop-blur border border-studio-700/60">
                <div className="flex items-center gap-1">
                  {box.role === 'creature' ? (
                    <Bug className="w-3 h-3 text-rose-400" />
                  ) : (
                    <User className="w-3 h-3 text-indigo-400" />
                  )}
                  <span className="font-semibold text-slate-100">{box.entityName}</span>
                </div>
                <span className="text-[10px] text-slate-400">
                  {Math.round(box.width * 100)}%w
                </span>
              </div>

              {/* Center crosshair */}
              <div className="flex items-center justify-center opacity-40">
                <Move className="w-4 h-4 text-white" />
              </div>

              {/* Action & Pose Tag */}
              <div className="text-[10px] text-slate-200 bg-studio-950/80 px-1 py-0.5 rounded truncate backdrop-blur border border-studio-800">
                {box.action}
              </div>

              {/* Corner Resize Handle */}
              <div className="absolute -bottom-1 -right-1 w-3 h-3 bg-white rounded-full border-2 border-indigo-600 shadow cursor-se-resize" />
            </div>
          );
        })}
      </div>
    </div>
  );
};
