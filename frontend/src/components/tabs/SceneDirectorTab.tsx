import React, { useState } from 'react';
import { ShotPlan, BoundingBox } from '../../types';
import { SpatialFramingCanvas } from '../SpatialFramingCanvas';
import { 
  Film, 
  Camera, 
  Clock, 
  Sliders, 
  Play, 
  Plus, 
  CheckCircle2, 
  Sparkles,
  Cpu
} from 'lucide-react';

export const SceneDirectorTab: React.FC = () => {
  const [selectedShotId, setSelectedShotId] = useState<string>('shot_01');
  const [selectedBoxId, setSelectedBoxId] = useState<string>('box_sunny');

  // Shot List with initial spatial bounding boxes
  const [shots, setShots] = useState<ShotPlan[]>([
    {
      id: 'shot_01',
      sceneId: 'sc_01',
      shotNumber: 1,
      shotType: 'medium',
      cameraAngle: 'low_angle',
      durationSeconds: 3.5,
      dialogue: '“Another ordinary day in the outskirts... or so I hoped.”',
      narration: 'Sunny sat on the cold stone transit bench, fingers coiled around his meager rations.',
      locationId: 'transit_bench',
      assignedGpu: 'GPU 0 (Port 8188)',
      status: 'passed_qc',
      qcScore: 94.2,
      boxes: [
        {
          id: 'box_sunny',
          entityName: 'Sunny',
          x: 0.28,
          y: 0.22,
          width: 0.44,
          height: 0.72,
          color: '#6366f1',
          role: 'protagonist',
          expression: 'cautious_cynical',
          action: 'sitting_hunched_on_bench_holding_rations',
        }
      ]
    },
    {
      id: 'shot_02',
      sceneId: 'sc_01',
      shotNumber: 2,
      shotType: 'wide',
      cameraAngle: 'eye_level',
      durationSeconds: 4.0,
      dialogue: '',
      narration: 'In the shadow of the crumbling monorail, a scavenger centipede twitched its mandibles.',
      locationId: 'transit_bench',
      assignedGpu: 'GPU 1 (Port 8189)',
      status: 'pending',
      boxes: [
        {
          id: 'box_sunny_02',
          entityName: 'Sunny',
          x: 0.15,
          y: 0.35,
          width: 0.28,
          height: 0.55,
          color: '#6366f1',
          role: 'protagonist',
          expression: 'alert_fear',
          action: 'standing_defensive',
        },
        {
          id: 'box_monster',
          entityName: 'Carapace Scavenger',
          x: 0.62,
          y: 0.40,
          width: 0.32,
          height: 0.50,
          color: '#f43f5e',
          role: 'creature',
          expression: 'predatory',
          action: 'crawling_along_monorail_girder',
        }
      ]
    }
  ]);

  const currentShot = shots.find((s) => s.id === selectedShotId) || shots[0];

  const handleBoxesChange = (newBoxes: BoundingBox[]) => {
    setShots(
      shots.map((s) => (s.id === currentShot.id ? { ...s, boxes: newBoxes } : s))
    );
  };

  const activeBox = currentShot.boxes.find((b) => b.id === selectedBoxId) || currentShot.boxes[0];

  return (
    <div className="grid grid-cols-12 gap-5 p-6 max-w-[1920px] mx-auto">
      {/* Left Column: Scene Beats & Shot Selection */}
      <div className="col-span-12 lg:col-span-3 flex flex-col gap-4">
        <div className="bg-studio-900 border border-studio-800 rounded-xl p-4 shadow-xl">
          <div className="flex items-center justify-between mb-3 border-b border-studio-800 pb-2">
            <div className="flex items-center gap-2">
              <Film className="w-4 h-4 text-indigo-400" />
              <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                Scene 1: Outskirts Transit Hub
              </h3>
            </div>
            <span className="text-[10px] font-mono text-slate-400">2 Shots</span>
          </div>

          <div className="space-y-2.5">
            {shots.map((shot) => (
              <div
                key={shot.id}
                onClick={() => {
                  setSelectedShotId(shot.id);
                  if (shot.boxes.length > 0) setSelectedBoxId(shot.boxes[0].id);
                }}
                className={`p-3 rounded-lg border cursor-pointer transition-all ${
                  selectedShotId === shot.id
                    ? 'bg-indigo-500/10 border-indigo-500 shadow-md'
                    : 'bg-studio-950/60 border-studio-800 hover:border-studio-700'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-white">Shot #{shot.shotNumber}</span>
                    <span className="text-[10px] font-mono uppercase px-1.5 py-0.5 rounded bg-studio-800 text-indigo-300">
                      {shot.shotType.replace('_', ' ')}
                    </span>
                  </div>
                  <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                    shot.status === 'passed_qc' 
                      ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' 
                      : 'bg-amber-950 text-amber-400 border border-amber-800'
                  }`}>
                    {shot.status === 'passed_qc' ? `QC ${shot.qcScore}%` : 'Draft'}
                  </span>
                </div>

                <p className="text-[11px] text-slate-300 italic line-clamp-2 mb-2">
                  {shot.narration || shot.dialogue}
                </p>

                <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 border-t border-studio-800/60 pt-1.5">
                  <span className="flex items-center gap-1">
                    <Clock className="w-3 h-3 text-studio-400" />
                    {shot.durationSeconds}s
                  </span>
                  <span className="text-indigo-400">{shot.assignedGpu}</span>
                </div>
              </div>
            ))}

            <button className="w-full py-2 bg-studio-950 hover:bg-studio-800 border border-dashed border-studio-700 rounded-lg text-xs font-medium text-slate-300 flex items-center justify-center gap-1.5 transition-all">
              <Plus className="w-3.5 h-3.5" />
              <span>Add New Shot to Beat</span>
            </button>
          </div>
        </div>
      </div>

      {/* Center Column: Interactive Spatial Framing Canvas */}
      <div className="col-span-12 lg:col-span-6 flex flex-col gap-4">
        <div className="bg-studio-900 border border-studio-800 rounded-xl p-4 shadow-xl">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h2 className="text-sm font-bold text-white flex items-center gap-2">
                <span>Spatial Regional Framing</span>
                <span className="text-xs font-mono font-normal text-indigo-400 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-800">
                  WF-SHOT-GRP Compositor
                </span>
              </h2>
              <p className="text-xs text-slate-400">
                Drag boxes to set 2D regional conditioning masks. ControlNet anchors background geometry.
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button 
                onClick={() => {
                  const newBox: BoundingBox = {
                    id: `box_${Date.now()}`,
                    entityName: 'Subject',
                    x: 0.4,
                    y: 0.3,
                    width: 0.3,
                    height: 0.6,
                    color: '#10b981',
                    role: 'supporting',
                    expression: 'neutral',
                    action: 'standing'
                  };
                  handleBoxesChange([...currentShot.boxes, newBox]);
                  setSelectedBoxId(newBox.id);
                }}
                className="px-2.5 py-1 bg-studio-800 hover:bg-studio-700 text-slate-200 rounded text-xs font-medium border border-studio-700 flex items-center gap-1"
              >
                <Plus className="w-3 h-3" />
                <span>Add Box</span>
              </button>
            </div>
          </div>

          <SpatialFramingCanvas
            backgroundImageUrl="https://images.unsplash.com/photo-1518709268805-4e9042af9f23?q=80&w=1200&auto=format&fit=crop"
            boxes={currentShot.boxes}
            onBoxesChange={handleBoxesChange}
            selectedBoxId={selectedBoxId}
            onSelectBox={(id) => setSelectedBoxId(id)}
          />

          <div className="mt-3 flex items-center justify-between text-xs text-slate-400 bg-studio-950 px-3 py-2 rounded-lg border border-studio-800 font-mono">
            <span>Active Subject Count: {currentShot.boxes.length}</span>
            <span className="text-indigo-400">Cross-Attention Mask Isolation: Active</span>
          </div>
        </div>
      </div>

      {/* Right Column: Shot & Regional Prompt Inspector */}
      <div className="col-span-12 lg:col-span-3 flex flex-col gap-4">
        {/* Shot Configuration */}
        <div className="bg-studio-900 border border-studio-800 rounded-xl p-4 shadow-xl">
          <div className="flex items-center justify-between mb-3 border-b border-studio-800 pb-2">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-400" />
              <h3 className="text-xs font-bold text-white uppercase tracking-wider">
                Shot Parameters
              </h3>
            </div>
            <span className="text-xs font-mono text-indigo-300">#{currentShot.shotNumber}</span>
          </div>

          <div className="space-y-3 text-xs">
            <div>
              <label className="text-slate-400 font-medium block mb-1">Camera Lens & Angle</label>
              <div className="grid grid-cols-2 gap-2">
                <select className="bg-studio-950 border border-studio-700 rounded-lg px-2.5 py-1.5 text-slate-200">
                  <option value="medium">Medium Shot (50mm)</option>
                  <option value="wide">Wide Angle (28mm)</option>
                  <option value="close_up">Close-up (85mm)</option>
                  <option value="extreme_wide">Extreme Wide (18mm)</option>
                </select>
                <select className="bg-studio-950 border border-studio-700 rounded-lg px-2.5 py-1.5 text-slate-200">
                  <option value="low_angle">Low Angle (+15°)</option>
                  <option value="eye_level">Eye Level (0°)</option>
                  <option value="high_angle">High Angle (-20°)</option>
                </select>
              </div>
            </div>

            <div>
              <label className="text-slate-400 font-medium block mb-1">Assigned GPU Worker</label>
              <div className="flex items-center gap-2 px-3 py-2 bg-studio-950 border border-studio-800 rounded-lg">
                <Cpu className="w-4 h-4 text-emerald-400" />
                <span className="text-slate-200 font-mono text-[11px]">{currentShot.assignedGpu}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Selected Bounding Box Regional Conditioning Inspector */}
        {activeBox && (
          <div className="bg-studio-900 border border-studio-800 rounded-xl p-4 shadow-xl">
            <div className="flex items-center justify-between mb-3 border-b border-studio-800 pb-2">
              <span className="text-xs font-bold text-indigo-300">
                Region: {activeBox.entityName}
              </span>
              <span className="text-[10px] font-mono text-slate-400">
                [{activeBox.x}, {activeBox.y}, {activeBox.width}, {activeBox.height}]
              </span>
            </div>

            <div className="space-y-3 text-xs">
              <div>
                <label className="text-slate-400 font-medium block mb-1">Pose & Action Tag</label>
                <input
                  type="text"
                  value={activeBox.action}
                  onChange={(e) => {
                    const updated = currentShot.boxes.map((b) =>
                      b.id === activeBox.id ? { ...b, action: e.target.value } : b
                    );
                    handleBoxesChange(updated);
                  }}
                  className="w-full bg-studio-950 border border-studio-700 rounded-lg px-3 py-1.5 text-slate-200 font-mono text-[11px]"
                />
              </div>

              <div>
                <label className="text-slate-400 font-medium block mb-1">Expression Anchor</label>
                <input
                  type="text"
                  value={activeBox.expression}
                  onChange={(e) => {
                    const updated = currentShot.boxes.map((b) =>
                      b.id === activeBox.id ? { ...b, expression: e.target.value } : b
                    );
                    handleBoxesChange(updated);
                  }}
                  className="w-full bg-studio-950 border border-studio-700 rounded-lg px-3 py-1.5 text-slate-200 font-mono text-[11px]"
                />
              </div>

              <div className="pt-2 border-t border-studio-800">
                <button className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium shadow-md transition-all flex items-center justify-center gap-2">
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Dispatch Shot to ComfyUI</span>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

