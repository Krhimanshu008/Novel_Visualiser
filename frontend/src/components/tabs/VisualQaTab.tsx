import React, { useState } from 'react';
import { ShieldCheck, AlertTriangle, CheckCircle2, RefreshCw, Sparkles, Wrench, Eye } from 'lucide-react';

export const VisualQaTab: React.FC = () => {
  const [activeShot, setActiveShot] = useState<string>('shot_01');
  const [showDefectOverlay, setShowDefectOverlay] = useState<boolean>(true);
  const [isPatching, setIsPatching] = useState<boolean>(false);

  const auditData = {
    shot_01: {
      overallScore: 94.2,
      passed: true,
      imageUrl: 'https://images.unsplash.com/photo-1518709268805-4e9042af9f23?q=80&w=1200&auto=format&fit=crop',
      gate1: {
        passed: true,
        blurLaplacian: 842.1,
        exposureMean: 112.5,
        contrastStd: 56.4
      },
      gate2: {
        passed: true,
        wd14Score: 0.94,
        faceSim: 0.88,
        matchedTags: ['1boy', 'pale_skin', 'black_hair', 'dark_tunic', 'cautious_expression'],
        missingTags: []
      },
      gate3: {
        passed: true,
        vlmModel: 'Kaggle Gemini 2.5 Flash Multimodal Vision Audit',
        anatomy: 'Pristine 5-finger anatomical fidelity, seated silhouette aligns with stone bench.',
        lighting: 'Strong directional morning light matches background shadow vectors.',
        defects: []
      }
    },
    shot_02: {
      overallScore: 68.5,
      passed: false,
      imageUrl: 'https://images.unsplash.com/photo-1549490349-8643362247b5?q=80&w=1200&auto=format&fit=crop',
      gate1: {
        passed: true,
        blurLaplacian: 790.3,
        exposureMean: 95.1,
        contrastStd: 61.2
      },
      gate2: {
        passed: false,
        wd14Score: 0.62,
        faceSim: 0.58,
        matchedTags: ['1boy', 'black_hair'],
        missingTags: ['pale_skin', 'tattered_hood']
      },
      gate3: {
        passed: false,
        vlmModel: 'Kaggle Gemini 2.5 Flash Multimodal Vision Audit',
        anatomy: 'Right hand has 6 fused fingers near the monorail girder.',
        lighting: 'Monster carapace reflections are consistent with ambient sky.',
        defects: ['Right Hand Polydactyly Defect [x: 0.38, y: 0.52, w: 0.12, h: 0.15]']
      },
      defectBox: { x: 0.38, y: 0.52, w: 0.12, h: 0.15, label: 'Hand Defect' }
    }
  };

  const current = auditData[activeShot as keyof typeof auditData] || auditData.shot_01;

  const handleTriggerInpaint = () => {
    setIsPatching(true);
    setTimeout(() => {
      setIsPatching(false);
      alert('WF-EDIT-INPAINT dispatched to GPU 1 (gpu1.krhimanshu.in). Inpainting regional mask...');
    }, 1200);
  };

  return (
    <div className="grid grid-cols-12 gap-5 p-6 max-w-[1920px] mx-auto">
      {/* Left Column: Shot Triage List */}
      <div className="col-span-12 lg:col-span-3 flex flex-col gap-3">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
          Visual QA Review Queue
        </h3>

        {Object.entries(auditData).map(([key, item]) => (
          <div
            key={key}
            onClick={() => setActiveShot(key)}
            className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
              activeShot === key
                ? 'bg-indigo-500/10 border-indigo-500 shadow-md'
                : 'bg-studio-900 border-studio-800 hover:border-studio-700'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-white uppercase">{key.replace('_', ' ')}</span>
              <span
                className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${
                  item.passed
                    ? 'bg-emerald-950 text-emerald-400 border-emerald-800'
                    : 'bg-rose-950 text-rose-400 border-rose-800'
                }`}
              >
                {item.passed ? 'PASSED (94%)' : 'ACTION NEEDED (68%)'}
              </span>
            </div>
            <p className="text-[11px] text-slate-400">
              {item.passed ? 'All 3 Gates cleared.' : 'Gate 2 & 3 flagged anatomical defects.'}
            </p>
          </div>
        ))}
      </div>

      {/* Center Column: Render Preview with Defect Overlay */}
      <div className="col-span-12 lg:col-span-5 flex flex-col gap-4">
        <div className="bg-studio-900 border border-studio-800 rounded-xl p-4 shadow-xl">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-bold text-white uppercase tracking-wider">
              Rendered Shot & Defect Detection
            </h3>
            <label className="flex items-center gap-1.5 text-xs text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={showDefectOverlay}
                onChange={(e) => setShowDefectOverlay(e.target.checked)}
                className="rounded bg-studio-950 border-studio-700 text-rose-500 focus:ring-0"
              />
              <span>Highlight Defect Region</span>
            </label>
          </div>

          <div className="relative aspect-[16/9] bg-studio-950 rounded-lg overflow-hidden border border-studio-800">
            <img src={current.imageUrl} alt="Rendered Shot" className="w-full h-full object-cover" />

            {/* Defect Box Overlay if Present */}
            {showDefectOverlay && (current as any).defectBox && (
              <div
                className="absolute border-2 border-rose-500 bg-rose-500/20 rounded animate-pulse"
                style={{
                  left: `${(current as any).defectBox.x * 100}%`,
                  top: `${(current as any).defectBox.y * 100}%`,
                  width: `${(current as any).defectBox.w * 100}%`,
                  height: `${(current as any).defectBox.h * 100}%`,
                }}
              >
                <div className="absolute -top-6 left-0 bg-rose-600 text-white font-mono text-[10px] px-1.5 py-0.5 rounded shadow">
                  {(current as any).defectBox.label}
                </div>
              </div>
            )}
          </div>

          {/* Action Trigger */}
          <div className="mt-3 flex items-center justify-between">
            <span className="text-xs font-mono text-slate-400">
              Resolution: 1216×832 (Native SDXL)
            </span>
            {!(current as any).passed && (
              <button
                onClick={handleTriggerInpaint}
                disabled={isPatching}
                className="px-3.5 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold shadow-md shadow-rose-600/20 transition-all flex items-center gap-1.5"
              >
                <Wrench className="w-3.5 h-3.5" />
                <span>{isPatching ? 'Patching via GPU 1...' : 'Differential Inpaint Patch'}</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Right Column: Multi-Gate Deep Audit Breakdown */}
      <div className="col-span-12 lg:col-span-4 flex flex-col gap-4">
        {/* Gate 1 */}
        <div className="bg-studio-900 border border-studio-800 rounded-xl p-4 shadow-xl">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Gate 1: OpenCV Signal & Blur Audit</span>
            </h4>
            <span className="text-[10px] font-mono text-emerald-400">Passed</span>
          </div>
          <div className="grid grid-cols-3 gap-2 text-center text-xs bg-studio-950 p-2.5 rounded-lg border border-studio-800 font-mono">
            <div>
              <span className="text-slate-400 text-[10px] block">Laplacian</span>
              <span className="text-white font-bold">{current.gate1.blurLaplacian.toFixed(0)}</span>
            </div>
            <div>
              <span className="text-slate-400 text-[10px] block">Mean Exp</span>
              <span className="text-white font-bold">{current.gate1.exposureMean.toFixed(1)}</span>
            </div>
            <div>
              <span className="text-slate-400 text-[10px] block">Contrast</span>
              <span className="text-white font-bold">{current.gate1.contrastStd.toFixed(1)}</span>
            </div>
          </div>
        </div>

        {/* Gate 2 */}
        <div className="bg-studio-900 border border-studio-800 rounded-xl p-4 shadow-xl">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
              {current.gate2.passed ? (
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              ) : (
                <AlertTriangle className="w-4 h-4 text-rose-400" />
              )}
              <span>Gate 2: WD14 Tag & Face Identity</span>
            </h4>
            <span className={`text-[10px] font-mono ${current.gate2.passed ? 'text-emerald-400' : 'text-rose-400'}`}>
              Match: {(current.gate2.wd14Score * 100).toFixed(0)}%
            </span>
          </div>
          <div className="space-y-1.5 text-xs">
            <div className="flex flex-wrap gap-1">
              {current.gate2.matchedTags.map((t) => (
                <span key={t} className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-300 border border-emerald-800/40">
                  ✓ {t}
                </span>
              ))}
              {current.gate2.missingTags.map((t) => (
                <span key={t} className="text-[10px] font-mono px-2 py-0.5 rounded bg-rose-950/60 text-rose-300 border border-rose-800/40">
                  ✗ missing {t}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Gate 3 */}
        <div className="bg-studio-900 border border-studio-800 rounded-xl p-4 shadow-xl">
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-xs font-bold text-white flex items-center gap-1.5">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              <span>Gate 3: Kaggle Multimodal VLM Audit</span>
            </h4>
            <span className="text-[10px] font-mono text-indigo-300">Gemini 2.5 Flash</span>
          </div>
          <div className="space-y-2 text-xs bg-studio-950 p-3 rounded-lg border border-studio-800 leading-relaxed text-slate-300">
            <p><strong className="text-white">Anatomy:</strong> {current.gate3.anatomy}</p>
            <p><strong className="text-white">Lighting:</strong> {current.gate3.lighting}</p>
            {current.gate3.defects.length > 0 && (
              <div className="p-2 rounded bg-rose-950/40 border border-rose-900/60 text-rose-300 font-mono text-[11px]">
                Defects: {current.gate3.defects.join(', ')}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

