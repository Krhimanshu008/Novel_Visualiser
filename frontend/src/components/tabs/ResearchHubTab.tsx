import React, { useState } from 'react';
import { CharacterDossier } from '../../types';
import { Sparkles, Brain, ShieldAlert, Tag, FileText, CheckCircle2, ChevronRight, BookOpen } from 'lucide-react';

export const ResearchHubTab: React.FC = () => {
  const [selectedEntity, setSelectedEntity] = useState<string>('sunny');
  const [currentChapter, setCurrentChapter] = useState<number>(1);
  const [showThinking, setShowThinking] = useState<boolean>(true);

  const dossiers: Record<string, CharacterDossier> = {
    sunny: {
      id: 'sunny',
      name: 'Sunny (Sunless)',
      novel: 'Shadow Slave',
      epoch: 'Epoch 1: Sleeper (N-24 Outskirts)',
      validFrom: 1,
      validUntil: 98,
      danbooruTags: [
        '1boy', 'pale_skin', 'black_hair', 'messy_hair', 'dark_eyes', 
        'slender_build', 'dark_tunic', 'worn_clothes', 'tattered_hood',
        'cautious_expression', 'shadow_aura'
      ],
      fluxPrompt: 'A slender, pale young man with disheveled raven-black hair and intense dark observant eyes, wearing a worn charcoal survival tunic and patched dark cloak in a gritty futuristic slum.',
      antiSpoilerNegative: [
        'silver_armor', 'onyx_mantle', 'master_robe', 'saint_radiance', 
        'shadow_blade', 'royal_crown', 'muscular_physique', 'glowing_eyes'
      ],
      reasoningTrace: `Thought Process:
1. Novel: Shadow Slave, Arc 1: First Nightmare & Outskirts.
2. Current Chapter Grounding: Chapter 1.
3. Chronological Boundary: Sunny has not entered the Dream Realm yet. He is an impoverished Sleeper surviving in the outskirts of the N24 megacity.
4. Trait Expiration Analysis: The Onyx Mantle and Saint attributes occur at Chapters 400+, so they are strictly quarantined into the ComfyUI Anti-Spoiler Negative Block.
5. Danbooru Token Synthesis: Mapped visual attributes to Illustrious SDXL Danbooru tags with high weight on pale_skin and worn_clothes.`,
      avatarUrl: 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=300&auto=format&fit=crop'
    },
    nephis: {
      id: 'nephis',
      name: 'Nephism (Changing Star)',
      novel: 'Shadow Slave',
      epoch: 'Epoch 1: Awakened Candidate',
      validFrom: 15,
      validUntil: 350,
      danbooruTags: [
        '1girl', 'silver_hair', 'long_hair', 'calm_grey_eyes', 'fair_complexion',
        'slender_athletic', 'white_undershirt', 'combat_trousers', 'stoic_demeanor'
      ],
      fluxPrompt: 'A striking young woman with long shimmering silver hair and piercing cold grey eyes, wearing a tailored light combat undershirt with an unflinching resolute expression.',
      antiSpoilerNegative: [
        'immortal_flame_armor', 'angelic_wings', 'radiant_sword', 'scarred_face'
      ],
      reasoningTrace: `Thought Process:
1. Ingestion: Fandom wiki data mapped to Arc 1 Forgotten Shore arrival.
2. Trait quarantine: Her transcendent white flame armor does not exist in Chapter 1.`,
      avatarUrl: 'https://images.unsplash.com/photo-1517841905240-472988babdf9?q=80&w=300&auto=format&fit=crop'
    }
  };

  const active = dossiers[selectedEntity] || dossiers['sunny'];

  return (
    <div className="grid grid-cols-12 gap-5 p-6 max-w-[1920px] mx-auto">
      {/* Left Column: Source Ingest & Entity Directory */}
      <div className="col-span-12 lg:col-span-4 flex flex-col gap-4">
        {/* Source Configuration Card */}
        <div className="bg-studio-900 border border-studio-800 rounded-xl p-4 shadow-lg">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-indigo-400" />
              <h2 className="text-sm font-semibold text-white">Novel Ingestion & Grounding</h2>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
              Universal Engine
            </span>
          </div>

          <div className="space-y-3 text-xs">
            <div>
              <label className="text-slate-400 font-medium block mb-1">Novel Title & Genre</label>
              <div className="grid grid-cols-2 gap-2">
                <input
                  type="text"
                  defaultValue="Shadow Slave"
                  className="bg-studio-950 border border-studio-700 rounded-lg px-3 py-1.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
                <input
                  type="text"
                  defaultValue="Dark Fantasy / Cultivation"
                  className="bg-studio-950 border border-studio-700 rounded-lg px-3 py-1.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            <div>
              <label className="text-slate-400 font-medium block mb-1">Current Chapter Slider (Anti-Spoiler Gate)</label>
              <div className="flex items-center gap-3">
                <input
                  type="range"
                  min="1"
                  max="50"
                  value={currentChapter}
                  onChange={(e) => setCurrentChapter(Number(e.target.value))}
                  className="w-full accent-indigo-500"
                />
                <span className="font-mono text-indigo-400 font-semibold px-2 py-1 bg-studio-950 rounded border border-studio-800">
                  Ch.{currentChapter}
                </span>
              </div>
            </div>

            <div className="pt-2">
              <button className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg font-medium transition-all shadow-md flex items-center justify-center gap-2">
                <Brain className="w-4 h-4" />
                <span>Run Universal Reasoning Extractor</span>
              </button>
            </div>
          </div>
        </div>

        {/* Entity Dossier Directory */}
        <div className="bg-studio-900 border border-studio-800 rounded-xl p-4 shadow-lg flex-1">
          <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
            Reasoned Entity Dossiers
          </h3>

          <div className="space-y-2">
            {Object.values(dossiers).map((d) => (
              <div
                key={d.id}
                onClick={() => setSelectedEntity(d.id)}
                className={`p-3 rounded-lg border cursor-pointer transition-all flex items-center justify-between ${
                  selectedEntity === d.id
                    ? 'bg-indigo-500/10 border-indigo-500 shadow-md'
                    : 'bg-studio-950/60 border-studio-800 hover:border-studio-700'
                }`}
              >
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-studio-800 border border-studio-700 overflow-hidden">
                    <img src={d.avatarUrl} alt={d.name} className="w-full h-full object-cover" />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-white">{d.name}</h4>
                    <p className="text-[11px] text-slate-400">{d.epoch}</p>
                    <span className="text-[10px] font-mono text-indigo-300">
                      Ch.{d.validFrom} – {d.validUntil}
                    </span>
                  </div>
                </div>
                <ChevronRight className={`w-4 h-4 ${selectedEntity === d.id ? 'text-indigo-400' : 'text-slate-600'}`} />
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right Column: Active Dossier & Reasoning Trace */}
      <div className="col-span-12 lg:col-span-8 flex flex-col gap-4">
        {/* Main Dossier Card */}
        <div className="bg-studio-900 border border-studio-800 rounded-xl p-5 shadow-xl">
          <div className="flex items-start justify-between border-b border-studio-800 pb-4 mb-4">
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-bold text-white">{active.name}</h2>
                <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-indigo-950 text-indigo-300 border border-indigo-800/60">
                  {active.epoch}
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Canonical groundings extracted from novel prose and grounded against Chapter {currentChapter}.
              </p>
            </div>

            <div className="text-right">
              <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                Anti-Spoiler Verified
              </span>
              <span className="text-[10px] text-slate-500 block">SHA-256 Idempotency Cached</span>
            </div>
          </div>

          {/* Locked Danbooru Prompt Tags */}
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5 text-xs font-medium text-slate-300">
                <Tag className="w-3.5 h-3.5 text-indigo-400" />
                <span>Locked Danbooru Identity Block (Illustrious SDXL)</span>
              </div>
              <span className="text-[10px] font-mono text-slate-400">{active.danbooruTags.length} tags</span>
            </div>
            <div className="flex flex-wrap gap-1.5 p-3 rounded-lg bg-studio-950 border border-studio-800">
              {active.danbooruTags.map((tag) => (
                <span
                  key={tag}
                  className="text-xs font-mono px-2 py-0.5 rounded bg-studio-850 text-indigo-200 border border-studio-700"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>

          {/* Flux Natural Language Prompt */}
          <div className="mb-4">
            <div className="flex items-center gap-1.5 text-xs font-medium text-slate-300 mb-2">
              <FileText className="w-3.5 h-3.5 text-purple-400" />
              <span>Flux.1 Natural Language Synthesis</span>
            </div>
            <div className="p-3 rounded-lg bg-studio-950 border border-studio-800 text-xs text-slate-200 font-sans leading-relaxed">
              "{active.fluxPrompt}"
            </div>
          </div>

          {/* Anti-Spoiler Negative Quarantine Block */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-1.5 text-xs font-medium text-rose-400">
                <ShieldAlert className="w-3.5 h-3.5" />
                <span>Future Transformation Quarantined (Negative Prompt)</span>
              </div>
              <span className="text-[10px] font-mono text-rose-400/80">Auto-Injected (Ch.{currentChapter})</span>
            </div>
            <div className="flex flex-wrap gap-1.5 p-3 rounded-lg bg-rose-950/20 border border-rose-900/40">
              {active.antiSpoilerNegative.map((tag) => (
                <span
                  key={tag}
                  className="text-xs font-mono px-2 py-0.5 rounded bg-rose-950/60 text-rose-300 border border-rose-800/40"
                >
                  +{tag}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Chain-of-Thought Reasoning Inspector */}
        <div className="bg-studio-900 border border-studio-800 rounded-xl p-4 shadow-lg">
          <div 
            onClick={() => setShowThinking(!showThinking)}
            className="flex items-center justify-between cursor-pointer"
          >
            <div className="flex items-center gap-2">
              <Brain className="w-4 h-4 text-amber-400" />
              <h3 className="text-xs font-semibold text-slate-200">
                Universal LLM Chain-of-Thought Trace (&lt;think&gt;)
              </h3>
            </div>
            <span className="text-xs font-mono text-slate-400">
              {showThinking ? 'Collapse' : 'Expand'}
            </span>
          </div>

          {showThinking && (
            <div className="mt-3 p-3.5 rounded-lg bg-studio-950 border border-studio-800 font-mono text-xs text-slate-400 whitespace-pre-wrap leading-relaxed">
              {active.reasoningTrace}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

