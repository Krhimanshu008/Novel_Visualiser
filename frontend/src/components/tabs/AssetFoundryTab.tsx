import React, { useState } from 'react';
import { Layers, MapPin, User, Bug, Sun, Check, ExternalLink, Image as ImageIcon } from 'lucide-react';

export const AssetFoundryTab: React.FC = () => {
  const [subTab, setSubTab] = useState<'locations' | 'characters' | 'creatures' | 'environments'>('locations');
  const [selectedAsset, setSelectedAsset] = useState<string>('transit_bench');

  const locationAssets = [
    {
      id: 'transit_bench',
      name: 'Outskirts Transit Bench N24',
      type: 'Urban Slum / Transit Node',
      lighting: 'Harsh Directional Sunlight / Low Slum Haze',
      plateUrl: 'https://images.unsplash.com/photo-1518709268805-4e9042af9f23?q=80&w=1200&auto=format&fit=crop',
      depthUrl: 'https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=1200&auto=format&fit=crop',
      lineartUrl: 'https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?q=80&w=1200&auto=format&fit=crop',
      reverseUrl: 'https://images.unsplash.com/photo-1509198397868-475647b2a1e5?q=80&w=1200&auto=format&fit=crop',
    },
    {
      id: 'ruined_temple',
      name: 'Forgotten Shore - Sunken Sun Temple',
      type: 'Ancient Coral Ruins',
      lighting: 'Eerie Violet Bioluminescent Depths',
      plateUrl: 'https://images.unsplash.com/photo-1544551763-46a013bb70d5?q=80&w=1200&auto=format&fit=crop',
      depthUrl: 'https://images.unsplash.com/photo-1550684848-fac1c5b4e853?q=80&w=1200&auto=format&fit=crop',
      lineartUrl: 'https://images.unsplash.com/photo-1579783900882-c0d3dad7b119?q=80&w=1200&auto=format&fit=crop',
      reverseUrl: 'https://images.unsplash.com/photo-1544551763-46a013bb70d5?q=80&w=1200&auto=format&fit=crop',
    }
  ];

  const creatures = [
    {
      id: 'carapace_scavenger',
      name: 'Carapace Centipede Scavenger',
      rank: 'Awakened',
      threat: 'Lethal / Swarm',
      prompt: 'Nightmare creature, chitinous segmented armor, multi-jointed bladed limbs, venomous mandibles, dark purple carapace sheen.',
      imageUrl: 'https://images.unsplash.com/photo-1549490349-8643362247b5?q=80&w=800&auto=format&fit=crop'
    },
    {
      id: 'shadow_abomination',
      name: 'Void Shadow Lurker',
      rank: 'Fallen',
      threat: 'Calamity',
      prompt: 'Formless nightmare entity, tendrils of smoke, empty glowing eye sockets, shifting smoke silhouette, jagged obsidian teeth.',
      imageUrl: 'https://images.unsplash.com/photo-1509281373149-e957c6296406?q=80&w=800&auto=format&fit=crop'
    }
  ];

  const activeLoc = locationAssets.find((l) => l.id === selectedAsset) || locationAssets[0];

  return (
    <div className="grid grid-cols-12 gap-5 p-6 max-w-[1920px] mx-auto">
      {/* Sub-Navigation Bar */}
      <div className="col-span-12 flex items-center justify-between border-b border-studio-800 pb-3">
        <div className="flex items-center gap-2">
          {(['locations', 'characters', 'creatures', 'environments'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setSubTab(tab)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold capitalize transition-all ${
                subTab === tab
                  ? 'bg-indigo-600 text-white shadow-md'
                  : 'bg-studio-900 text-slate-400 hover:text-white border border-studio-800'
              }`}
            >
              {tab === 'locations' && <MapPin className="w-3.5 h-3.5" />}
              {tab === 'characters' && <User className="w-3.5 h-3.5" />}
              {tab === 'creatures' && <Bug className="w-3.5 h-3.5" />}
              {tab === 'environments' && <Sun className="w-3.5 h-3.5" />}
              <span>{tab}</span>
            </button>
          ))}
        </div>

        <button className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-studio-800 hover:bg-studio-700 text-slate-200 text-xs font-medium border border-studio-700">
          <Layers className="w-3.5 h-3.5 text-indigo-400" />
          <span>Generate New {subTab.slice(0, -1)} Asset</span>
        </button>
      </div>

      {/* Main Content Area */}
      {subTab === 'locations' && (
        <>
          {/* Location Asset List */}
          <div className="col-span-12 lg:col-span-4 flex flex-col gap-3">
            <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Canonical Location Plates
            </h3>
            {locationAssets.map((loc) => (
              <div
                key={loc.id}
                onClick={() => setSelectedAsset(loc.id)}
                className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                  selectedAsset === loc.id
                    ? 'bg-indigo-500/10 border-indigo-500 shadow-md'
                    : 'bg-studio-900 border-studio-800 hover:border-studio-700'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <h4 className="text-xs font-bold text-white">{loc.name}</h4>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-studio-950 text-indigo-300 border border-studio-800">
                    4 Plates
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 mb-2">{loc.type}</p>
                <div className="text-[10px] text-slate-400 font-mono flex items-center gap-1.5">
                  <Sun className="w-3 h-3 text-amber-400" />
                  <span>{loc.lighting}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Location 4-Plate Matrix View */}
          <div className="col-span-12 lg:col-span-8 flex flex-col gap-4">
            <div className="bg-studio-900 border border-studio-800 rounded-xl p-5 shadow-xl">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h3 className="text-sm font-bold text-white">{activeLoc.name}</h3>
                  <p className="text-xs text-slate-400">WF-FOUNDRY-LOC Multi-Angle ControlNet Anchors</p>
                </div>
                <span className="text-xs font-mono text-emerald-400 flex items-center gap-1">
                  <Check className="w-3.5 h-3.5" />
                  Continuity Locked
                </span>
              </div>

              {/* 4-Plate Grid */}
              <div className="grid grid-cols-2 gap-4">
                {/* Plate 1: Wide Establishing */}
                <div className="group relative rounded-lg overflow-hidden border border-studio-800 bg-studio-950">
                  <img
                    src={activeLoc.plateUrl}
                    alt="Wide Establishing"
                    className="w-full aspect-[16/9] object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute top-2 left-2 px-2 py-0.5 bg-studio-950/80 backdrop-blur rounded text-[10px] font-mono text-white border border-studio-700">
                    1. Wide Establishing Plate (RGB)
                  </div>
                </div>

                {/* Plate 2: Depth ControlNet */}
                <div className="group relative rounded-lg overflow-hidden border border-studio-800 bg-studio-950">
                  <img
                    src={activeLoc.depthUrl}
                    alt="Depth Map"
                    className="w-full aspect-[16/9] object-cover filter contrast-125 group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute top-2 left-2 px-2 py-0.5 bg-studio-950/80 backdrop-blur rounded text-[10px] font-mono text-indigo-300 border border-indigo-700/60">
                    2. Depth Map (Union ControlNet 0.65)
                  </div>
                </div>

                {/* Plate 3: Lineart / Canny */}
                <div className="group relative rounded-lg overflow-hidden border border-studio-800 bg-studio-950">
                  <img
                    src={activeLoc.lineartUrl}
                    alt="Lineart"
                    className="w-full aspect-[16/9] object-cover filter invert group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute top-2 left-2 px-2 py-0.5 bg-studio-950/80 backdrop-blur rounded text-[10px] font-mono text-amber-300 border border-amber-700/60">
                    3. Lineart / Edge Geometry (Canny 0.45)
                  </div>
                </div>

                {/* Plate 4: Reverse Reciprocal */}
                <div className="group relative rounded-lg overflow-hidden border border-studio-800 bg-studio-950">
                  <img
                    src={activeLoc.reverseUrl}
                    alt="Reverse Shot"
                    className="w-full aspect-[16/9] object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute top-2 left-2 px-2 py-0.5 bg-studio-950/80 backdrop-blur rounded text-[10px] font-mono text-emerald-300 border border-emerald-700/60">
                    4. Reverse Reciprocal Plate (180° Rule)
                  </div>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Creatures & Monsters View */}
      {subTab === 'creatures' && (
        <div className="col-span-12 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
          {creatures.map((c) => (
            <div key={c.id} className="bg-studio-900 border border-studio-800 rounded-xl overflow-hidden shadow-xl">
              <div className="relative aspect-[16/10] bg-studio-950">
                <img src={c.imageUrl} alt={c.name} className="w-full h-full object-cover" />
                <div className="absolute top-2.5 right-2.5 px-2.5 py-1 rounded bg-rose-950/80 border border-rose-800 backdrop-blur text-[11px] font-bold text-rose-300">
                  Rank: {c.rank}
                </div>
              </div>

              <div className="p-4 space-y-2.5">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-bold text-white">{c.name}</h4>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-studio-800 text-slate-300">
                    {c.threat}
                  </span>
                </div>
                <p className="text-xs text-slate-400 font-sans leading-relaxed">
                  "{c.prompt}"
                </p>
                <div className="pt-2 border-t border-studio-800 flex items-center justify-between text-xs">
                  <span className="text-indigo-400 font-mono text-[11px]">WF-FOUNDRY-CREATURE</span>
                  <button className="text-slate-300 hover:text-white font-medium flex items-center gap-1">
                    <span>Inspect LoRA</span>
                    <ExternalLink className="w-3 h-3" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Characters Turnaround View */}
      {subTab === 'characters' && (
        <div className="col-span-12 bg-studio-900 border border-studio-800 rounded-xl p-5 shadow-xl">
          <div className="flex items-center justify-between mb-4 border-b border-studio-800 pb-3">
            <div>
              <h3 className="text-sm font-bold text-white">Sunny — Biological Epoch 1 Turnaround & Expression Grid</h3>
              <p className="text-xs text-slate-400">WF-FOUNDRY-CHAR (Front, Side, Back, 3/4 + 6 Key Expressions)</p>
            </div>
            <span className="text-xs font-mono text-emerald-400 bg-emerald-950/60 px-2.5 py-1 rounded border border-emerald-800">
              IP-Adapter Face Locked (Weight: 0.82)
            </span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            {['Front View', 'Side 90°', '3/4 Dynamic', 'Back Outfit'].map((view, i) => (
              <div key={view} className="relative aspect-[3/4] bg-studio-950 rounded-lg overflow-hidden border border-studio-800">
                <img
                  src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=400&auto=format&fit=crop"
                  alt={view}
                  className="w-full h-full object-cover"
                />
                <div className="absolute bottom-2 left-2 right-2 bg-studio-950/80 backdrop-blur rounded px-2 py-1 text-[11px] font-mono text-center text-slate-200 border border-studio-700">
                  {view}
                </div>
              </div>
            ))}
          </div>

          <h4 className="text-xs font-semibold text-slate-300 mb-2">Canonical 6-Expression Calibration</h4>
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
            {['Neutral Cautious', 'Cynical Smirk', 'Fear Shock', 'Combat Focus', 'Exhausted', 'Shadow Resonance'].map((exp) => (
              <div key={exp} className="p-2 bg-studio-950 rounded border border-studio-800 text-center text-[10px] font-mono text-slate-300">
                {exp}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Environments View */}
      {subTab === 'environments' && (
        <div className="col-span-12 bg-studio-900 border border-studio-800 rounded-xl p-5 shadow-xl">
          <h3 className="text-sm font-bold text-white mb-2">Environment & Atmosphere Matrix</h3>
          <p className="text-xs text-slate-400 mb-4">WF-SHOT-ENV Lighting, Weather, and Atmospheric Particles</p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              { title: 'Weather Setup', val: 'Overcast with Slum Ash Haze', tags: ['slum_haze', 'overcast_sky', 'dust_motes'] },
              { title: 'Time of Day', val: 'Mid-Morning (10:30 AM)', tags: ['morning_sun', 'harsh_shadows', 'angular_light'] },
              { title: 'Atmospheric Grade', val: 'Cinematic Desaturated Blue-Grey', tags: ['cool_shadows', 'warm_highlights', 'film_grain'] }
            ].map((env) => (
              <div key={env.title} className="p-4 bg-studio-950 rounded-lg border border-studio-800 space-y-2">
                <span className="text-xs font-semibold text-indigo-400">{env.title}</span>
                <p className="text-xs font-bold text-white">{env.val}</p>
                <div className="flex flex-wrap gap-1">
                  {env.tags.map((t) => (
                    <span key={t} className="text-[10px] font-mono px-2 py-0.5 rounded bg-studio-900 text-slate-300 border border-studio-800">
                      {t}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
