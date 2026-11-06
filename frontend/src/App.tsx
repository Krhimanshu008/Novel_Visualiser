import React, { useState, useEffect } from 'react';
import { TabType, GpuStats } from './types';
import { fetchClusterTelemetry } from './services/api';
import { Header } from './components/Header';
import { ResearchHubTab } from './components/tabs/ResearchHubTab';
import { AssetFoundryTab } from './components/tabs/AssetFoundryTab';
import { SceneDirectorTab } from './components/tabs/SceneDirectorTab';
import { GpuClusterTab } from './components/tabs/GpuClusterTab';
import { VisualQaTab } from './components/tabs/VisualQaTab';
import { CinemaSequencerTab } from './components/tabs/CinemaSequencerTab';

export function App() {
  const [currentTab, setCurrentTab] = useState<TabType>('director');
  const [gpus, setGpus] = useState<GpuStats[]>([
    {
      id: 'colab',
      name: 'Google Colab Worker (gcgpu)',
      url: 'https://gcgpu.krhimanshu.in',
      online: true,
      role: 'Environment & Wide Plates (WF-FOUNDRY-LOC)',
      vramTotalGb: 14.56,
      vramFreeGb: 14.46,
      vramUsedGb: 0.10,
      vramPercent: 1.0,
      queueRemaining: 0,
      latencyMs: 142
    },
    {
      id: 'gpu0',
      name: 'Kaggle Dual-GPU 0 (gpu0)',
      url: 'https://gpu0.krhimanshu.in',
      online: true,
      role: 'Character Sheets & LoRA Turnaround (WF-FOUNDRY-CHAR)',
      vramTotalGb: 14.56,
      vramFreeGb: 14.46,
      vramUsedGb: 0.10,
      vramPercent: 1.0,
      queueRemaining: 0,
      latencyMs: 118
    },
    {
      id: 'gpu1',
      name: 'Kaggle Dual-GPU 1 (gpu1)',
      url: 'https://gpu1.krhimanshu.in',
      online: true,
      role: 'Creatures, Regional Inpaint & YOLOv8 FaceDetailer',
      vramTotalGb: 14.56,
      vramFreeGb: 14.46,
      vramUsedGb: 0.10,
      vramPercent: 1.0,
      queueRemaining: 0,
      latencyMs: 125
    }
  ]);

  const loadClusterStats = async () => {
    try {
      const live = await fetchClusterTelemetry();
      if (live && live.length > 0) {
        setGpus(live);
      }
    } catch (e) {
      console.warn('Telemetry poll error', e);
    }
  };

  useEffect(() => {
    loadClusterStats();
    const interval = setInterval(loadClusterStats, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-studio-950 text-slate-100 flex flex-col font-sans">
      <Header
        currentTab={currentTab}
        onTabChange={setCurrentTab}
        gpus={gpus}
        projectTitle="Shadow Slave"
        chapterTitle="Chapter 1: Nightmarish Beginning"
      />

      <main className="flex-1 overflow-y-auto">
        {currentTab === 'research' && <ResearchHubTab />}
        {currentTab === 'foundry' && <AssetFoundryTab />}
        {currentTab === 'director' && <SceneDirectorTab />}
        {currentTab === 'cluster' && <GpuClusterTab gpus={gpus} onRefresh={loadClusterStats} />}
        {currentTab === 'qa' && <VisualQaTab />}
        {currentTab === 'cinema' && <CinemaSequencerTab />}
      </main>
    </div>
  );
}

export default App;
