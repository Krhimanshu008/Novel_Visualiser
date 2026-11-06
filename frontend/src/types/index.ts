// Types for Novel Visualiser Studio

export type TabType = 'research' | 'foundry' | 'director' | 'cluster' | 'qa' | 'cinema';

export interface GpuStats {
  id: string;
  name: string;
  url: string;
  online: boolean;
  role: string;
  vramTotalGb: number;
  vramFreeGb: number;
  vramUsedGb: number;
  vramPercent: number;
  queueRemaining: number;
  activeJob?: string;
  latencyMs: number;
}

export interface CharacterDossier {
  id: string;
  name: string;
  novel: string;
  epoch: string;
  validFrom: number;
  validUntil: number;
  danbooruTags: string[];
  fluxPrompt: string;
  antiSpoilerNegative: string[];
  reasoningTrace: string;
  avatarUrl: string;
}

export interface LocationAsset {
  id: string;
  name: string;
  plates: {
    wide: string;
    depth: string;
    lineart: string;
    reverse: string;
  };
  lighting: string;
  environment: string;
}

export interface CreatureAsset {
  id: string;
  name: string;
  rank: 'Dormant' | 'Awakened' | 'Fallen' | 'Corrupted' | 'Great' | 'Cursed' | 'Unholy';
  powerClass: string;
  visualTraits: string[];
  prompt: string;
  imageUrl: string;
}

export interface BoundingBox {
  id: string;
  entityName: string;
  x: number; // 0 to 1 percentage
  y: number; // 0 to 1 percentage
  width: number; // 0 to 1 percentage
  height: number; // 0 to 1 percentage
  color: string;
  role: 'protagonist' | 'supporting' | 'creature' | 'prop';
  expression: string;
  action: string;
}

export interface ShotPlan {
  id: string;
  sceneId: string;
  shotNumber: number;
  shotType: 'extreme_wide' | 'wide' | 'medium' | 'close_up' | 'extreme_close_up';
  cameraAngle: 'eye_level' | 'low_angle' | 'high_angle' | 'dutch_angle' | 'over_shoulder';
  durationSeconds: number;
  dialogue?: string;
  narration?: string;
  locationId: string;
  boxes: BoundingBox[];
  assignedGpu: string;
  status: 'pending' | 'rendering' | 'passed_qc' | 'failed_qc' | 'inpaint_retry';
  renderedImageUrl?: string;
  qcScore?: number;
}

export interface SceneBeat {
  id: string;
  sceneNumber: number;
  title: string;
  summary: string;
  emotionalTone: string;
  lightingSetup: string;
  shots: ShotPlan[];
}

export interface QaAuditResult {
  shotId: string;
  overallScore: number;
  gate1_technical: {
    passed: boolean;
    blurLaplacian: number;
    brightnessMean: number;
    contrastStd: number;
  };
  gate2_identity: {
    passed: boolean;
    wd14MatchScore: number;
    faceSimScore: number;
    detectedTags: string[];
    missingMandatoryTags: string[];
  };
  gate3_vlm: {
    passed: boolean;
    reasoningModel: string;
    anatomicalIntegrity: string;
    lightingConsistency: string;
    defectsDetected: string[];
    recommendedFixAction: 'approve' | 'differential_inpaint' | 'reseed';
  };
  inpaintBoundingBox?: {
    x: number;
    y: number;
    w: number;
    h: number;
    defectTarget: string;
  };
}

