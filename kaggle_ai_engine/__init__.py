"""
Novel Visualiser — Kaggle AI Engine

Leverages Kaggle's $10/day Model Proxy allowance to power advanced visual reasoning,
frontier story intelligence, character identity locking, and multimodal quality control.
"""

from kaggle_ai_engine.config import KaggleAISettings, get_kaggle_settings
from kaggle_ai_engine.consistency_architect import (
    CharacterIdentityCard,
    LocationPlateSpec,
    OutfitCard,
    VisualConsistencyArchitect,
)
from kaggle_ai_engine.integration import KagglePipelineBridge
from kaggle_ai_engine.proxy_client import KaggleModelProxyClient
from kaggle_ai_engine.quota_manager import KaggleQuotaManager, QuotaReport, QuotaTier
from kaggle_ai_engine.semantic_qc import KaggleSemanticQCInspector, SemanticQCVerdict
from kaggle_ai_engine.story_extractor import (
    ChapterAnalysisResult,
    ExtractedScene,
    ExtractedVisualDelta,
    KaggleStoryExtractor,
)
from kaggle_ai_engine.task_runner import KaggleBenchmarkTaskRunner

__all__ = [
    "KaggleAISettings",
    "get_kaggle_settings",
    "KaggleQuotaManager",
    "QuotaReport",
    "QuotaTier",
    "KaggleModelProxyClient",
    "VisualConsistencyArchitect",
    "CharacterIdentityCard",
    "OutfitCard",
    "LocationPlateSpec",
    "KaggleStoryExtractor",
    "ChapterAnalysisResult",
    "ExtractedScene",
    "ExtractedVisualDelta",
    "KaggleSemanticQCInspector",
    "SemanticQCVerdict",
    "KaggleBenchmarkTaskRunner",
    "KagglePipelineBridge",
]

