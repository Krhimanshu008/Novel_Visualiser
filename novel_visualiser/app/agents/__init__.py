"""Novel Visualiser — Autonomous Continuity Subagents.

This package houses the autonomous subagents responsible for visual worldbuilding,
spatial anchoring, character identity locking, and quality assurance:

Subagents:
- LocationContinuitySubagent: Manages environment memory, multi-angle 4-plate matrices,
  OpenCV 3D depth/lineart structural guides, and camera angle matching across chapters.
- CharacterContinuitySubagent: Manages immutable biological identity locks, socioeconomic
  poverty wear / torn outfits, 4-sheet turnaround/expression matrices, and inpainting anchors.
- SceneDirectorAgent: Orchestrator agent coordinating shot budgeting, narration, and camera choreography.
- QCContinuityAgent: Multi-gate validation agent enforcing identity, tag adherence, and sharpness.
"""

from __future__ import annotations

from app.agents.location_subagent import LocationContinuitySubagent
from app.agents.character_subagent import CharacterContinuitySubagent
from app.agents.biological_evolution_auditor import BiologicalEvolutionAuditor
from app.agents.environment_subagent import EnvironmentContinuitySubagent
from app.agents.creature_subagent import CreatureMonsterSubagent
from app.agents.compositing_subagent import CompositingSubagent

__all__ = [
    "LocationContinuitySubagent",
    "CharacterContinuitySubagent",
    "BiologicalEvolutionAuditor",
    "EnvironmentContinuitySubagent",
    "CreatureMonsterSubagent",
    "CompositingSubagent",
]

