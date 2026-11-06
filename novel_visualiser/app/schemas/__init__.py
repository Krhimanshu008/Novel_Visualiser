"""Novel Visualiser Pydantic Schemas."""

from app.schemas.story import *
from app.schemas.world import *
from app.schemas.scene import *
from app.schemas.research import *
from app.schemas.visual_context import (
    VisualStyleProfile,
    DanbooruIdentityLock,
    OutfitCard,
    LocationPlateSpec,
    SubjectVisualState,
    ShotVisualContext,
    PromptBlockCollection,
)
from app.schemas.arc_hierarchy import (
    ArcType,
    StoryTimeline,
    ParallelThread,
    ArcNode,
    SceneBudgetSpec,
    ChapterBudgetSpec,
)
from app.schemas.environment_subagent import (
    TimeOfDayEnum,
    WeatherConditionEnum,
    EnvironmentalDeepObservation,
    LightingRigSpec,
    EnvironmentalAtmosphereCard,
    ShotEnvironmentAnchor,
)
from app.schemas.creature_subagent import (
    CreatureRankEnum,
    CreatureClassEnum,
    CreatureTaxonomyEnum,
    CreatureDeepObservation,
    CreatureScaleSpec,
    SingleBestiaryPlateSpec,
    CreatureCard,
    ItemPropCard,
    ShotCreatureAnchor,
)
from app.schemas.compositing_subagent import (
    CompositingModeEnum,
    BoundingBox2D,
    ShotCompositionSpec,
    RenderedShotArtifact,
)
from app.schemas.orchestrator import (
    SceneBreakdownBeat,
    SceneOrchestrationPlan,
    ChapterProductionPlan,
)
from app.schemas.qc import (
    QCVerdictEnum,
    Gate1TagAuditResult,
    Gate2OpenCVMetricResult,
    Gate3VisionAuditResult,
    ComprehensiveQCAudit,
)

