"""
Novel Visualiser — FastAPI Application & Review Endpoints

Provides REST endpoints for:
- Project creation and overview
- Chapter management
- Human Review Queue (Entity approval, Visual Reference approval, QC triage)
- Spoiler-Safe Knowledge queries & Hierarchical Asset Resolver
- Pipeline orchestration trigger and status
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import (
    AssetVersion,
    Chapter,
    ChapterStatus,
    Entity,
    EntityAlias,
    GenerationJob,
    JobStatus,
    Project,
    ResearchClaim,
    Shot,
    ShotStatus,
    VisualRef,
)
from app.database.session import get_db, init_db
from app.orchestrator.resolver import get_safe_knowledge, resolve_asset_for_scene

logger = logging.getLogger("novel_visualiser")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and media directories on startup."""
    settings = get_settings()
    settings.project.ensure_directories()
    init_db()
    logger.info("Novel Visualiser service initialized successfully.")
    yield


app = FastAPI(
    title="Novel Visualiser API",
    description="Stateful, database-centric production pipeline converting novels to cinematic video.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic Request / Response models
# ---------------------------------------------------------------------------

class ProjectCreate(BaseModel):
    name: str = Field(..., description="Project name")
    novel_title: str = Field(..., description="Title of the source novel")
    author: Optional[str] = Field(None, description="Author name")
    description: Optional[str] = Field(None, description="Short project overview")


class ProjectSummary(BaseModel):
    id: str
    name: str
    novel_title: str
    author: Optional[str] = None
    chapters_count: int = 0
    entities_count: int = 0
    shots_count: int = 0


class ChapterCreate(BaseModel):
    chapter_number: int
    title: Optional[str] = None
    raw_text: str


class CharacterResearchRequest(BaseModel):
    character_name: str
    wiki_subdomain: Optional[str] = None


class EntityApprovalAction(BaseModel):
    entity_id: str
    approved: bool
    canonical_name_override: Optional[str] = None


class VisualRefApprovalAction(BaseModel):
    visual_ref_id: str
    approved: bool
    notes: Optional[str] = None


class ShotRetryAction(BaseModel):
    shot_id: str
    new_seed: Optional[int] = None
    prompt_override: Optional[str] = None


class ShotBudgetRequest(BaseModel):
    text: str
    scene_type: str = "dialogue"
    min_shots: int = 1
    max_shots: int = 20


class QCEvaluateRequest(BaseModel):
    image_path: str
    reference_image_path: Optional[str] = None
    checklist: Optional[list[str]] = None


# ---------------------------------------------------------------------------
# Health & Status
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check(db: Session = Depends(get_db)):
    """Check API and database health."""
    try:
        db.execute(select(func.count(Project.id))).scalar()
        db_status = "connected"
    except Exception as exc:
        db_status = f"unhealthy: {exc}"
    return {
        "status": "online",
        "database": db_status,
        "version": "0.1.0",
    }


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

@app.post("/projects", response_model=ProjectSummary, status_code=status.HTTP_201_CREATED, tags=["Projects"])
def create_project(data: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new novel project."""
    project = Project(
        name=data.name,
        novel_title=data.novel_title,
        author=data.author,
        description=data.description,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return ProjectSummary(
        id=project.id,
        name=project.name,
        novel_title=project.novel_title,
        author=project.author,
        chapters_count=0,
        entities_count=0,
        shots_count=0,
    )


@app.get("/projects", response_model=list[ProjectSummary], tags=["Projects"])
def list_projects(db: Session = Depends(get_db)):
    """List all projects with item counts."""
    projects = db.execute(select(Project)).scalars().all()
    results = []
    for p in projects:
        ch_count = db.execute(select(func.count(Chapter.id)).where(Chapter.project_id == p.id)).scalar() or 0
        ent_count = db.execute(select(func.count(Entity.id)).where(Entity.project_id == p.id)).scalar() or 0
        shot_count = db.execute(select(func.count(Shot.id)).join(Chapter).where(Chapter.project_id == p.id)).scalar() or 0
        results.append(
            ProjectSummary(
                id=p.id,
                name=p.name,
                novel_title=p.novel_title,
                author=p.author,
                chapters_count=ch_count,
                entities_count=ent_count,
                shots_count=shot_count,
            )
        )
    return results


@app.get("/projects/{project_id}", tags=["Projects"])
def get_project_details(project_id: str, db: Session = Depends(get_db)):
    """Get full details of a single project."""
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    chapters = db.execute(
        select(Chapter.id, Chapter.chapter_number, Chapter.title, Chapter.word_count, Chapter.status)
        .where(Chapter.project_id == project_id)
        .order_by(Chapter.chapter_number)
    ).all()

    return {
        "id": project.id,
        "name": project.name,
        "novel_title": project.novel_title,
        "author": project.author,
        "description": project.description,
        "chapters": [
            {
                "id": c.id,
                "chapter_number": c.chapter_number,
                "title": c.title,
                "word_count": c.word_count,
                "status": c.status.value if hasattr(c.status, "value") else str(c.status),
            }
            for c in chapters
        ],
    }


# ---------------------------------------------------------------------------
# Chapters
# ---------------------------------------------------------------------------

@app.post("/projects/{project_id}/chapters", status_code=status.HTTP_201_CREATED, tags=["Chapters"])
def add_chapter(project_id: str, data: ChapterCreate, db: Session = Depends(get_db)):
    """Add a chapter raw text to a project."""
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    existing = db.execute(
        select(Chapter).where(Chapter.project_id == project_id, Chapter.chapter_number == data.chapter_number)
    ).scalar_one_or_none()
    if existing:
        raise HTTPException(status_code=400, detail=f"Chapter {data.chapter_number} already exists")

    words = len(data.raw_text.split())
    chapter = Chapter(
        project_id=project_id,
        chapter_number=data.chapter_number,
        title=data.title or f"Chapter {data.chapter_number}",
        raw_text=data.raw_text,
        word_count=words,
        status=ChapterStatus.UNPARSED,
    )
    db.add(chapter)
    db.commit()
    db.refresh(chapter)

    return {
        "id": chapter.id,
        "chapter_number": chapter.chapter_number,
        "title": chapter.title,
        "word_count": chapter.word_count,
        "status": chapter.status.value,
    }


# ---------------------------------------------------------------------------
# Spoiler-Gate & Hierarchical Resolver
# ---------------------------------------------------------------------------

@app.get("/entities/{entity_id}/safe-knowledge", tags=["Intelligence"])
def query_safe_knowledge(
    entity_id: str,
    chapter_number: int = Query(..., ge=1, description="Current chapter boundary"),
    db: Session = Depends(get_db),
):
    """
    Query knowledge for an entity strictly bounded by chapter_number.
    Ensures zero spoiler leakage from future chapters.
    """
    try:
        return get_safe_knowledge(db, entity_id=entity_id, current_chapter=chapter_number)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/entities/{entity_id}/resolved-visual", tags=["Intelligence"])
def query_resolved_visual(
    entity_id: str,
    chapter_number: int = Query(..., ge=1, description="Target chapter number"),
    arc_id: Optional[str] = Query(None, description="Optional story arc UUID"),
    scene_id: Optional[str] = Query(None, description="Optional scene UUID"),
    db: Session = Depends(get_db),
):
    """
    Resolve hierarchical visual description:
    Project Canonical ⊕ Arc Default ⊕ Chapter Delta ⊕ Scene Override
    """
    try:
        return resolve_asset_for_scene(
            db=db,
            entity_id=entity_id,
            arc_id=arc_id,
            chapter_number=chapter_number,
            scene_id=scene_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ---------------------------------------------------------------------------
# Story Intelligence & World Architecture (Phase 2)
# ---------------------------------------------------------------------------

@app.post("/projects/{project_id}/chapters/{chapter_number}/analyze", tags=["Story Intelligence"])
def analyze_chapter_endpoint(
    project_id: str,
    chapter_number: int,
    db: Session = Depends(get_db),
):
    """Run Story Intelligence on a chapter (pre-tagging, scene breaks, events, visual deltas)."""
    from app.agents.story_intel import StoryIntelligenceAgent
    from app.agents.world_architect import WorldArchitectAgent

    chapter = db.execute(
        select(Chapter).where(
            Chapter.project_id == project_id,
            Chapter.chapter_number == chapter_number,
        )
    ).scalar_one_or_none()

    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    story_agent = StoryIntelligenceAgent()
    analysis = story_agent.process_chapter(db, chapter.id)

    world_agent = WorldArchitectAgent()
    reconciliation = world_agent.process_chapter_analysis(db, analysis)

    return {
        "chapter_number": chapter_number,
        "scenes_count": len(analysis.scenes),
        "events_count": len(analysis.events),
        "visual_state_deltas": len(analysis.visual_state_deltas),
        "reconciliation": reconciliation,
        "analysis": analysis.model_dump(),
    }


@app.get("/projects/{project_id}/chapters/{chapter_number}/scenes", tags=["Story Intelligence"])
def get_chapter_scenes_endpoint(
    project_id: str,
    chapter_number: int,
    db: Session = Depends(get_db),
):
    """Get all scenes, passage boundaries, and plot events for a chapter."""
    from app.database.models import Scene

    chapter = db.execute(
        select(Chapter).where(
            Chapter.project_id == project_id,
            Chapter.chapter_number == chapter_number,
        )
    ).scalar_one_or_none()

    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    scenes = db.execute(
        select(Scene).where(Scene.chapter_id == chapter.id).order_by(Scene.scene_number)
    ).scalars().all()

    return {
        "chapter_number": chapter_number,
        "chapter_title": chapter.title,
        "scenes": [
            {
                "id": sc.id,
                "scene_number": sc.scene_number,
                "time_of_day": sc.time_of_day,
                "weather": sc.weather,
                "mood": sc.mood,
                "description": sc.description,
                "events": [
                    {
                        "id": ev.id,
                        "type": ev.event_type.value,
                        "description": ev.description,
                        "importance": ev.importance.value,
                    }
                    for ev in sc.events
                ],
            }
            for sc in scenes
        ],
    }


# ---------------------------------------------------------------------------
# Research Engine (Phase 1)
# ---------------------------------------------------------------------------

@app.post("/projects/{project_id}/research/character", tags=["Research Engine"])
def research_character_endpoint(
    project_id: str,
    data: CharacterResearchRequest,
    db: Session = Depends(get_db),
):
    """Trigger wiki scraping & claim extraction for a character."""
    from app.agents.research_agent import ResearchAgent

    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    subdomain = data.wiki_subdomain or project.novel_title.lower().replace(" ", "")
    agent = ResearchAgent(wiki_subdomain=subdomain)
    try:
        return agent.research_character(db, project_id=project_id, character_query=data.character_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/projects/{project_id}/research/arcs", tags=["Research Engine"])
def research_arcs_endpoint(
    project_id: str,
    wiki_subdomain: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Harvest story volumes/arcs from wiki and populate project arcs."""
    from app.agents.research_agent import ResearchAgent

    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    subdomain = wiki_subdomain or project.novel_title.lower().replace(" ", "")
    agent = ResearchAgent(wiki_subdomain=subdomain)
    try:
        arcs = agent.research_arcs(db, project_id=project_id)
        return {"project_id": project_id, "arcs_harvested": len(arcs), "arcs": arcs}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# ComfyUI Image Production (Phase 3)
# ---------------------------------------------------------------------------

class TurnaroundCompileRequest(BaseModel):
    positive_prompt: str
    negative_prompt: Optional[str] = "low quality, worst quality, blurry"
    seed: Optional[int] = None
    checkpoint: Optional[str] = "Wai_Illustrious_v16.safetensors"


class LocationCompileRequest(BaseModel):
    positive_prompt: str
    negative_prompt: Optional[str] = "low quality, worst quality, blurry"
    seed: Optional[int] = None
    checkpoint: Optional[str] = "Wai_Illustrious_v16.safetensors"


class IPAdapterCompileRequest(BaseModel):
    reference_image_filename: str
    positive_prompt: str
    negative_prompt: Optional[str] = "low quality, blurry, different face"
    ipadapter_weight: Optional[float] = 0.8
    seed: Optional[int] = None


class LockReferenceRequest(BaseModel):
    entity_id: str
    image_path: str
    seed: Optional[int] = None
    ref_type: Optional[str] = "turnaround_sheet"
    lora_config: Optional[dict] = None
    ip_adapter_config: Optional[dict] = None


@app.get("/comfyui/health", tags=["ComfyUI Worker"])
def comfyui_health_check():
    """Verify ComfyUI connection (local or Kaggle/Colab tunnel) and check GPU stats."""
    from app.services.comfyui_client import ComfyUIClient
    client = ComfyUIClient()
    return client.check_health()


@app.post("/comfyui/compile/turnaround", tags=["ComfyUI Worker"])
def compile_turnaround_endpoint(data: TurnaroundCompileRequest):
    """Compile WF-001 Character Design Turnaround payload from user-tested template."""
    from app.services.comfyui_client import ComfyUIWorkflowCompiler
    return ComfyUIWorkflowCompiler.compile_character_design(
        positive_prompt=data.positive_prompt,
        negative_prompt=data.negative_prompt or "low quality, worst quality",
        seed=data.seed,
        checkpoint=data.checkpoint or "Wai_Illustrious_v16.safetensors",
    )


@app.post("/comfyui/compile/location", tags=["ComfyUI Worker"])
def compile_location_endpoint(data: LocationCompileRequest):
    """Compile WF-003 Master Location Concept Plate payload from user-tested template."""
    from app.services.comfyui_client import ComfyUIWorkflowCompiler
    return ComfyUIWorkflowCompiler.compile_location_plate(
        positive_prompt=data.positive_prompt,
        negative_prompt=data.negative_prompt or "low quality, worst quality",
        seed=data.seed,
        checkpoint=data.checkpoint or "Wai_Illustrious_v16.safetensors",
    )


@app.post("/comfyui/compile/ipadapter", tags=["ComfyUI Worker"])
def compile_ipadapter_endpoint(data: IPAdapterCompileRequest):
    """Compile WF-002 IPAdapter Character Consistency payload from user-tested template."""
    from app.services.comfyui_client import ComfyUIWorkflowCompiler
    return ComfyUIWorkflowCompiler.compile_ipadapter_consistency(
        reference_image_filename=data.reference_image_filename,
        positive_prompt=data.positive_prompt,
        negative_prompt=data.negative_prompt or "low quality, blurry",
        ipadapter_weight=data.ipadapter_weight or 0.8,
        seed=data.seed,
    )


@app.post("/comfyui/lock-reference", tags=["ComfyUI Worker"])
def lock_reference_endpoint(data: LockReferenceRequest, db: Session = Depends(get_db)):
    """Lock an approved reference image & seed into visual_refs for visual consistency."""
    from app.services.comfyui_client import ComfyUIClient
    from app.database.models import RefType
    client = ComfyUIClient()
    ref_type = RefType.TURNAROUND_SHEET
    if data.ref_type == "expression_grid":
        ref_type = RefType.EXPRESSION_GRID
    elif data.ref_type == "location_plate":
        ref_type = RefType.LOCATION_PLATE

    ref = client.lock_canonical_visual_reference(
        db=db,
        entity_id=data.entity_id,
        image_path=data.image_path,
        ref_type=ref_type,
        seed=data.seed,
        lora_config=data.lora_config,
        ip_adapter_config=data.ip_adapter_config,
    )
    return {"status": "locked", "visual_ref_id": ref.id, "entity_id": ref.entity_id, "image_path": ref.image_path}


@app.post("/shots/calculate-budget", tags=["Scene Production"])
def calculate_shot_budget_endpoint(data: ShotBudgetRequest):
    """Calculate shot budget, duration, and pacing curve based on word count (0 tokens)."""
    from app.agents.scene_director import calculate_shot_budget
    return calculate_shot_budget(
        text_or_passages=data.text,
        scene_type=data.scene_type,
        min_shots=data.min_shots,
        max_shots=data.max_shots,
    )


@app.post("/chapters/{chapter_id}/direct-shots", tags=["Scene Production"])
def direct_chapter_shots_endpoint(chapter_id: str, db: Session = Depends(get_db)):
    """Run Scene Director Agent on a chapter and generate database Shot & ShotSubject records."""
    from app.agents.scene_director import SceneDirectorAgent
    agent = SceneDirectorAgent()
    shots = agent.direct_chapter(db, chapter_id)
    return {
        "status": "success",
        "chapter_id": chapter_id,
        "shots_created": len(shots),
        "shots": [
            {
                "id": s.id,
                "shot_number": s.shot_number,
                "workflow_id": s.workflow_id,
                "shot_type": s.shot_type,
                "camera_angle": s.camera_angle,
                "motion_tier": s.motion_tier.value if hasattr(s.motion_tier, "value") else str(s.motion_tier),
                "subjects_count": len(s.subjects),
            }
            for s in shots
        ],
    }


@app.post("/shots/{shot_id}/compile-prompt", tags=["Scene Production"])
def compile_shot_prompt_endpoint(
    shot_id: str,
    chapter_number: int = Query(1, ge=1),
    arc_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Compile ComfyUI prompt & payload for a specific shot using Jinja2 & Asset Resolver."""
    shot = db.get(Shot, shot_id)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")

    from app.services.prompt_compiler import PromptCompiler
    compiler = PromptCompiler()
    compiled = compiler.compile_shot(
        shot=shot,
        db=db,
        chapter_number=chapter_number,
        arc_id=arc_id,
    )
    return compiled.model_dump()


@app.post("/qc/evaluate-image", tags=["Scene Production"])
def evaluate_image_qc_endpoint(data: QCEvaluateRequest):
    """Run 3-Gate QC on an image (Gate 1 technical, Gate 2 identity, Gate 3 semantic)."""
    from app.services.qc_service import QCService
    qc = QCService()
    g1 = qc.evaluate_gate1(data.image_path)
    g2 = qc.evaluate_gate2(data.image_path, reference_image_path=data.reference_image_path)
    g3 = qc.evaluate_gate3(data.image_path, checklist=data.checklist or ["character_visible"])
    overall = g1["passed"] and g2["passed"] and g3["passed"]
    return {
        "overall_passed": overall,
        "gate1": g1,
        "gate2": g2,
        "gate3": g3,
        "recommended_action": "approve" if overall else "retry_new_seed",
    }


# ---------------------------------------------------------------------------
# Human Review Gate
# ---------------------------------------------------------------------------

@app.get("/projects/{project_id}/review-queue", tags=["Human Review"])
def get_review_queue(project_id: str, db: Session = Depends(get_db)):
    """Fetch all pending items requiring human approval."""
    # 1. Unapproved visual reference sheets
    pending_refs = db.execute(
        select(VisualRef, Entity.canonical_name)
        .join(Entity, VisualRef.entity_id == Entity.id)
        .where(Entity.project_id == project_id, VisualRef.approved.is_(False))
    ).all()

    # 2. Shots that failed automated QC
    failed_shots = db.execute(
        select(Shot, Chapter.chapter_number)
        .join(Chapter, Shot.chapter_id == Chapter.id)
        .where(Chapter.project_id == project_id, Shot.status == ShotStatus.FAILED)
    ).all()

    return {
        "pending_visual_refs": [
            {
                "id": ref.id,
                "entity_id": ref.entity_id,
                "entity_name": name,
                "ref_type": ref.ref_type.value if hasattr(ref.ref_type, "value") else str(ref.ref_type),
                "image_path": ref.image_path,
                "seed": ref.seed,
            }
            for ref, name in pending_refs
        ],
        "failed_qc_shots": [
            {
                "id": s.id,
                "chapter_number": ch_num,
                "shot_number": s.shot_number,
                "shot_type": s.shot_type,
                "status": s.status.value if hasattr(s.status, "value") else str(s.status),
            }
            for s, ch_num in failed_shots
        ],
    }


@app.post("/review/approve-ref", tags=["Human Review"])
def approve_visual_reference(action: VisualRefApprovalAction, db: Session = Depends(get_db)):
    """Approve or reject a canonical visual reference sheet."""
    ref = db.get(VisualRef, action.visual_ref_id)
    if not ref:
        raise HTTPException(status_code=404, detail="Visual reference not found")

    ref.approved = action.approved
    if action.approved:
        from datetime import datetime
        ref.approved_at = datetime.utcnow()
    db.commit()

    return {
        "status": "success",
        "visual_ref_id": ref.id,
        "approved": ref.approved,
    }


@app.post("/review/retry-shot", tags=["Human Review"])
def retry_failed_shot(action: ShotRetryAction, db: Session = Depends(get_db)):
    """Reset a failed shot for re-generation with an optional seed or prompt override."""
    shot = db.get(Shot, action.shot_id)
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")

    shot.status = ShotStatus.PLANNED
    db.commit()

    return {
        "status": "success",
        "shot_id": shot.id,
        "new_status": shot.status.value,
    }


# ---------------------------------------------------------------------------
# Phase 5: Narration & Film Assembly Endpoints
# ---------------------------------------------------------------------------

@app.post("/chapters/{chapter_id}/narrate", tags=["Narration & Audio"])
def narrate_chapter(chapter_id: str, db: Session = Depends(get_db)):
    """
    Generate Hinglish narration, TTS audio tracks, and dual subtitles for a chapter.
    """
    import uuid
    from pathlib import Path
    from app.database.models import (
        NarrationSegment,
        Passage,
        Scene,
        SubtitleFormat,
        SubtitleLanguage,
        SubtitleTrack,
    )
    from app.services.audio_engine import HinglishScriptAdapter, SubtitleGenerator, TTSEngine

    chapter = db.get(Chapter, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    settings = get_settings()
    tts_engine = TTSEngine()
    script_adapter = HinglishScriptAdapter()
    sub_gen = SubtitleGenerator()

    shots = db.execute(
        select(Shot).where(Shot.chapter_id == chapter.id).order_by(Shot.shot_number)
    ).scalars().all()

    segments_data = []
    created_segments = []

    for idx, shot in enumerate(shots):
        passage_text = ""
        if shot.source_passage_ids:
            passages = db.execute(
                select(Passage).where(Passage.id.in_(shot.source_passage_ids))
            ).scalars().all()
            if passages:
                passage_text = " ".join(p.text for p in passages)

        if not passage_text:
            scene = db.get(Scene, shot.scene_id) if shot.scene_id else None
            passage_text = scene.mood if scene and scene.mood else f"Shot {shot.shot_number} in chapter {chapter.chapter_number}"

        adapted = script_adapter.adapt_passage_to_segment(
            shot_id=shot.id,
            segment_index=idx,
            passage_text=passage_text,
            mood=shot.camera_motion_preset or "tense",
        )

        audio_filename = f"ch_{chapter.chapter_number:02d}_shot_{shot.shot_number:03d}_{shot.id[:8]}.mp3"
        try:
            audio_path = tts_engine.synthesize(adapted.hinglish_tts_script, audio_filename)
            dur_ms = tts_engine.get_audio_duration_ms(audio_path)
        except Exception as err:
            logger.warning("TTS error for shot %s: %s", shot.id, err)
            dur_ms = max(len(passage_text.split()) * 250, 3000)
            audio_path = None

        shot.audio_duration_ms = dur_ms

        seg_rec = NarrationSegment(
            id=str(uuid.uuid4()),
            shot_id=shot.id,
            chapter_id=chapter.id,
            scene_id=shot.scene_id,
            hinglish_tts_script=adapted.hinglish_tts_script,
            subtitle_en=adapted.subtitle_en,
            subtitle_hi=adapted.subtitle_hi,
            prosody_tag=adapted.prosody_tag,
            audio_path=str(audio_path) if audio_path else None,
            audio_duration_ms=dur_ms,
        )
        db.add(seg_rec)
        created_segments.append(seg_rec)

        segments_data.append({
            "shot_id": shot.id,
            "subtitle_en": adapted.subtitle_en,
            "subtitle_hi": adapted.subtitle_hi,
            "duration_ms": dur_ms,
        })

    db.commit()

    # Subtitles
    subs_dir = Path(settings.project.media_root) / "subtitles"
    subs_dir.mkdir(parents=True, exist_ok=True)
    ass_path = subs_dir / f"chapter_{chapter.chapter_number:02d}_dual.ass"
    srt_path = subs_dir / f"chapter_{chapter.chapter_number:02d}_en.srt"

    sub_gen.generate_dual_ass(segments_data, ass_path)
    sub_gen.generate_srt(segments_data, srt_path, lang="en")

    db.add(SubtitleTrack(
        id=str(uuid.uuid4()),
        chapter_id=chapter.id,
        language=SubtitleLanguage.DUAL,
        format=SubtitleFormat.ASS,
        file_path=str(ass_path),
    ))
    db.add(SubtitleTrack(
        id=str(uuid.uuid4()),
        chapter_id=chapter.id,
        language=SubtitleLanguage.EN,
        format=SubtitleFormat.SRT,
        file_path=str(srt_path),
    ))
    db.commit()

    return {
        "status": "success",
        "chapter_id": chapter.id,
        "chapter_number": chapter.chapter_number,
        "segments_count": len(created_segments),
        "subtitle_ass": str(ass_path),
        "subtitle_srt": str(srt_path),
    }


@app.post("/chapters/{chapter_id}/render", tags=["Film Assembly"])
def render_chapter_film(
    chapter_id: str,
    fps: int = Query(30, description="Frames per second"),
    db: Session = Depends(get_db),
):
    """
    Render the master cinematic film for a chapter using FFmpeg with motion, VFX, and dual subtitles.
    """
    from pathlib import Path
    from app.database.models import (
        NarrationSegment,
        SubtitleLanguage,
        SubtitleTrack,
    )
    from app.schemas.scene import FilmEditSpec
    from app.services.film_editor import FilmEditor

    chapter = db.get(Chapter, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    settings = get_settings()
    editor = FilmEditor()

    shots = db.execute(
        select(Shot).where(Shot.chapter_id == chapter.id).order_by(Shot.shot_number)
    ).scalars().all()

    jobs = db.execute(
        select(GenerationJob).where(GenerationJob.shot_id.in_([s.id for s in shots]))
    ).scalars().all()
    job_map = {j.shot_id: j for j in jobs}

    narration_segs = db.execute(
        select(NarrationSegment).where(NarrationSegment.chapter_id == chapter.id)
    ).scalars().all()
    narr_map = {n.shot_id: n for n in narration_segs}

    sub_track = db.execute(
        select(SubtitleTrack).where(
            SubtitleTrack.chapter_id == chapter.id,
            SubtitleTrack.language == SubtitleLanguage.DUAL,
        )
    ).scalars().first()

    shot_dicts = []
    for s in shots:
        job = job_map.get(s.id)
        rgb_p = job.output_rgb_path if job else None
        if not rgb_p or not Path(rgb_p).exists():
            continue
        narr = narr_map.get(s.id)
        shot_dicts.append({
            "shot_id": s.id,
            "rgb_path": rgb_p,
            "depth_path": job.output_depth_path if job else None,
            "audio_path": narr.audio_path if narr else None,
            "duration_ms": s.audio_duration_ms or 3000,
            "motion_tier": s.motion_tier.value if hasattr(s.motion_tier, "value") else str(s.motion_tier),
            "camera_motion": s.camera_motion_preset or "slow_zoom_in",
            "vfx_overlay": s.vfx_overlay_tag,
        })

    if not shot_dicts:
        raise HTTPException(
            status_code=400,
            detail="No generated shots with valid RGB images found for this chapter",
        )

    out_films_dir = Path(settings.project.media_root) / "rendered_films"
    out_films_dir.mkdir(parents=True, exist_ok=True)
    master_mp4 = out_films_dir / f"CH{chapter.chapter_number:02d}_Master.mp4"

    spec = FilmEditSpec(
        chapter_number=chapter.chapter_number,
        shots=shot_dicts,
        background_music_track=None,
        subtitle_en_path=sub_track.file_path if sub_track else None,
        subtitle_hi_path=sub_track.file_path if sub_track else None,
        output_path=str(master_mp4),
        resolution=(1920, 1080),
        fps=fps,
    )

    result_path = editor.assemble_film(spec)

    return {
        "status": "success",
        "chapter_id": chapter.id,
        "chapter_number": chapter.chapter_number,
        "film_path": str(result_path),
        "shots_count": len(shot_dicts),
        "filesize_mb": round(result_path.stat().st_size / (1024 * 1024), 2),
    }


@app.get("/chapters/{chapter_id}/film-status", tags=["Film Assembly"])
def get_chapter_film_status(chapter_id: str, db: Session = Depends(get_db)):
    """Check the film assembly and rendering status for a chapter."""
    from pathlib import Path
    chapter = db.get(Chapter, chapter_id)
    if not chapter:
        raise HTTPException(status_code=404, detail="Chapter not found")

    settings = get_settings()
    out_films_dir = Path(settings.project.media_root) / "rendered_films"
    master_mp4 = out_films_dir / f"CH{chapter.chapter_number:02d}_Master.mp4"

    exists = master_mp4.exists()
    return {
        "chapter_id": chapter.id,
        "chapter_number": chapter.chapter_number,
        "film_rendered": exists,
        "film_path": str(master_mp4) if exists else None,
        "filesize_mb": round(master_mp4.stat().st_size / (1024 * 1024), 2) if exists else 0.0,
    }


# ---------------------------------------------------------------------------
# 3-GPU Cluster Telemetry & Static Web Studio Mounts
# ---------------------------------------------------------------------------

@app.get("/cluster/status", tags=["3-GPU Cluster"])
def get_cluster_status():
    """Probe all 3 GPUs (Colab, GPU0, GPU1) and return live hardware telemetry."""
    import time
    import urllib.request
    import json
    from app.config import get_settings
    settings = get_settings()

    targets = [
        {
            "id": "colab",
            "name": "Google Colab Worker (gcgpu)",
            "url": settings.comfyui.colab_url or "https://gcgpu.krhimanshu.in",
            "role": "Environment & Wide Plates (WF-FOUNDRY-LOC)"
        },
        {
            "id": "gpu0",
            "name": "Kaggle Dual-GPU 0 (gpu0)",
            "url": settings.comfyui.gpu0_url or "https://gpu0.krhimanshu.in",
            "role": "Character Sheets & LoRA Turnaround (WF-FOUNDRY-CHAR)"
        },
        {
            "id": "gpu1",
            "name": "Kaggle Dual-GPU 1 (gpu1)",
            "url": settings.comfyui.gpu1_url or "https://gpu1.krhimanshu.in",
            "role": "Creatures, Regional Inpaint & YOLOv8 FaceDetailer"
        }
    ]

    results = []
    for t in targets:
        url = t["url"].rstrip("/")
        stats_url = f"{url}/system_stats"
        start_t = time.time()
        try:
            req = urllib.request.Request(stats_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=3.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latency = int((time.time() - start_t) * 1000)
                devices = data.get("devices", [])
                vram_total = 14.56
                vram_free = 14.46
                if devices:
                    dev = devices[0]
                    vram_total = dev.get("vram_total", 14.56 * (1024**3)) / (1024**3)
                    vram_free = dev.get("vram_free", 14.46 * (1024**3)) / (1024**3)
                vram_used = max(0.0, vram_total - vram_free)
                vram_pct = (vram_used / vram_total * 100.0) if vram_total > 0 else 0.0

                results.append({
                    "id": t["id"],
                    "name": t["name"],
                    "url": t["url"],
                    "online": True,
                    "role": t["role"],
                    "vramTotalGb": round(vram_total, 2),
                    "vramFreeGb": round(vram_free, 2),
                    "vramUsedGb": round(vram_used, 2),
                    "vramPercent": round(vram_pct, 1),
                    "queueRemaining": 0,
                    "latencyMs": latency
                })
        except Exception as e:
            results.append({
                "id": t["id"],
                "name": t["name"],
                "url": t["url"],
                "online": False,
                "role": t["role"],
                "vramTotalGb": 14.56,
                "vramFreeGb": 0.0,
                "vramUsedGb": 14.56,
                "vramPercent": 100.0,
                "queueRemaining": 0,
                "latencyMs": 999,
                "error": str(e)
            })

    return results


# Mount media and production frontend
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Media outputs
output_path = Path("output").resolve()
output_path.mkdir(parents=True, exist_ok=True)
app.mount("/output", StaticFiles(directory=str(output_path)), name="output")

# Web studio frontend
frontend_dist = Path("frontend/dist").resolve()
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")

