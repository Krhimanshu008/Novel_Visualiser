"""
Novel Visualiser — SQLAlchemy 2.0+ Database Models

Four-layer schema for the AI-powered novel-to-video pipeline:

    Layer 1 — Story:      projects → arcs → chapters → passages → scenes → events
    Layer 2 — Research:   research_sources → research_claims / research_images
    Layer 3 — World:      entities → entity_aliases / asset_versions / visual_refs / entity_relationships
    Layer 4 — Production: shots → shot_subjects / generation_jobs / narration_segments / subtitle_tracks

All primary keys are UUID-4 strings (36-char) for SQLite compatibility.
Enum fields are stored as VARCHAR via ``sa.Enum(..., native_enum=False)``.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import List, Optional

import sqlalchemy as sa
from sqlalchemy import (
    Column,
    ForeignKey,
    Index,
    JSON,
    LargeBinary,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uuid() -> str:
    """Generate a new UUID-4 string for use as a default PK value."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Declarative Base
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    """Shared declarative base for all Novel Visualiser models."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
#  Enums
# ═══════════════════════════════════════════════════════════════════════════

# -- Layer 1: Story --------------------------------------------------------

class ChapterStatus(str, enum.Enum):
    """Processing status for a chapter."""
    UNPARSED = "unparsed"
    PARSED = "parsed"
    SCENES_EXTRACTED = "scenes_extracted"
    PRODUCTION_READY = "production_ready"
    COMPLETE = "complete"


class PassageType(str, enum.Enum):
    """Semantic type of a passage / paragraph."""
    NARRATION = "narration"
    DIALOGUE = "dialogue"
    DESCRIPTION = "description"
    ACTION = "action"


class EventType(str, enum.Enum):
    """Classification of a story event."""
    PLOT_BEAT = "plot_beat"
    CHARACTER_INTRO = "character_intro"
    BATTLE = "battle"
    DISCOVERY = "discovery"
    DIALOGUE_EXCHANGE = "dialogue_exchange"
    ENVIRONMENTAL = "environmental"
    POWER_AWAKENING = "power_awakening"


class Importance(str, enum.Enum):
    """Narrative importance tier."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CLIMAX = "climax"


# -- Layer 2: Research -----------------------------------------------------

class SourceType(str, enum.Enum):
    """Research source category."""
    FANDOM_WIKI = "fandom_wiki"
    OFFICIAL_ART = "official_art"
    COMMUNITY_ART = "community_art"
    PUBLICATION = "publication"
    WEB_ARTICLE = "web_article"


class LicenseStatus(str, enum.Enum):
    """Intellectual-property status of a source."""
    UNKNOWN = "unknown"
    PUBLIC_DOMAIN = "public_domain"
    FAIR_USE = "fair_use"
    RESTRICTED = "restricted"


class ClaimType(str, enum.Enum):
    """What a research claim is about."""
    APPEARANCE = "appearance"
    ABILITY = "ability"
    EQUIPMENT = "equipment"
    RELATIONSHIP = "relationship"
    LOCATION_FEATURE = "location_feature"
    TITLE = "title"
    ALIAS = "alias"
    BACKSTORY = "backstory"


class Confidence(str, enum.Enum):
    """Reliability grade of a claim."""
    CONFIRMED = "confirmed"
    LIKELY = "likely"
    SPECULATIVE = "speculative"


class ImageCategory(str, enum.Enum):
    """Category of a research image."""
    OFFICIAL_COVER = "official_cover"
    OFFICIAL_ILLUSTRATION = "official_illustration"
    COMMUNITY_ART = "community_art"
    SCREENSHOT = "screenshot"
    REFERENCE_PHOTO = "reference_photo"


# -- Layer 3: World --------------------------------------------------------

class EntityType(str, enum.Enum):
    """World-entity classification."""
    CHARACTER = "character"
    LOCATION = "location"
    CREATURE = "creature"
    ITEM = "item"
    POWER = "power"
    EFFECT = "effect"
    FACTION = "faction"


class AliasType(str, enum.Enum):
    """How an alias refers to an entity."""
    NAME = "name"
    TITLE = "title"
    NICKNAME = "nickname"
    DESCRIPTION_REFERENCE = "description_reference"
    CODENAME = "codename"


class AssetLevel(str, enum.Enum):
    """Specificity level for an asset version's visual attributes."""
    PROJECT_CANONICAL = "project_canonical"
    ARC_DEFAULT = "arc_default"
    CHAPTER_DELTA = "chapter_delta"
    SCENE_OVERRIDE = "scene_override"


class RefType(str, enum.Enum):
    """Type of visual-reference sheet."""
    TURNAROUND_SHEET = "turnaround_sheet"
    EXPRESSION_GRID = "expression_grid"
    LOCATION_PLATE = "location_plate"
    ITEM_REF = "item_ref"
    CREATURE_REF = "creature_ref"
    ACTION_POSE = "action_pose"


class RelationshipType(str, enum.Enum):
    """Inter-entity relationship kind."""
    ALLY = "ally"
    ENEMY = "enemy"
    FAMILY = "family"
    MENTOR = "mentor"
    SUBORDINATE = "subordinate"
    LOVER = "lover"
    RIVAL = "rival"
    CREATOR = "creator"
    MEMBER_OF = "member_of"


# -- Layer 4: Production ---------------------------------------------------

class MotionTier(str, enum.Enum):
    """Motion complexity tier for a shot."""
    STATIC_PARALLAX = "static_parallax"
    I2V_CLIMAX = "i2v_climax"


class ShotStatus(str, enum.Enum):
    """Pipeline status of a shot."""
    PLANNED = "planned"
    PROMPT_COMPILED = "prompt_compiled"
    GENERATING = "generating"
    QC_PENDING = "qc_pending"
    APPROVED = "approved"
    FAILED = "failed"


class JobStatus(str, enum.Enum):
    """Status of a ComfyUI generation job."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class ProsodyTag(str, enum.Enum):
    """TTS prosody hint for narration."""
    CALM_NARRATION = "calm_narration"
    TENSE_WHISPER = "tense_whisper"
    DRAMATIC_RISE = "dramatic_rise"
    INTENSE_ACTION = "intense_action"


class SubtitleLanguage(str, enum.Enum):
    """Subtitle language code."""
    EN = "en"
    HI = "hi"
    DUAL = "dual"


class SubtitleFormat(str, enum.Enum):
    """Subtitle file format."""
    ASS = "ass"
    SRT = "srt"


# ═══════════════════════════════════════════════════════════════════════════
#  Layer 1 — Story Models
# ═══════════════════════════════════════════════════════════════════════════

class Project(Base):
    """Top-level novel project container."""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    novel_title: Mapped[str] = mapped_column(sa.String(500), nullable=False)
    author: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    arcs: Mapped[List["Arc"]] = relationship(
        back_populates="project", cascade="all, delete-orphan",
    )
    chapters: Mapped[List["Chapter"]] = relationship(
        back_populates="project", cascade="all, delete-orphan",
    )
    entities: Mapped[List["Entity"]] = relationship(
        back_populates="project", cascade="all, delete-orphan",
    )
    research_sources: Mapped[List["ResearchSource"]] = relationship(
        back_populates="project", cascade="all, delete-orphan",
    )
    entity_relationships: Mapped[List["EntityRelationship"]] = relationship(
        back_populates="project", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Project id={self.id!r} name={self.name!r}>"


class Arc(Base):
    """A narrative arc spanning one or more chapters."""

    __tablename__ = "arcs"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    project_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False,
    )
    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)
    arc_number: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    start_chapter: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    end_chapter: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    project: Mapped["Project"] = relationship(back_populates="arcs")
    chapters: Mapped[List["Chapter"]] = relationship(back_populates="arc")
    asset_versions: Mapped[List["AssetVersion"]] = relationship(back_populates="arc")

    def __repr__(self) -> str:
        return f"<Arc id={self.id!r} name={self.name!r} arc_number={self.arc_number}>"


class Chapter(Base):
    """A single chapter of the novel."""

    __tablename__ = "chapters"
    __table_args__ = (
        UniqueConstraint("project_id", "chapter_number", name="uq_chapter_number_per_project"),
    )

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    project_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False,
    )
    arc_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), ForeignKey("arcs.id", ondelete="SET NULL"), nullable=True,
    )
    chapter_number: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    title: Mapped[str] = mapped_column(sa.String(500), nullable=False)
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    word_count: Mapped[int] = mapped_column(sa.Integer, default=0, nullable=False)
    status: Mapped[ChapterStatus] = mapped_column(
        sa.Enum(ChapterStatus, native_enum=False, length=30),
        default=ChapterStatus.UNPARSED,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    project: Mapped["Project"] = relationship(back_populates="chapters")
    arc: Mapped[Optional["Arc"]] = relationship(back_populates="chapters")
    passages: Mapped[List["Passage"]] = relationship(
        back_populates="chapter", cascade="all, delete-orphan",
    )
    scenes: Mapped[List["Scene"]] = relationship(
        back_populates="chapter", cascade="all, delete-orphan",
    )
    events: Mapped[List["Event"]] = relationship(
        back_populates="chapter", cascade="all, delete-orphan",
    )
    shots: Mapped[List["Shot"]] = relationship(
        back_populates="chapter", cascade="all, delete-orphan",
    )
    narration_segments: Mapped[List["NarrationSegment"]] = relationship(
        back_populates="chapter", cascade="all, delete-orphan",
    )
    subtitle_tracks: Mapped[List["SubtitleTrack"]] = relationship(
        back_populates="chapter", cascade="all, delete-orphan",
    )
    asset_versions: Mapped[List["AssetVersion"]] = relationship(back_populates="chapter")

    def __repr__(self) -> str:
        return (
            f"<Chapter id={self.id!r} chapter_number={self.chapter_number} "
            f"title={self.title!r} status={self.status.value!r}>"
        )


class Passage(Base):
    """A single paragraph or dialogue beat within a chapter."""

    __tablename__ = "passages"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    chapter_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False,
    )
    paragraph_index: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    passage_type: Mapped[PassageType] = mapped_column(
        sa.Enum(PassageType, native_enum=False, length=20),
        default=PassageType.NARRATION,
        nullable=False,
    )
    tagged_entity_ids: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True, doc="List of entity UUIDs detected by the pre-tagger.",
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    chapter: Mapped["Chapter"] = relationship(back_populates="passages")

    def __repr__(self) -> str:
        return (
            f"<Passage id={self.id!r} chapter_id={self.chapter_id!r} "
            f"paragraph_index={self.paragraph_index}>"
        )


class Scene(Base):
    """A logical scene within a chapter, bounded by passage indices."""

    __tablename__ = "scenes"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    chapter_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False,
    )
    scene_number: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    start_passage_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("passages.id"), nullable=False,
    )
    end_passage_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("passages.id"), nullable=False,
    )
    location_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), ForeignKey("entities.id", ondelete="SET NULL"), nullable=True,
    )
    time_of_day: Mapped[Optional[str]] = mapped_column(sa.String(50), nullable=True)
    weather: Mapped[Optional[str]] = mapped_column(sa.String(100), nullable=True)
    mood: Mapped[Optional[str]] = mapped_column(sa.String(100), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    shot_count_target: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    chapter: Mapped["Chapter"] = relationship(back_populates="scenes")
    start_passage: Mapped["Passage"] = relationship(foreign_keys=[start_passage_id])
    end_passage: Mapped["Passage"] = relationship(foreign_keys=[end_passage_id])
    location: Mapped[Optional["Entity"]] = relationship(
        foreign_keys=[location_id], back_populates="scenes_as_location",
    )
    events: Mapped[List["Event"]] = relationship(
        back_populates="scene", cascade="all, delete-orphan",
    )
    shots: Mapped[List["Shot"]] = relationship(
        back_populates="scene", cascade="all, delete-orphan",
    )
    narration_segments: Mapped[List["NarrationSegment"]] = relationship(
        back_populates="scene", cascade="all, delete-orphan",
    )
    asset_versions: Mapped[List["AssetVersion"]] = relationship(back_populates="scene")

    def __repr__(self) -> str:
        return (
            f"<Scene id={self.id!r} chapter_id={self.chapter_id!r} "
            f"scene_number={self.scene_number}>"
        )


class Event(Base):
    """A discrete plot event or beat within a scene."""

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    scene_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False,
    )
    chapter_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False,
    )
    event_type: Mapped[EventType] = mapped_column(
        sa.Enum(EventType, native_enum=False, length=30), nullable=False,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    source_passage_ids: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    characters_involved: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    importance: Mapped[Importance] = mapped_column(
        sa.Enum(Importance, native_enum=False, length=10),
        default=Importance.MEDIUM,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    scene: Mapped["Scene"] = relationship(back_populates="events")
    chapter: Mapped["Chapter"] = relationship(back_populates="events")

    def __repr__(self) -> str:
        return (
            f"<Event id={self.id!r} event_type={self.event_type.value!r} "
            f"importance={self.importance.value!r}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Layer 2 — Research Models
# ═══════════════════════════════════════════════════════════════════════════

class ResearchSource(Base):
    """An external source scraped or referenced for world-building data."""

    __tablename__ = "research_sources"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    project_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False,
    )
    source_type: Mapped[SourceType] = mapped_column(
        sa.Enum(SourceType, native_enum=False, length=20), nullable=False,
    )
    url: Mapped[str] = mapped_column(sa.String(2048), nullable=False)
    title: Mapped[str] = mapped_column(sa.String(500), nullable=False)
    scraped_at: Mapped[Optional[datetime]] = mapped_column(sa.DateTime, nullable=True)
    license_status: Mapped[LicenseStatus] = mapped_column(
        sa.Enum(LicenseStatus, native_enum=False, length=20),
        default=LicenseStatus.UNKNOWN,
        nullable=False,
    )
    raw_content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    project: Mapped["Project"] = relationship(back_populates="research_sources")
    claims: Mapped[List["ResearchClaim"]] = relationship(
        back_populates="source", cascade="all, delete-orphan",
    )
    images: Mapped[List["ResearchImage"]] = relationship(
        back_populates="source", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<ResearchSource id={self.id!r} title={self.title!r}>"


class ResearchClaim(Base):
    """A single factual claim extracted from a research source."""

    __tablename__ = "research_claims"
    __table_args__ = (
        Index("ix_claims_entity_chapter", "entity_id", "valid_from_chapter"),
    )

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    source_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), ForeignKey("research_sources.id", ondelete="SET NULL"), nullable=True,
    )
    entity_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), ForeignKey("entities.id", ondelete="SET NULL"), nullable=True,
    )
    claim_type: Mapped[ClaimType] = mapped_column(
        sa.Enum(ClaimType, native_enum=False, length=20), nullable=False,
    )
    claim_text: Mapped[str] = mapped_column(Text, nullable=False)
    structured_data: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, doc="Parsed visual attributes as key-value pairs.",
    )
    valid_from_chapter: Mapped[int] = mapped_column(sa.Integer, default=1, nullable=False)
    valid_to_chapter: Mapped[Optional[int]] = mapped_column(
        sa.Integer, nullable=True, doc="NULL means the claim is valid indefinitely.",
    )
    confidence: Mapped[Confidence] = mapped_column(
        sa.Enum(Confidence, native_enum=False, length=15),
        default=Confidence.LIKELY,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    source: Mapped["ResearchSource"] = relationship(back_populates="claims")
    entity: Mapped[Optional["Entity"]] = relationship(back_populates="research_claims")

    def __repr__(self) -> str:
        return (
            f"<ResearchClaim id={self.id!r} claim_type={self.claim_type.value!r} "
            f"confidence={self.confidence.value!r}>"
        )


class ResearchImage(Base):
    """An image associated with a research source."""

    __tablename__ = "research_images"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    source_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), ForeignKey("research_sources.id", ondelete="SET NULL"), nullable=True,
    )
    entity_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), ForeignKey("entities.id", ondelete="SET NULL"), nullable=True,
    )
    image_url: Mapped[str] = mapped_column(sa.String(2048), nullable=False)
    local_path: Mapped[Optional[str]] = mapped_column(sa.String(1024), nullable=True)
    category: Mapped[ImageCategory] = mapped_column(
        sa.Enum(ImageCategory, native_enum=False, length=30), nullable=False,
    )
    caption: Mapped[Optional[str]] = mapped_column(sa.String(1000), nullable=True)
    width: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    source: Mapped["ResearchSource"] = relationship(back_populates="images")
    entity: Mapped[Optional["Entity"]] = relationship(back_populates="research_images")

    def __repr__(self) -> str:
        return f"<ResearchImage id={self.id!r} category={self.category.value!r}>"


# ═══════════════════════════════════════════════════════════════════════════
#  Layer 3 — World Models
# ═══════════════════════════════════════════════════════════════════════════

class Entity(Base):
    """A world entity: character, location, creature, item, power, effect, or faction."""

    __tablename__ = "entities"
    __table_args__ = (
        Index("ix_entities_project", "project_id"),
    )

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    project_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False,
    )
    entity_type: Mapped[EntityType] = mapped_column(
        sa.Enum(EntityType, native_enum=False, length=15), nullable=False,
    )
    canonical_name: Mapped[str] = mapped_column(sa.String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    first_appearance_chapter: Mapped[Optional[int]] = mapped_column(
        sa.Integer, nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(sa.Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    project: Mapped["Project"] = relationship(back_populates="entities")
    aliases: Mapped[List["EntityAlias"]] = relationship(
        back_populates="entity", cascade="all, delete-orphan",
    )
    asset_versions: Mapped[List["AssetVersion"]] = relationship(
        back_populates="entity", cascade="all, delete-orphan",
    )
    visual_refs: Mapped[List["VisualRef"]] = relationship(
        back_populates="entity", cascade="all, delete-orphan",
    )
    research_claims: Mapped[List["ResearchClaim"]] = relationship(back_populates="entity")
    research_images: Mapped[List["ResearchImage"]] = relationship(back_populates="entity")
    shot_subjects: Mapped[List["ShotSubject"]] = relationship(back_populates="entity")
    scenes_as_location: Mapped[List["Scene"]] = relationship(
        back_populates="location", foreign_keys="[Scene.location_id]",
    )

    # entity_relationships handled via two separate backrefs below
    relationships_as_a: Mapped[List["EntityRelationship"]] = relationship(
        back_populates="entity_a", foreign_keys="[EntityRelationship.entity_id_a]",
    )
    relationships_as_b: Mapped[List["EntityRelationship"]] = relationship(
        back_populates="entity_b", foreign_keys="[EntityRelationship.entity_id_b]",
    )

    def __repr__(self) -> str:
        return (
            f"<Entity id={self.id!r} canonical_name={self.canonical_name!r} "
            f"type={self.entity_type.value!r}>"
        )


class EntityAlias(Base):
    """An alternate name, title, or nickname for an entity."""

    __tablename__ = "entity_aliases"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    entity_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False,
    )
    alias: Mapped[str] = mapped_column(sa.String(500), nullable=False)
    alias_type: Mapped[AliasType] = mapped_column(
        sa.Enum(AliasType, native_enum=False, length=25), nullable=False,
    )
    valid_from_chapter: Mapped[int] = mapped_column(sa.Integer, default=1, nullable=False)
    reveal_chapter: Mapped[Optional[int]] = mapped_column(
        sa.Integer, nullable=True,
        doc="Chapter in which the reader learns this alias belongs to the entity.",
    )
    is_primary: Mapped[bool] = mapped_column(sa.Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    entity: Mapped["Entity"] = relationship(back_populates="aliases")

    def __repr__(self) -> str:
        return f"<EntityAlias id={self.id!r} alias={self.alias!r}>"


class AssetVersion(Base):
    """A versioned set of visual attributes for an entity at a specific scope level."""

    __tablename__ = "asset_versions"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    entity_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False,
    )
    level: Mapped[AssetLevel] = mapped_column(
        sa.Enum(AssetLevel, native_enum=False, length=25), nullable=False,
    )
    arc_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), ForeignKey("arcs.id", ondelete="SET NULL"), nullable=True,
    )
    chapter_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), ForeignKey("chapters.id", ondelete="SET NULL"), nullable=True,
    )
    scene_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), ForeignKey("scenes.id", ondelete="SET NULL"), nullable=True,
    )
    visual_attributes: Mapped[dict] = mapped_column(
        JSON, nullable=False, doc="Key-value visual description dictionary.",
    )
    change_description: Mapped[Optional[str]] = mapped_column(
        sa.String(1000), nullable=True, doc="What changed and why.",
    )
    valid_from_chapter: Mapped[int] = mapped_column(sa.Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    entity: Mapped["Entity"] = relationship(back_populates="asset_versions")
    arc: Mapped[Optional["Arc"]] = relationship(back_populates="asset_versions")
    chapter: Mapped[Optional["Chapter"]] = relationship(back_populates="asset_versions")
    scene: Mapped[Optional["Scene"]] = relationship(back_populates="asset_versions")
    visual_refs: Mapped[List["VisualRef"]] = relationship(back_populates="asset_version")

    def __repr__(self) -> str:
        return (
            f"<AssetVersion id={self.id!r} entity_id={self.entity_id!r} "
            f"level={self.level.value!r}>"
        )


class VisualRef(Base):
    """A generated or curated visual reference image for an entity."""

    __tablename__ = "visual_refs"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    entity_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False,
    )
    asset_version_id: Mapped[Optional[str]] = mapped_column(
        sa.String(36), ForeignKey("asset_versions.id", ondelete="SET NULL"), nullable=True,
    )
    ref_type: Mapped[RefType] = mapped_column(
        sa.Enum(RefType, native_enum=False, length=25), nullable=False,
    )
    image_path: Mapped[str] = mapped_column(sa.String(1024), nullable=False)
    seed: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    lora_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    ip_adapter_config: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    face_embedding_blob: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary, nullable=True,
    )
    clip_embedding_blob: Mapped[Optional[bytes]] = mapped_column(
        LargeBinary, nullable=True,
    )
    approved: Mapped[bool] = mapped_column(sa.Boolean, default=False, nullable=False)
    approved_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime, nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    entity: Mapped["Entity"] = relationship(back_populates="visual_refs")
    asset_version: Mapped[Optional["AssetVersion"]] = relationship(
        back_populates="visual_refs",
    )

    def __repr__(self) -> str:
        return (
            f"<VisualRef id={self.id!r} entity_id={self.entity_id!r} "
            f"ref_type={self.ref_type.value!r} approved={self.approved}>"
        )


class EntityRelationship(Base):
    """A directed relationship between two entities."""

    __tablename__ = "entity_relationships"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    project_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False,
    )
    entity_id_a: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False,
    )
    entity_id_b: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False,
    )
    relationship_type: Mapped[RelationshipType] = mapped_column(
        sa.Enum(RelationshipType, native_enum=False, length=15), nullable=False,
    )
    description: Mapped[Optional[str]] = mapped_column(sa.String(1000), nullable=True)
    valid_from_chapter: Mapped[int] = mapped_column(sa.Integer, default=1, nullable=False)
    valid_to_chapter: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    project: Mapped["Project"] = relationship(back_populates="entity_relationships")
    entity_a: Mapped["Entity"] = relationship(
        back_populates="relationships_as_a", foreign_keys=[entity_id_a],
    )
    entity_b: Mapped["Entity"] = relationship(
        back_populates="relationships_as_b", foreign_keys=[entity_id_b],
    )

    def __repr__(self) -> str:
        return (
            f"<EntityRelationship id={self.id!r} "
            f"{self.entity_id_a!r} --[{self.relationship_type.value}]--> {self.entity_id_b!r}>"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Layer 4 — Production Models
# ═══════════════════════════════════════════════════════════════════════════

class Shot(Base):
    """A single camera shot within a scene, ready for image generation."""

    __tablename__ = "shots"
    __table_args__ = (
        Index("ix_shots_chapter", "chapter_id"),
    )

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    scene_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False,
    )
    chapter_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False,
    )
    shot_number: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    workflow_id: Mapped[str] = mapped_column(
        sa.String(20), nullable=False, doc="Workflow tag, e.g. 'WF-001' to 'WF-008'.",
    )
    shot_type: Mapped[str] = mapped_column(
        sa.String(30), nullable=False,
        doc="Shot framing: close_up, medium, wide, extreme_wide, two_shot, group, pov.",
    )
    camera_angle: Mapped[str] = mapped_column(
        sa.String(30), nullable=False,
        doc="Camera angle: eye_level, low_angle, high_angle, dutch_angle, birds_eye, worms_eye.",
    )
    lighting_override: Mapped[Optional[str]] = mapped_column(
        sa.String(200), nullable=True,
    )
    vfx_overlay_tag: Mapped[Optional[str]] = mapped_column(
        sa.String(100), nullable=True,
    )
    motion_tier: Mapped[MotionTier] = mapped_column(
        sa.Enum(MotionTier, native_enum=False, length=20),
        default=MotionTier.STATIC_PARALLAX,
        nullable=False,
    )
    camera_motion_preset: Mapped[Optional[str]] = mapped_column(
        sa.String(50), nullable=True,
        doc="E.g. slow_zoom_in, pan_left_to_right, dolly_up, fast_push_in.",
    )
    source_passage_ids: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    audio_duration_ms: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    status: Mapped[ShotStatus] = mapped_column(
        sa.Enum(ShotStatus, native_enum=False, length=20),
        default=ShotStatus.PLANNED,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    scene: Mapped["Scene"] = relationship(back_populates="shots")
    chapter: Mapped["Chapter"] = relationship(back_populates="shots")
    subjects: Mapped[List["ShotSubject"]] = relationship(
        back_populates="shot", cascade="all, delete-orphan",
    )
    generation_jobs: Mapped[List["GenerationJob"]] = relationship(
        back_populates="shot", cascade="all, delete-orphan",
    )
    narration_segments: Mapped[List["NarrationSegment"]] = relationship(
        back_populates="shot", cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<Shot id={self.id!r} scene_id={self.scene_id!r} "
            f"shot_number={self.shot_number} status={self.status.value!r}>"
        )


class ShotSubject(Base):
    """An entity placed within a shot with bounding box and pose metadata."""

    __tablename__ = "shot_subjects"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    shot_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("shots.id", ondelete="CASCADE"), nullable=False,
    )
    entity_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False,
    )
    region_box: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True, doc="[x1, y1, x2, y2] normalised 0-1.",
    )
    pose: Mapped[Optional[str]] = mapped_column(sa.String(200), nullable=True)
    expression: Mapped[Optional[str]] = mapped_column(sa.String(200), nullable=True)
    z_order: Mapped[int] = mapped_column(sa.Integer, default=0, nullable=False)

    # -- relationships --
    shot: Mapped["Shot"] = relationship(back_populates="subjects")
    entity: Mapped["Entity"] = relationship(back_populates="shot_subjects")

    def __repr__(self) -> str:
        return (
            f"<ShotSubject id={self.id!r} shot_id={self.shot_id!r} "
            f"entity_id={self.entity_id!r}>"
        )


class GenerationJob(Base):
    """A ComfyUI image-generation job tracked with three QC gates."""

    __tablename__ = "generation_jobs"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    shot_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("shots.id", ondelete="CASCADE"), nullable=False,
    )
    workflow_id: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    seed: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    comfyui_params: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True, doc="Full ComfyUI node parameter payload.",
    )
    output_rgb_path: Mapped[Optional[str]] = mapped_column(
        sa.String(1024), nullable=True,
    )
    output_depth_path: Mapped[Optional[str]] = mapped_column(
        sa.String(1024), nullable=True,
    )

    # QC gates
    qc_gate1_passed: Mapped[Optional[bool]] = mapped_column(sa.Boolean, nullable=True)
    qc_gate1_details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    qc_gate2_passed: Mapped[Optional[bool]] = mapped_column(sa.Boolean, nullable=True)
    qc_gate2_details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    qc_gate3_passed: Mapped[Optional[bool]] = mapped_column(sa.Boolean, nullable=True)
    qc_gate3_details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    qc_score: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)

    attempt_number: Mapped[int] = mapped_column(sa.Integer, default=1, nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        sa.Enum(JobStatus, native_enum=False, length=15),
        default=JobStatus.QUEUED,
        nullable=False,
    )
    error_message: Mapped[Optional[str]] = mapped_column(sa.String(2000), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(sa.DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(sa.DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    shot: Mapped["Shot"] = relationship(back_populates="generation_jobs")

    def __repr__(self) -> str:
        return (
            f"<GenerationJob id={self.id!r} shot_id={self.shot_id!r} "
            f"status={self.status.value!r} attempt={self.attempt_number}>"
        )


class NarrationSegment(Base):
    """A TTS narration segment aligned to a shot, with Hinglish script and subtitles."""

    __tablename__ = "narration_segments"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    shot_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("shots.id", ondelete="CASCADE"), nullable=False,
    )
    chapter_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False,
    )
    scene_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("scenes.id", ondelete="CASCADE"), nullable=False,
    )
    hinglish_tts_script: Mapped[str] = mapped_column(Text, nullable=False)
    subtitle_en: Mapped[str] = mapped_column(sa.String(2000), nullable=False)
    subtitle_hi: Mapped[str] = mapped_column(sa.String(2000), nullable=False)
    prosody_tag: Mapped[Optional[str]] = mapped_column(
        sa.String(30), nullable=True,
        doc="Prosody hint: calm_narration, tense_whisper, dramatic_rise, intense_action.",
    )
    audio_path: Mapped[Optional[str]] = mapped_column(sa.String(1024), nullable=True)
    audio_duration_ms: Mapped[Optional[int]] = mapped_column(sa.Integer, nullable=True)
    word_timestamps: Mapped[Optional[list]] = mapped_column(
        JSON, nullable=True, doc="WhisperX word-level timestamps.",
    )
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    shot: Mapped["Shot"] = relationship(back_populates="narration_segments")
    chapter: Mapped["Chapter"] = relationship(back_populates="narration_segments")
    scene: Mapped["Scene"] = relationship(back_populates="narration_segments")

    def __repr__(self) -> str:
        return f"<NarrationSegment id={self.id!r} shot_id={self.shot_id!r}>"


class SubtitleTrack(Base):
    """An exported subtitle file for a chapter."""

    __tablename__ = "subtitle_tracks"

    id: Mapped[str] = mapped_column(
        sa.String(36), primary_key=True, default=_uuid,
    )
    chapter_id: Mapped[str] = mapped_column(
        sa.String(36), ForeignKey("chapters.id", ondelete="CASCADE"), nullable=False,
    )
    language: Mapped[SubtitleLanguage] = mapped_column(
        sa.Enum(SubtitleLanguage, native_enum=False, length=5), nullable=False,
    )
    format: Mapped[SubtitleFormat] = mapped_column(
        sa.Enum(SubtitleFormat, native_enum=False, length=5), nullable=False,
    )
    file_path: Mapped[str] = mapped_column(sa.String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, default=datetime.utcnow, nullable=False,
    )

    # -- relationships --
    chapter: Mapped["Chapter"] = relationship(back_populates="subtitle_tracks")

    def __repr__(self) -> str:
        return (
            f"<SubtitleTrack id={self.id!r} chapter_id={self.chapter_id!r} "
            f"language={self.language.value!r} format={self.format.value!r}>"
        )
