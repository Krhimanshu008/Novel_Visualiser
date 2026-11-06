import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def generate_uuid() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Project(Base):
    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False, unique=True, index=True)
    novel_title = Column(String(255), nullable=False)
    author = Column(String(255), nullable=True)
    wiki_url = Column(String(512), nullable=True)       # e.g. "https://lordofthemysteries.fandom.com" or None for EPUB-only
    epub_path = Column(String(1024), nullable=True)     # e.g. "data/novels/book.epub"
    genre_style = Column(String(255), nullable=True)    # e.g. "Dark Fantasy", "Victorian Occult", "Xianxia"
    created_at = Column(DateTime, default=utc_now, nullable=False)

    arcs = relationship("Arc", back_populates="project", cascade="all, delete-orphan")
    sources = relationship("ResearchSource", back_populates="project", cascade="all, delete-orphan")
    entities = relationship("Entity", back_populates="project", cascade="all, delete-orphan")


class Arc(Base):
    __tablename__ = "arcs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    arc_number = Column(Integer, nullable=False)
    start_chapter = Column(Integer, nullable=False)
    end_chapter = Column(Integer, nullable=False)

    project = relationship("Project", back_populates="arcs")


class ResearchSource(Base):
    __tablename__ = "research_sources"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    source_type = Column(String(64), nullable=False, default="fandom_wiki")  # fandom_wiki, epub_novel, hybrid
    url = Column(String(1024), nullable=False)
    title = Column(String(255), nullable=False)
    license_status = Column(String(128), nullable=False, default="CC-BY-SA")
    content_hash = Column(String(64), nullable=True, index=True)  # SHA-256 hash to prevent redundant LLM calls
    scraped_at = Column(DateTime, default=utc_now, nullable=False)

    project = relationship("Project", back_populates="sources")
    claims = relationship("ResearchClaim", back_populates="source", cascade="all, delete-orphan")
    images = relationship("ResearchImage", back_populates="source", cascade="all, delete-orphan")


class Entity(Base):
    __tablename__ = "entities"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    canonical_name = Column(String(255), nullable=False, index=True)
    entity_type = Column(String(64), nullable=False, default="character")  # character, location, creature, artifact
    first_appearance_chapter = Column(Integer, nullable=False, default=1)

    project = relationship("Project", back_populates="entities")
    aliases = relationship("EntityAlias", back_populates="entity", cascade="all, delete-orphan")
    claims = relationship("ResearchClaim", back_populates="entity", cascade="all, delete-orphan")
    images = relationship("ResearchImage", back_populates="entity", cascade="all, delete-orphan")


class EntityAlias(Base):
    __tablename__ = "entity_aliases"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    entity_id = Column(String(36), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    alias = Column(String(255), nullable=False, index=True)
    alias_type = Column(String(64), nullable=False, default="alternate_name")  # birth_name, title, persona, true_name
    valid_from_chapter = Column(Integer, nullable=False, default=1)
    reveal_chapter = Column(Integer, nullable=False, default=1)

    entity = relationship("Entity", back_populates="aliases")


class ResearchClaim(Base):
    __tablename__ = "research_claims"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    source_id = Column(String(36), ForeignKey("research_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_id = Column(String(36), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    era_name = Column(String(255), nullable=True, default="Base Era")
    claim_type = Column(String(64), nullable=False, default="appearance")  # core_trait, attire, equipment, aura, environment, anatomy
    claim_text = Column(Text, nullable=False)
    danbooru_tags = Column(Text, nullable=True)       # JSON list of Danbooru tags for this claim/era
    flux_prompt = Column(Text, nullable=True)         # Natural language visual description for Flux/Video models
    negative_tags = Column(Text, nullable=True)       # JSON list of explicit anti-spoiler tags
    valid_from_chapter = Column(Integer, nullable=False, default=1)
    valid_until_chapter = Column(Integer, nullable=True)  # Null = permanent; Integer = expires after this chapter

    source = relationship("ResearchSource", back_populates="claims")
    entity = relationship("Entity", back_populates="claims")


class ResearchImage(Base):
    __tablename__ = "research_images"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    source_id = Column(String(36), ForeignKey("research_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_id = Column(String(36), ForeignKey("entities.id", ondelete="CASCADE"), nullable=False, index=True)
    image_url = Column(String(1024), nullable=False)
    local_path = Column(String(1024), nullable=True)
    category = Column(String(64), nullable=False, default="reference_photo")
    caption = Column(Text, nullable=True)

    source = relationship("ResearchSource", back_populates="images")
    entity = relationship("Entity", back_populates="images")