import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from sqlalchemy.orm import Session

from app.database.models import (
    Project,
    Arc,
    ResearchSource,
    Entity,
    EntityAlias,
    ResearchClaim,
    ResearchImage,
)
from app.agents.reasoning_extractor import UniversalReasoningExtractor
from app.agents.epub_harvester import EPUBAppearanceHarvester


class ResearchAgent:
    """
    Universal, Novel-Agnostic Researcher Agent (Phase 1).
    Works with ANY Fandom/MediaWiki subdomain, custom wiki URL, or standalone EPUB novel.
    """

    def __init__(
        self,
        wiki_subdomain: str | None = None,
        scraper: Any | None = None,
        reasoning_extractor: UniversalReasoningExtractor | None = None,
        export_dir: str | Path = "data/research_exports",
        ref_dir: str | Path = "data/references",
        offline_mode: bool = False,
    ):
        self.wiki_subdomain = wiki_subdomain
        self.scraper = scraper
        self.reasoning = reasoning_extractor or UniversalReasoningExtractor(offline_mode=offline_mode)
        self.export_dir = Path(export_dir)
        self.ref_dir = Path(ref_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.ref_dir.mkdir(parents=True, exist_ok=True)

    def research_entity(
        self,
        db: Session,
        project_id: str,
        entity_query: str,
        entity_type: str = "character",
        debut_chapter_hint: int = 1,
    ) -> dict[str, Any]:
        """
        Universal entrypoint for Characters, Locations, Creatures, or Artifacts across ANY novel.
        """
        project = db.query(Project).filter_by(id=project_id).first()
        if not project:
            raise ValueError(f"Project '{project_id}' not found in database.")

        # 1. Load Novel Arc Timeline from SQLite so the Reasoning Model can map Volumes/Arcs -> Chapters
        arcs = db.query(Arc).filter_by(project_id=project.id).order_by(Arc.arc_number).all()
        arc_timeline = [
            {
                "arc_number": a.arc_number,
                "name": a.name,
                "start_chapter": a.start_chapter,
                "end_chapter": a.end_chapter,
            }
            for a in arcs
        ]

        # 2. Deterministic Harvesting (Wiki + Optional EPUB) — 0 LLM Tokens
        canonical_name = entity_query
        debut_ch = debut_chapter_hint
        raw_aliases: list[str] = []
        raw_infobox: dict[str, str] = {}
        raw_prose_parts: list[str] = []
        source_url = f"local://{project.name}/{entity_query.replace(' ', '_')}"
        page_title = entity_query

        active_scraper = self._resolve_scraper(project)
        if active_scraper:
            pages = active_scraper.search_pages(entity_query, limit=3)
            if pages:
                page_title = pages[0]
            infobox = active_scraper.parse_infobox(page_title)
            if infobox:
                canonical_name = infobox.canonical_name or page_title
                debut_ch = infobox.first_appearance_chapter or debut_ch
                raw_aliases = infobox.aliases or []
                raw_infobox = infobox.attributes or {}

            domain = getattr(active_scraper, "wiki_subdomain", "wiki")
            source_url = (
                f"{project.wiki_url.rstrip('/')}/wiki/{page_title.replace(' ', '_')}"
                if project.wiki_url
                else f"https://{domain}.fandom.com/wiki/{page_title.replace(' ', '_')}"
            )

            # Pull relevant descriptive sections depending on entity_type
            target_sections = (
                ("Appearance", "Personality", "Abilities", "Equipment", "History")
                if entity_type == "character"
                else ("Description", "Appearance", "Geography", "Abilities", "Overview")
            )
            for sec in target_sections:
                sec_text = active_scraper.get_section_text(page_title, sec)
                if sec_text:
                    raw_prose_parts.append(f"[{sec}] {sec_text}")

        # 3. Enrich or Fallback to Direct Novel EPUB Harvesting (0 LLM Tokens)
        if project.epub_path and Path(project.epub_path).exists():
            epub_harvester = EPUBAppearanceHarvester(epub_path=project.epub_path)
            epub_res = epub_harvester.extract_debut_visual_passages(
                character_name=canonical_name,
                debut_chapter=debut_ch,
                early_aliases=raw_aliases[:3],
                chapter_window=2,
            )
            if epub_res["found"]:
                raw_prose_parts.append(f"[EPUB Debut Ch. {debut_ch}] {epub_res['combined_text']}")
                if not active_scraper:
                    source_url = f"epub://{Path(project.epub_path).name}#ch-{debut_ch}-{canonical_name}"

        combined_prose = "\n\n".join(raw_prose_parts)
        hash_input = json.dumps(
            {"infobox": raw_infobox, "aliases": raw_aliases, "prose": combined_prose, "arcs": arc_timeline},
            sort_keys=True,
        )
        content_hash = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()

        # 4. Check SQLite Content Hash Cache (0 LLM Tokens on Re-runs!)
        source = db.query(ResearchSource).filter_by(project_id=project.id, url=source_url).first()
        entity = db.query(Entity).filter_by(project_id=project.id, canonical_name=canonical_name).first()

        if source and entity and source.content_hash == content_hash:
            existing_claims = db.query(ResearchClaim).filter_by(entity_id=entity.id).count()
            if existing_claims > 0:
                return self._build_summary_from_db(db, entity, source, raw_infobox, cached=True)

        # 5. Invoke Universal Reasoning Engine ONCE
        reasoned = self.reasoning.analyze_entity(
            novel_title=project.novel_title,
            entity_name=canonical_name,
            entity_type=entity_type,
            genre_style=project.genre_style,
            debut_chapter=debut_ch,
            raw_aliases=raw_aliases,
            raw_infobox=raw_infobox,
            raw_prose=combined_prose,
            arc_timeline=arc_timeline,
        )

        # 6. Persist Source, Entity, Spoiler-Gated Aliases & Temporal Era Claims in SQLite
        if not source:
            source = ResearchSource(
                project_id=project.id,
                source_type="hybrid" if (active_scraper and project.epub_path) else ("fandom_wiki" if active_scraper else "epub_novel"),
                url=source_url,
                title=f"{project.novel_title} - {reasoned.canonical_name}",
                content_hash=content_hash,
            )
            db.add(source)
            db.flush()
        else:
            source.content_hash = content_hash

        if not entity:
            entity = Entity(
                project_id=project.id,
                canonical_name=reasoned.canonical_name,
                entity_type=reasoned.entity_type,
                first_appearance_chapter=reasoned.first_appearance_chapter,
            )
            db.add(entity)
            db.flush()

        # Upsert Aliases
        db.query(EntityAlias).filter_by(entity_id=entity.id).delete()
        for al in reasoned.aliases:
            db.add(
                EntityAlias(
                    entity_id=entity.id,
                    alias=al.alias,
                    alias_type=al.alias_type,
                    valid_from_chapter=al.valid_from_chapter,
                    reveal_chapter=al.reveal_chapter,
                )
            )

        # Upsert Temporal Era Claims
        db.query(ResearchClaim).filter_by(entity_id=entity.id, source_id=source.id).delete()
        for era in reasoned.visual_eras:
            tags_json = json.dumps(era.danbooru_tags)
            neg_json = json.dumps(era.anti_spoiler_negative_tags)
            for c in era.claims:
                db.add(
                    ResearchClaim(
                        source_id=source.id,
                        entity_id=entity.id,
                        era_name=era.era_name,
                        claim_type=c.get("claim_type", "appearance"),
                        claim_text=c.get("claim_text", ""),
                        danbooru_tags=tags_json,
                        flux_prompt=era.flux_natural_prompt,
                        negative_tags=neg_json,
                        valid_from_chapter=era.valid_from_chapter,
                        valid_until_chapter=era.valid_until_chapter,
                    )
                )

        # 7. Download Reference Plates (if Wiki Scraper is active)
        downloaded_images: list[str] = []
        if active_scraper:
            safe_folder = re.sub(r"[^\w\-]", "_", entity.canonical_name)
            target_dir = self.ref_dir / safe_folder
            target_dir.mkdir(parents=True, exist_ok=True)
            for img_ref in active_scraper.get_image_urls(page_title, max_images=8):
                dest = target_dir / img_ref.filename
                if not dest.exists():
                    active_scraper.download_image(img_ref.image_url, dest)
                if dest.exists():
                    rel_path = dest.as_posix()
                    downloaded_images.append(rel_path)
                    if not db.query(ResearchImage).filter_by(entity_id=entity.id, local_path=rel_path).first():
                        db.add(
                            ResearchImage(
                                source_id=source.id,
                                entity_id=entity.id,
                                image_url=img_ref.image_url,
                                local_path=rel_path,
                                category=img_ref.category,
                                caption=img_ref.caption,
                            )
                        )

        db.commit()
        return self._export_universal_dossier(entity, source_url, raw_infobox, reasoned, downloaded_images)

    def research_character(self, db: Session, project_id: str, character_query: str) -> dict[str, Any]:
        """Backwards-compatible wrapper for character research."""
        return self.research_entity(db, project_id, character_query, entity_type="character")

    def research_location(self, db: Session, project_id: str, location_query: str) -> dict[str, Any]:
        """Wrapper for location research."""
        return self.research_entity(db, project_id, location_query, entity_type="location")

    def research_creature(self, db: Session, project_id: str, creature_query: str) -> dict[str, Any]:
        """Wrapper for creature/monster research."""
        return self.research_entity(db, project_id, creature_query, entity_type="creature")

    def _resolve_scraper(self, project: Project) -> Any | None:
        if self.scraper:
            return self.scraper
        # Dynamically infer Fandom subdomain from project.wiki_url if available
        if project.wiki_url:
            host = urlparse(project.wiki_url).netloc
            if ".fandom.com" in host:
                subdomain = host.split(".fandom.com")[0]
                # If FandomWikiScraper is defined in this module, instantiate it dynamically
                scraper_cls = globals().get("FandomWikiScraper")
                if scraper_cls:
                    return scraper_cls(wiki_subdomain=subdomain)
        elif self.wiki_subdomain:
            scraper_cls = globals().get("FandomWikiScraper")
            if scraper_cls:
                return scraper_cls(wiki_subdomain=self.wiki_subdomain)
        return None

    def _export_universal_dossier(
        self,
        entity: Entity,
        source_url: str,
        infobox: dict[str, str],
        reasoned: Any,
        downloaded_images: list[str],
    ) -> dict[str, Any]:
        safe_name = re.sub(r"[^\w\-]", "_", entity.canonical_name)
        json_path = self.export_dir / f"{safe_name}_dossier.json"
        md_path = self.export_dir / f"{safe_name}_dossier.md"

        total_claims = sum(len(era.claims) for era in reasoned.visual_eras)
        dossier_data = {
            "entity_id": entity.id,
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type,
            "genre_detected": reasoned.genre_detected,
            "source_url": source_url,
            "first_appearance_chapter": entity.first_appearance_chapter,
            "aliases": [a.model_dump() for a in reasoned.aliases],
            "visual_eras": [e.model_dump() for e in reasoned.visual_eras],
            "downloaded_images": downloaded_images,
            "claims_count": total_claims,
            "infobox": infobox,
            "reasoning_trace": reasoned.reasoning_trace,
            "dossier_path": json_path.as_posix(),
            "dossier_markdown_path": md_path.as_posix(),
        }
        json_path.write_text(json.dumps(dossier_data, indent=2, ensure_ascii=False), encoding="utf-8")

        md_lines = [
            f"# {entity.canonical_name} — Universal Canonical Dossier",
            f"- **Entity Type**: `{entity.entity_type}` | **Genre**: `{reasoned.genre_detected}`",
            f"- **Source**: {source_url}",
            f"- **First Appearance**: Chapter {entity.first_appearance_chapter}",
            "\n## Reasoning Chain-of-Thought Summary",
            f"> {reasoned.reasoning_trace}",
            "\n## Spoiler-Gated Aliases",
            *[
                f"- **{a.alias}** (`{a.alias_type}`) — Valid From Ch. {a.valid_from_chapter} | Reveal Ch. {a.reveal_chapter}"
                for a in reasoned.aliases
            ],
            "\n## Temporal Visual Eras",
        ]
        for era in reasoned.visual_eras:
            until_str = f"Ch. {era.valid_until_chapter}" if era.valid_until_chapter else "Present (Ongoing)"
            md_lines.extend([
                f"### {era.era_name} (Ch. {era.valid_from_chapter} → {until_str})",
                f"- **Danbooru Prompt**: `{', '.join(era.danbooru_tags)}`",
                f"- **Flux Natural Prompt**: {era.flux_natural_prompt}",
                f"- **Anti-Spoiler Negative Tags**: `{', '.join(era.anti_spoiler_negative_tags)}`",
                "- **Atomic Claims**:",
                *[f"  - ({c.get('claim_type')}) {c.get('claim_text')}" for c in era.claims],
                "",
            ])

        md_path.write_text("\n".join(md_lines), encoding="utf-8")
        return dossier_data

    def _build_summary_from_db(
        self, db: Session, entity: Entity, source: ResearchSource, infobox: dict[str, str], cached: bool = True
    ) -> dict[str, Any]:
        safe_name = re.sub(r"[^\w\-]", "_", entity.canonical_name)
        json_path = self.export_dir / f"{safe_name}_dossier.json"
        md_path = self.export_dir / f"{safe_name}_dossier.md"
        aliases = [a.alias for a in db.query(EntityAlias).filter_by(entity_id=entity.id).all()]
        images = [img.local_path for img in db.query(ResearchImage).filter_by(entity_id=entity.id).all() if img.local_path]
        claims_count = db.query(ResearchClaim).filter_by(entity_id=entity.id).count()
        return {
            "status": "cached_hit" if cached else "ok",
            "entity_id": entity.id,
            "canonical_name": entity.canonical_name,
            "source_url": source.url,
            "first_appearance_chapter": entity.first_appearance_chapter,
            "aliases": aliases,
            "downloaded_images": images,
            "claims_count": claims_count,
            "infobox": infobox,
            "dossier_path": json_path.as_posix(),
            "dossier_markdown_path": md_path.as_posix(),
        }