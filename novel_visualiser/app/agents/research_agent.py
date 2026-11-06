"""
Novel Visualiser — External Research & Reference Agent (Agent 1)

Implements the hybrid research workflow:
- 85% Deterministic Python (MediaWiki API parser, BeautifulSoup infobox scraper,
  imageinfo harvester, arc index parser) — 0 LLM tokens, 100% structured accuracy.
- 15% Targeted LLM (Atomic visual claim extractor with chapter boundary tagging).

Stores research into:
- research_sources
- research_claims
- research_images
- entities & entity_aliases (with valid_from_chapter and reveal_chapter)
"""

from __future__ import annotations

import json
import logging
import re
import hashlib
import urllib.parse
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
from bs4 import BeautifulSoup
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import (
    AliasType,
    Arc,
    ClaimType,
    Entity,
    EntityAlias,
    EntityType,
    ImageCategory,
    LicenseStatus,
    Project,
    ResearchClaim,
    ResearchImage,
    ResearchSource,
    SourceType,
)
from app.schemas.research import (
    ArcStructure,
    ResearchClaim as ResearchClaimSchema,
    ResearchImageRef,
    WikiInfoboxData,
)

logger = logging.getLogger("novel_visualiser.research")


class FandomWikiScraper:
    """
    Deterministic scraper consuming Fandom's free MediaWiki API.
    Zero LLM tokens required.
    """

    def __init__(self, subdomain: str = "shadowslave", user_agent: str | None = None, wiki_subdomain: str | None = None):
        self.subdomain = wiki_subdomain or subdomain or "shadowslave"
        self.wiki_subdomain = self.subdomain
        self.base_url = f"https://{self.subdomain}.fandom.com/api.php"
        self.wiki_url = f"https://{self.subdomain}.fandom.com/wiki"
        self.headers = {
            "User-Agent": user_agent or "NovelVisualiser/0.1.0 (Web novel visualizer agent; educational research)"
        }
        self.client = httpx.Client(timeout=15.0, headers=self.headers, follow_redirects=True)

    def close(self):
        self.client.close()

    def search_pages(self, query: str, limit: int = 5) -> list[str]:
        """Search wiki page titles using OpenSearch API."""
        params = {
            "action": "opensearch",
            "search": query,
            "limit": limit,
            "format": "json",
        }
        try:
            resp = self.client.get(self.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) >= 2:
                return data[1]
            return []
        except Exception as exc:
            logger.warning("Wiki search failed for '%s': %s", query, exc)
            return []

    def get_page_sections(self, page_title: str) -> list[dict[str, Any]]:
        """Get the table of contents / sections for a page."""
        params = {
            "action": "parse",
            "page": page_title,
            "prop": "sections",
            "format": "json",
        }
        try:
            resp = self.client.get(self.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("parse", {}).get("sections", [])
        except Exception as exc:
            logger.warning("Failed to get sections for '%s': %s", page_title, exc)
            return []

    def get_section_text(self, page_title: str, section_index: int | str) -> str:
        """Fetch raw HTML of a specific section and extract clean text."""
        params = {
            "action": "parse",
            "page": page_title,
            "section": section_index,
            "prop": "text",
            "format": "json",
        }
        try:
            resp = self.client.get(self.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            html = data.get("parse", {}).get("text", {}).get("*", "")
            soup = BeautifulSoup(html, "html.parser")
            paragraphs = [p.get_text(strip=True) for p in soup.find_all("p") if p.get_text(strip=True)]
            return "\n\n".join(paragraphs)
        except Exception as exc:
            logger.warning("Failed to get section %s for '%s': %s", section_index, page_title, exc)
            return ""

    def parse_infobox(self, page_title: str) -> Optional[WikiInfoboxData]:
        """
        Extract structured infobox fields from a character or location page.
        Handles Fandom's Portable Infobox format.
        """
        params = {
            "action": "parse",
            "page": page_title,
            "prop": "text",
            "format": "json",
        }
        try:
            resp = self.client.get(self.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.warning("MediaWiki error for '%s': %s", page_title, data["error"])
                return None

            html = data.get("parse", {}).get("text", {}).get("*", "")
            soup = BeautifulSoup(html, "html.parser")
            infobox = soup.find("aside", class_="portable-infobox")
            if not infobox:
                logger.info("No portable-infobox found on '%s'", page_title)
                return None

            fields: dict[str, str] = {}
            for item in infobox.find_all(["div", "section"], attrs={"data-source": True}):
                key = item.get("data-source", "").strip()
                label = item.find(class_="pi-data-label")
                value = item.find(class_="pi-data-value")
                label_txt = label.get_text(strip=True) if label else key
                # Use comma separator so list/br elements are clean individual items
                value_txt = value.get_text(separator=", ", strip=True) if value else ""
                # Strip reference brackets like [1], [22]
                value_clean = re.sub(r"\[\d+\]", "", value_txt).strip()
                fields[label_txt] = value_clean

            # Parse aliases
            aliases_raw = fields.get("Alias", "") or fields.get("Aliases", "")
            aliases = []
            if aliases_raw:
                raw_parts = re.split(r"[,/;\n]", aliases_raw)
                for part in raw_parts:
                    cleaned = re.sub(r"\(.*?\)", "", part).strip().strip('"\'')
                    if cleaned and cleaned not in aliases and len(cleaned) < 50:
                        aliases.append(cleaned)

            # Titles can also yield aliases
            titles_raw = fields.get("Titles", "") or fields.get("Title", "")
            if titles_raw:
                for part in re.split(r"[,/;\n]", titles_raw):
                    cleaned = re.sub(r"\(.*?\)", "", part).strip().strip('"\'')
                    if cleaned and cleaned not in aliases and len(cleaned) < 50:
                        aliases.append(cleaned)

            # First appearance / Debut
            first_app = (
                fields.get("First Appearance")
                or fields.get("Debut")
                or fields.get("Chapter")
                or None
            )

            # Species / Rank / Affiliation
            species = fields.get("Species") or fields.get("Race") or None
            rank = fields.get("Rank") or fields.get("Class") or None
            affiliation = fields.get("Affiliation") or fields.get("Faction") or None
            status_val = fields.get("Vital Status") or fields.get("Status") or None

            return WikiInfoboxData(
                page_title=page_title,
                page_url=f"{self.wiki_url}/{urllib.parse.quote(page_title)}",
                infobox_fields=fields,
                aliases=aliases,
                first_appearance=first_app,
                species=species,
                rank=rank,
                affiliation=affiliation,
                status=status_val,
            )
        except Exception as exc:
            logger.error("Failed to parse infobox for '%s': %s", page_title, exc)
            return None

    def get_image_urls(self, page_title: str, max_images: int = 10) -> list[ResearchImageRef]:
        """
        Fetch all images embedded on the page, query imageinfo for high-res CDN URLs,
        and classify them into official_cover, official_illustration, or community_art.
        """
        params = {
            "action": "parse",
            "page": page_title,
            "prop": "images",
            "format": "json",
        }
        results: list[ResearchImageRef] = []
        try:
            resp = self.client.get(self.base_url, params=params)
            resp.raise_for_status()
            data = resp.json()
            image_names = data.get("parse", {}).get("images", [])[:max_images]

            # Filter out UI icons (like Info-icon, flag icons)
            filtered_names = [
                img for img in image_names
                if not any(ignore in img.lower() for ignore in ["icon", "arrow", "flag", "star", "bullet"])
            ]

            for img_name in filtered_names:
                file_title = f"File:{img_name}" if not img_name.startswith("File:") else img_name
                info_params = {
                    "action": "query",
                    "titles": file_title,
                    "prop": "imageinfo",
                    "iiprop": "url|size|mime",
                    "format": "json",
                }
                info_resp = self.client.get(self.base_url, params=info_params)
                if info_resp.status_code == 200:
                    info_data = info_resp.json()
                    pages = info_data.get("query", {}).get("pages", {})
                    for _, pinfo in pages.items():
                        ii_list = pinfo.get("imageinfo", [])
                        if ii_list:
                            ii = ii_list[0]
                            img_url = ii.get("url")
                            if not img_url:
                                continue

                            # Classify category
                            lower_name = img_name.lower()
                            if "cover" in lower_name or "volume" in lower_name:
                                cat = "official_cover"
                            elif "official" in lower_name or "card" in lower_name or "chibi" in lower_name:
                                cat = "official_illustration"
                            elif "fan" in lower_name or "art" in lower_name:
                                cat = "community_art"
                            else:
                                cat = "reference_photo"

                            results.append(
                                ResearchImageRef(
                                    image_url=img_url,
                                    source_url=f"{self.wiki_url}/{urllib.parse.quote(file_title)}",
                                    entity_name=page_title,
                                    category=cat,
                                    caption=f"{page_title} - {img_name}",
                                    license_status="fair_use",
                                )
                            )
            return results
        except Exception as exc:
            logger.error("Failed to fetch image URLs for '%s': %s", page_title, exc)
            return []

    def download_image(self, url: str, target_path: Path) -> Optional[Path]:
        """Download a reference image from CDN and save to disk."""
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            resp = self.client.get(url)
            if resp.status_code == 200:
                target_path.write_bytes(resp.content)
                logger.info("Saved reference image: %s", target_path)
                return target_path
        except Exception as exc:
            logger.warning("Failed to download image from %s: %s", url, exc)
        return None

    def get_volume_arcs(self, max_volumes: int = 15) -> list[ArcStructure]:
        """
        Scrape story volumes/arcs from the wiki.
        Pulls Volume titles, start/end chapters from the portable infobox.
        """
        titles = self.search_pages("Volume ", limit=max_volumes)
        arcs: list[ArcStructure] = []
        for t in titles:
            if not t.lower().startswith("volume"):
                continue
            v_match = re.search(r"volume\s+(\d+)", t, re.IGNORECASE)
            if not v_match:
                continue
            vol_num = int(v_match.group(1))

            infobox = self.parse_infobox(t)
            start_ch, end_ch = 1, 999
            if infobox:
                chap_covered = infobox.infobox_fields.get("ChaptersCovered", "")
                m = re.search(r"(\d+)\s*[-–]\s*(\d+)", chap_covered)
                if m:
                    start_ch, end_ch = int(m.group(1)), int(m.group(2))
                elif re.search(r"(\d+)", chap_covered):
                    start_ch = int(re.search(r"(\d+)", chap_covered).group(1))

            arcs.append(
                ArcStructure(
                    arc_name=t,
                    arc_number=vol_num,
                    start_chapter=start_ch,
                    end_chapter=end_ch,
                    description=f"{t} (Chapters {start_ch}-{end_ch})",
                )
            )

        arcs.sort(key=lambda a: a.arc_number)
        return arcs


class LLMClaimExtractor:
    """
    Extracts atomic visual facts from unstructured wiki prose (e.g. Appearance paragraphs)
    and estimates valid_from_chapter to protect spoiler boundaries.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        self.settings = get_settings()
        self.api_key = api_key or self.settings.llm.google_api_key
        self.model = model

    def extract_claims(
        self,
        entity_name: str,
        section_name: str,
        section_text: str,
        source_url: str,
    ) -> list[ResearchClaimSchema]:
        """
        Break an Appearance or Equipment text into atomic visual facts with chapter boundaries.
        Uses structured heuristic parsing when LLM API key is not set, or Gemini Flash when available.
        """
        if not section_text.strip():
            return []

        # If Google API key is configured, invoke Gemini Flash
        if self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                return self._extract_with_gemini(entity_name, section_name, section_text, source_url)
            except Exception as exc:
                logger.warning("Gemini claim extraction failed, falling back to deterministic parser: %s", exc)

        # 100% Deterministic Fallback: Rule-based paragraph splitting & chapter regex tagger
        return self._extract_deterministic(entity_name, section_name, section_text, source_url)

    def _extract_deterministic(
        self,
        entity_name: str,
        section_name: str,
        section_text: str,
        source_url: str,
    ) -> list[ResearchClaimSchema]:
        """
        Rule-based atomic claim splitter:
        - Detects chapter mentions like 'Chapter 1', 'Chapter 29', 'Chapter 176'
        - Default chapter = 1 (Debut) if no later chapter is cited
        - Strips citation footnotes like [1], [22]
        """
        claims: list[ResearchClaimSchema] = []
        paragraphs = section_text.split("\n\n")

        current_chapter = 1

        for p in paragraphs:
            clean_p = re.sub(r"\[\d+\]", "", p).strip()
            if not clean_p or len(clean_p) < 15:
                continue

            # Detect chapter patterns: "Chapter 176", "in Chapter 29", "Chapter 309"
            ch_match = re.search(r"chapter\s+(\d+)", clean_p, re.IGNORECASE)
            if ch_match:
                current_chapter = int(ch_match.group(1))

            claim_type_str = "appearance"
            if "equipment" in section_name.lower() or "weapon" in section_name.lower() or "armor" in clean_p.lower():
                claim_type_str = "equipment"
            elif "ability" in section_name.lower() or "flaw" in clean_p.lower() or "aspect" in clean_p.lower():
                claim_type_str = "ability"

            claims.append(
                ResearchClaimSchema(
                    source_url=source_url,
                    source_type="fandom_wiki",
                    entity_name=entity_name,
                    claim_type=claim_type_str,
                    claim_text=clean_p,
                    valid_from_chapter=current_chapter,
                    confidence="confirmed",
                )
            )

        return claims

    def _extract_with_gemini(
        self,
        entity_name: str,
        section_name: str,
        section_text: str,
        source_url: str,
    ) -> list[ResearchClaimSchema]:
        """Extract atomic facts using Google Gemini API structured output."""
        from google import genai

        client = genai.Client(api_key=self.api_key)
        prompt = f"""
You are an expert research analyst for an AI adaptation of a web novel.
Analyze the following wiki section about {entity_name} ({section_name}).
Break it down into a list of atomic visual facts, and for each fact, determine the earliest chapter number where it applies (valid_from_chapter).

Rules:
1. Base appearance at debut / introduction = chapter 1.
2. If text explicitly mentions a chapter or event (e.g. "After surviving his First Nightmare", "Chapter 176", "Labyrinth"), assign that chapter number.
3. Be concise and strictly factual.
4. Output JSON list matching the schema:
   [{{ "claim_text": "...", "claim_type": "appearance"|"equipment"|"ability", "valid_from_chapter": 1 }}]

Section Text:
{section_text[:4000]}
"""
        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        )

        data = json.loads(response.text)
        claims = []
        for item in data:
            claims.append(
                ResearchClaimSchema(
                    source_url=source_url,
                    source_type="fandom_wiki",
                    entity_name=entity_name,
                    claim_type=item.get("claim_type", "appearance"),
                    claim_text=item.get("claim_text", ""),
                    valid_from_chapter=item.get("valid_from_chapter", 1),
                    confidence="confirmed",
                )
            )
        return claims


class ResearchAgent:
    """
    Coordinator agent for Phase 1.
    Connects Scraper + LLM Claim Extractor to SQLAlchemy database records.
    """

    def __init__(
        self,
        wiki_subdomain: str = "shadowslave",
        scraper: Any | None = None,
        reasoning_extractor: Any | None = None,
        export_dir: str | Path = "data/research_exports",
        ref_dir: str | Path = "data/references",
        offline_mode: bool = False,
    ):
        self.wiki_subdomain = wiki_subdomain
        self.scraper = scraper or FandomWikiScraper(subdomain=wiki_subdomain)
        self.reasoning_extractor = reasoning_extractor
        self.export_dir = Path(export_dir)
        self.ref_dir = Path(ref_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        self.ref_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = LLMClaimExtractor()
        if reasoning_extractor is None and offline_mode:
            from app.agents.reasoning_extractor import UniversalReasoningExtractor
            self.reasoning = UniversalReasoningExtractor(offline_mode=True)
        else:
            self.reasoning = reasoning_extractor

    def research_character(
        self,
        db: Session,
        project_id: str,
        character_query: str,
    ) -> dict[str, Any]:
        if self.reasoning_extractor is not None:
            return self.research_entity(db, project_id, character_query, entity_type="character")
        """
        Research a character by name:
        1. Find canonical wiki page via search
        2. Parse infobox -> create/update Entity & EntityAlias records
        3. Parse Appearance section -> create ResearchClaim records with valid_from_chapter
        4. Harvester images -> create ResearchImage records
        5. Persist everything into SQLite
        """
        logger.info("Starting research for character '%s' on %s.fandom.com", character_query, self.wiki_subdomain)

        # 1. Search page
        search_results = self.scraper.search_pages(character_query, limit=3)
        page_title = search_results[0] if search_results else character_query

        # 2. Register research source
        source_url = f"{self.scraper.wiki_url}/{urllib.parse.quote(page_title)}"
        source = db.execute(
            select(ResearchSource).where(
                ResearchSource.project_id == project_id,
                ResearchSource.url == source_url,
            )
        ).scalar_one_or_none()

        if not source:
            source = ResearchSource(
                project_id=project_id,
                source_type=SourceType.FANDOM_WIKI,
                url=source_url,
                title=f"{page_title} - {self.wiki_subdomain.capitalize()} Wiki",
                license_status=LicenseStatus.FAIR_USE,
                scraped_at=datetime.utcnow(),
            )
            db.add(source)
            db.flush()

        # 3. Parse infobox
        infobox = self.scraper.parse_infobox(page_title)
        first_chap = 1
        if infobox and infobox.first_appearance:
            match = re.search(r"\d+", infobox.first_appearance)
            if match:
                first_chap = int(match.group(0))

        # 4. Create or update Entity
        entity = db.execute(
            select(Entity).where(
                Entity.project_id == project_id,
                Entity.canonical_name == page_title,
            )
        ).scalar_one_or_none()

        if not entity:
            entity = Entity(
                project_id=project_id,
                canonical_name=page_title,
                entity_type=EntityType.CHARACTER,
                first_appearance_chapter=first_chap,
            )
            db.add(entity)
            db.flush()

        # 5. Populate Aliases from infobox
        created_aliases = []
        if infobox:
            all_alias_candidates = [character_query] + infobox.aliases
            for al in all_alias_candidates:
                al_clean = al.strip()
                if not al_clean or al_clean == page_title:
                    continue

                existing_alias = db.execute(
                    select(EntityAlias).where(
                        EntityAlias.entity_id == entity.id,
                        EntityAlias.alias == al_clean,
                    )
                ).scalar_one_or_none()

                if not existing_alias:
                    # Determine reveal chapter: if title contains 'Mongrel', delay reveal
                    reveal_ch = 1
                    al_type = AliasType.NICKNAME
                    if "mongrel" in al_clean.lower():
                        reveal_ch = 25
                        al_type = AliasType.TITLE

                    new_alias = EntityAlias(
                        entity_id=entity.id,
                        alias=al_clean,
                        alias_type=al_type,
                        valid_from_chapter=1 if reveal_ch == 1 else 20,
                        reveal_chapter=reveal_ch,
                    )
                    db.add(new_alias)
                    created_aliases.append(al_clean)

        # 6. Extract Claims from Appearance section
        sections = self.scraper.get_page_sections(page_title)
        app_sec_index = None
        for s in sections:
            line = s.get("line", "").lower()
            if "appearance" in line:
                app_sec_index = s.get("index")
                break

        created_claims = []
        if app_sec_index:
            sec_text = self.scraper.get_section_text(page_title, app_sec_index)
            claims = self.extractor.extract_claims(
                entity_name=page_title,
                section_name="Appearance",
                section_text=sec_text,
                source_url=source_url,
            )
            for c in claims:
                ctype = ClaimType.APPEARANCE
                if c.claim_type == "equipment":
                    ctype = ClaimType.EQUIPMENT
                elif c.claim_type == "ability":
                    ctype = ClaimType.ABILITY

                claim_db = ResearchClaim(
                    source_id=source.id,
                    entity_id=entity.id,
                    claim_type=ctype,
                    claim_text=c.claim_text,
                    valid_from_chapter=c.valid_from_chapter,
                )
                db.add(claim_db)
                created_claims.append(c.claim_text[:50] + "...")

        # 7. Harvest Images and download to local disk
        images = self.scraper.get_image_urls(page_title, max_images=6)
        created_images = []
        downloaded_refs = []
        settings = get_settings()
        refs_dir = Path(settings.project.references_dir) / page_title
        refs_dir.mkdir(parents=True, exist_ok=True)

        for img in images:
            # Extract real filename before /revision/ in Fandom CDN URLs
            m = re.search(r"/images/[^/]+/[^/]+/([^/?#]+)", img.image_url)
            if m:
                base_fname = m.group(1)
            else:
                base_fname = Path(urllib.parse.urlparse(img.image_url).path).name
            clean_fname = re.sub(r"[^\w\.-]", "_", base_fname) or f"img_{len(created_images)}.png"
            target_file = refs_dir / clean_fname

            saved_path = self.scraper.download_image(img.image_url, target_file)

            img_db = ResearchImage(
                source_id=source.id,
                entity_id=entity.id,
                image_url=img.image_url,
                local_path=str(saved_path) if saved_path else None,
                category=ImageCategory(img.category),
                caption=img.caption,
            )
            db.add(img_db)
            created_images.append(img.image_url)
            if saved_path:
                downloaded_refs.append(str(saved_path))

        db.commit()

        # 8. Export Human-Readable Research Dossier (.md and .json)
        exports_dir = Path("data/research_exports")
        exports_dir.mkdir(parents=True, exist_ok=True)
        md_path = exports_dir / f"{page_title}_dossier.md"
        json_path = exports_dir / f"{page_title}_dossier.json"

        md_lines = [
            f"# Research Dossier: {page_title}",
            f"**Canonical Entity ID:** `{entity.id}`  ",
            f"**Source URL:** [{source_url}]({source_url})  ",
            f"**First Appearance:** Chapter {first_chap}  ",
            f"**Scraped At:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}  \n",
            "## 1. Infobox Attributes",
        ]
        if infobox:
            for k, v in infobox.infobox_fields.items():
                md_lines.append(f"- **{k}:** {v}")

        md_lines.append("\n## 2. Entity Aliases & Reveal Boundaries")
        for al in created_aliases:
            al_obj = db.execute(
                select(EntityAlias).where(EntityAlias.entity_id == entity.id, EntityAlias.alias == al)
            ).scalar_one_or_none()
            if al_obj:
                md_lines.append(
                    f"- **{al}** ({al_obj.alias_type.value}) — Valid from Ch {al_obj.valid_from_chapter}, Revealed at Ch {al_obj.reveal_chapter}"
                )
            else:
                md_lines.append(f"- **{al}**")

        md_lines.append(f"\n## 3. Reference Images Saved to Disk ({len(downloaded_refs)})")
        for ref_file in downloaded_refs:
            md_lines.append(f"- `{ref_file}`")

        md_lines.append(f"\n## 4. Visual Claims by Chapter Boundary")
        all_claims = db.execute(
            select(ResearchClaim).where(ResearchClaim.entity_id == entity.id).order_by(ResearchClaim.valid_from_chapter)
        ).scalars().all()
        for cl in all_claims:
            md_lines.append(f"### Chapter {cl.valid_from_chapter}+ [{cl.claim_type.value.upper()}]")
            md_lines.append(f"> {cl.claim_text}\n")

        md_path.write_text("\n".join(md_lines), encoding="utf-8")

        dossier_data = {
            "entity_id": entity.id,
            "canonical_name": page_title,
            "source_url": source_url,
            "first_appearance_chapter": first_chap,
            "aliases": created_aliases,
            "downloaded_images": downloaded_refs,
            "claims_count": len(all_claims),
            "infobox": infobox.infobox_fields if infobox else {},
            "dossier_markdown_path": str(md_path),
        }
        json_path.write_text(json.dumps(dossier_data, indent=2), encoding="utf-8")

        logger.info(
            "Research completed for '%s': %d aliases, %d claims, %d images saved (Dossier: %s)",
            page_title,
            len(created_aliases),
            len(created_claims),
            len(downloaded_refs),
            md_path,
        )

        return {
            "entity_id": entity.id,
            "canonical_name": page_title,
            "source_url": source_url,
            "aliases_added": created_aliases,
            "claims_count": len(created_claims),
            "images_count": len(downloaded_refs),
            "dossier_path": str(md_path),
        }

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

        # 1. Load Novel Arc Timeline from SQLite
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
                canonical_name = infobox.canonical_name or infobox.page_title or page_title
                debut_ch = infobox.first_appearance_chapter or debut_ch
                raw_aliases = infobox.aliases or []
                raw_infobox = infobox.infobox_fields or infobox.attributes or {}

            domain = getattr(active_scraper, "wiki_subdomain", getattr(active_scraper, "subdomain", "wiki"))
            source_url = (
                f"{project.wiki_url.rstrip('/')}/wiki/{page_title.replace(' ', '_')}"
                if project.wiki_url
                else f"https://{domain}.fandom.com/wiki/{page_title.replace(' ', '_')}"
            )

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
            from app.agents.epub_harvester import EPUBAppearanceHarvester
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

        if source and entity and getattr(source, "content_hash", None) == content_hash:
            existing_claims = db.query(ResearchClaim).filter_by(entity_id=entity.id).count()
            if existing_claims > 0:
                return self._build_summary_from_db(db, entity, source, raw_infobox, cached=True)

        # 5. Invoke Universal Reasoning Engine ONCE
        if not getattr(self, "reasoning", None):
            from app.agents.reasoning_extractor import UniversalReasoningExtractor
            self.reasoning = UniversalReasoningExtractor(offline_mode=True)

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

    def research_location(self, db: Session, project_id: str, location_query: str) -> dict[str, Any]:
        """Wrapper for location research."""
        return self.research_entity(db, project_id, location_query, entity_type="location")

    def research_creature(self, db: Session, project_id: str, creature_query: str) -> dict[str, Any]:
        """Wrapper for creature/monster research."""
        return self.research_entity(db, project_id, creature_query, entity_type="creature")

    def _resolve_scraper(self, project: Project) -> Any | None:
        if self.scraper:
            return self.scraper
        if project.wiki_url:
            host = urlparse(project.wiki_url).netloc
            if ".fandom.com" in host:
                subdomain = host.split(".fandom.com")[0]
                return FandomWikiScraper(subdomain=subdomain)
        elif self.wiki_subdomain:
            return FandomWikiScraper(subdomain=self.wiki_subdomain)
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
        aliases_list = [a.alias for a in reasoned.aliases]
        dossier_data = {
            "entity_id": entity.id,
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type,
            "genre_detected": reasoned.genre_detected,
            "source_url": source_url,
            "first_appearance_chapter": entity.first_appearance_chapter,
            "aliases": aliases_list,
            "aliases_added": aliases_list,
            "visual_eras": [e.model_dump() for e in reasoned.visual_eras],
            "downloaded_images": downloaded_images,
            "images_count": len(downloaded_images),
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
            "aliases_added": aliases,
            "downloaded_images": images,
            "images_count": len(images),
            "claims_count": claims_count,
            "infobox": infobox,
            "dossier_path": json_path.as_posix(),
            "dossier_markdown_path": md_path.as_posix(),
        }

    def research_arcs(self, db: Session, project_id: str) -> list[dict[str, Any]]:
        """
        Harvest story volumes/arcs from the wiki and populate the arcs table in SQLite.
        """
        arcs = self.scraper.get_volume_arcs()
        saved = []
        for a in arcs:
            existing = db.execute(
                select(Arc).where(
                    Arc.project_id == project_id,
                    Arc.arc_number == a.arc_number,
                )
            ).scalar_one_or_none()

            if not existing:
                arc_obj = Arc(
                    project_id=project_id,
                    name=a.arc_name,
                    arc_number=a.arc_number,
                    start_chapter=a.start_chapter,
                    end_chapter=a.end_chapter,
                    description=a.description,
                )
                db.add(arc_obj)
                saved.append({"name": a.arc_name, "start": a.start_chapter, "end": a.end_chapter})
            else:
                existing.start_chapter = a.start_chapter
                existing.end_chapter = a.end_chapter
                saved.append({"name": existing.name, "start": existing.start_chapter, "end": existing.end_chapter})

        db.commit()
        logger.info("Research completed for arcs: %d volumes saved", len(saved))
        return saved
