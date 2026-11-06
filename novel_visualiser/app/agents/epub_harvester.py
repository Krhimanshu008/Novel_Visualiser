"""
Novel Visualiser — Zero-Token EPUB Chapter Indexer & Appearance Harvester

Extracts debut appearance passages directly from local EPUB files using Python's
built-in zipfile and BeautifulSoup without external heavy dependencies.
Implements SHA-256 caching so re-runs require 0 LLM tokens.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from typing import Any, Optional

from bs4 import BeautifulSoup
from sqlalchemy.orm import Session

from app.database.models import Entity, ResearchClaim, ResearchSource

VISUAL_KEYWORDS = {
    "hair", "eye", "eyes", "skin", "pale", "dark", "face", "tall", "short",
    "slender", "build", "physique", "wore", "wear", "wearing", "armor", "mantle",
    "tunic", "cloak", "mask", "sword", "scar", "silver", "black", "grey", "youth",
    "robe", "robes", "clothes", "garb", "attire", "features", "gaze", "complexion",
}


class EPUBAppearanceHarvester:
    """Zero-token EPUB chapter indexer and targeted visual passage harvester."""

    def __init__(self, epub_path: str | Path = "data/novels/Shadow Slave - Volume 1 To 7 Collection.epub"):
        self.epub_path = Path(epub_path)

    def extract_debut_visual_passages(
        self,
        character_name: str,
        debut_chapter: int = 1,
        early_aliases: list[str] | None = None,
        chapter_window: int = 2,
    ) -> dict[str, Any]:
        """
        Scans only `debut_chapter` to `debut_chapter + chapter_window - 1` inside the EPUB.
        Returns matched visual sentences and their SHA-256 content hash.
        """
        if not self.epub_path.exists():
            return {
                "found": False,
                "reason": f"EPUB not found at {self.epub_path}",
                "sentences": [],
                "combined_text": "",
                "content_hash": "",
            }

        search_terms = {character_name.lower()}
        if early_aliases:
            search_terms.update(a.lower() for a in early_aliases if a)

        target_chapters = set(range(debut_chapter, debut_chapter + max(1, chapter_window)))
        matched_sentences: list[str] = []

        try:
            with zipfile.ZipFile(self.epub_path, "r") as zf:
                html_files = [
                    f for f in zf.namelist()
                    if f.lower().endswith((".xhtml", ".html", ".htm"))
                ]

                for filename in html_files:
                    raw_bytes = zf.read(filename)
                    # Fast byte-level pre-check before BeautifulSoup parsing
                    text_preview = raw_bytes[:4096].decode("utf-8", errors="ignore")
                    ch_num = self._detect_chapter_number(filename, text_preview)
                    if ch_num not in target_chapters:
                        continue

                    soup = BeautifulSoup(raw_bytes, "html.parser")
                    paragraphs = [
                        p.get_text(" ", strip=True)
                        for p in soup.find_all(["p", "div"])
                        if p.get_text(strip=True)
                    ]

                    for idx, para in enumerate(paragraphs):
                        para_lower = para.lower()
                        if any(term in para_lower for term in search_terms):
                            # Inspect current paragraph and immediate next paragraph
                            window_text = " ".join(paragraphs[idx : idx + 2])
                            for sentence in re.split(r"(?<=[.!?])\s+", window_text):
                                words = set(re.findall(r"[a-z]+", sentence.lower()))
                                if words & VISUAL_KEYWORDS and len(sentence.strip()) > 15:
                                    clean_s = sentence.strip()
                                    if clean_s not in matched_sentences:
                                        matched_sentences.append(clean_s)
        except Exception as exc:
            return {
                "found": False,
                "reason": f"Error reading EPUB: {exc}",
                "sentences": [],
                "combined_text": "",
                "content_hash": "",
            }

        combined_text = " ".join(matched_sentences)
        content_hash = hashlib.sha256(combined_text.encode("utf-8")).hexdigest() if combined_text else ""

        return {
            "found": bool(matched_sentences),
            "debut_chapter": debut_chapter,
            "sentences": matched_sentences,
            "combined_text": combined_text,
            "content_hash": content_hash,
        }

    def harvest_into_db(
        self,
        db: Session,
        project_id: str,
        entity: Entity,
        claim_extractor: Any,
        early_aliases: list[str] | None = None,
    ) -> dict[str, Any]:
        """Extracts debut passages from the EPUB and persists atomic claims to SQLite."""
        result = self.extract_debut_visual_passages(
            character_name=entity.canonical_name,
            debut_chapter=entity.first_appearance_chapter or 1,
            early_aliases=early_aliases,
        )
        if not result["found"]:
            return {"status": "skipped", "claims_added": 0, "reason": result.get("reason", "No visual sentences found")}

        source_url = f"epub://{self.epub_path.name}#chapter-{entity.first_appearance_chapter}"

        # Idempotency check: if source URL + content_hash already exist, skip LLM call (0 tokens)
        existing_source = (
            db.query(ResearchSource)
            .filter_by(project_id=project_id, url=source_url)
            .first()
        )
        if existing_source and getattr(existing_source, "content_hash", None) == result["content_hash"]:
            existing_count = db.query(ResearchClaim).filter_by(source_id=existing_source.id, entity_id=entity.id).count()
            if existing_count > 0:
                return {"status": "cached", "claims_added": 0, "cached_claims": existing_count}

        if not existing_source:
            existing_source = ResearchSource(
                project_id=project_id,
                source_type="epub_novel",
                url=source_url,
                title=f"{self.epub_path.stem} (Ch. {entity.first_appearance_chapter})",
                license_status="canonical_epub",
            )
            if hasattr(existing_source, "content_hash"):
                existing_source.content_hash = result["content_hash"]
            db.add(existing_source)
            db.flush()

        extracted_claims = claim_extractor.extract_claims(
            entity_name=entity.canonical_name,
            section_name="EPUB Debut Passage",
            section_text=result["combined_text"],
            source_url=source_url,
        )

        added = 0
        for c in extracted_claims:
            claim_text = c.claim_text if hasattr(c, "claim_text") else (c["claim_text"] if isinstance(c, dict) else str(c))
            claim_type = c.claim_type if hasattr(c, "claim_type") else (c.get("claim_type", "appearance") if isinstance(c, dict) else "appearance")
            valid_from = getattr(c, "valid_from_chapter", None) or (c.get("valid_from_chapter") if isinstance(c, dict) else None) or (entity.first_appearance_chapter or 1)
            db.add(
                ResearchClaim(
                    source_id=existing_source.id,
                    entity_id=entity.id,
                    claim_type=claim_type,
                    claim_text=claim_text,
                    valid_from_chapter=valid_from,
                )
            )
            added += 1

        db.commit()
        return {"status": "harvested", "claims_added": added, "sentences_analyzed": len(result["sentences"])}

    @staticmethod
    def _detect_chapter_number(filename: str, text_preview: str) -> int | None:
        # Match heading like "Chapter 1", "Chapter 18:"
        m = re.search(r"\bchapter\s+(\d+)\b", text_preview, re.IGNORECASE)
        if m:
            return int(m.group(1))
        # Fallback to filename pattern like "ch18.xhtml" or "chapter_001.html"
        m_file = re.search(r"(?:ch|chapter)[_-]?0*(\d+)\b", Path(filename).stem, re.IGNORECASE)
        if m_file:
            return int(m_file.group(1))
        return None

