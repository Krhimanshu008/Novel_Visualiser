"""External Research & Spoiler Boundaries contracts.

Schemas for wiki scraping results, atomic research claims, visual reference
images, spoiler-safe knowledge bundles, and story arc structures. These
contracts govern how external information is ingested, filtered, and made
available to downstream agents without leaking future plot details.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class WikiInfoboxData(BaseModel):
    """Structured data parsed from a Fandom Wiki infobox."""

    page_title: str = Field(default="", description="Title of the wiki page")
    page_url: str = Field(default="", description="Full URL of the wiki page")
    infobox_fields: dict[str, str] = Field(
        default_factory=dict, description="Raw key-value pairs extracted from the infobox"
    )
    canonical_name: str | None = Field(default=None, description="Canonical name from infobox")
    attributes: dict[str, str] = Field(
        default_factory=dict, description="Alias for infobox_fields"
    )
    first_appearance_chapter: int | None = Field(
        default=None, description="First appearance chapter number"
    )
    aliases: list[str] = Field(
        default_factory=list, description="Known aliases listed on the wiki page"
    )
    first_appearance: str | None = Field(
        default=None, description="Chapter or episode of first appearance"
    )
    species: str | None = Field(default=None, description="Species or race")
    rank: str | None = Field(default=None, description="Rank or title")
    affiliation: str | None = Field(
        default=None, description="Primary faction or group affiliation"
    )
    status: str | None = Field(
        default=None, description="Current status, e.g. 'Alive', 'Deceased'"
    )

    @model_validator(mode="before")
    @classmethod
    def _harmonize_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("canonical_name") and data.get("page_title"):
                data["canonical_name"] = data["page_title"]
            if not data.get("page_title") and data.get("canonical_name"):
                data["page_title"] = data["canonical_name"]
            if not data.get("attributes") and data.get("infobox_fields"):
                data["attributes"] = dict(data["infobox_fields"])
            if not data.get("infobox_fields") and data.get("attributes"):
                data["infobox_fields"] = dict(data["attributes"])
            if data.get("first_appearance_chapter") is None and data.get("first_appearance"):
                m = re.search(r"\d+", str(data["first_appearance"]))
                if m:
                    data["first_appearance_chapter"] = int(m.group(0))
            if data.get("first_appearance") is None and data.get("first_appearance_chapter") is not None:
                data["first_appearance"] = f"Chapter {data['first_appearance_chapter']}"
        return data


class ResearchClaim(BaseModel):
    """An atomic visual/lore fact extracted from research sources."""

    claim_id: str | None = Field(
        default=None, description="Unique identifier for this claim"
    )
    source_url: str = Field(default="", description="URL of the source where this claim was found")
    source_type: Literal[
        "fandom_wiki", "official_art", "community_art", "publication", "web_article", "epub_novel", "hybrid"
    ] | str = Field(default="fandom_wiki", description="Category of the research source")
    entity_name: str = Field(default="", description="Entity name as referenced in the source")
    entity_id: str | None = Field(
        default=None, description="Resolved entity ID after entity matching"
    )
    claim_type: Literal[
        "appearance",
        "ability",
        "equipment",
        "relationship",
        "location_feature",
        "title",
        "alias",
        "backstory",
        "physique",
        "hair",
        "eyes",
        "attire",
        "aura",
        "environment",
        "anatomy",
        "core_trait",
    ] | str = Field(default="appearance", description="Classification of the claim")
    claim_text: str = Field(description="The atomic fact statement")
    structured_visual: dict | None = Field(
        default=None, description="Optional parsed visual attributes as key-value pairs"
    )
    valid_from_chapter: int = Field(
        default=1, description="Chapter from which this fact becomes relevant"
    )
    valid_to_chapter: int | None = Field(
        default=None,
        description="Chapter after which this fact is no longer valid; None means forever",
    )
    valid_until_chapter: int | None = Field(
        default=None,
        description="Alias for valid_to_chapter",
    )
    era_name: str | None = Field(default=None, description="Name of the visual era")
    danbooru_tags: list[str] | None = Field(default=None, description="List of Danbooru tags")
    flux_prompt: str | None = Field(default=None, description="Natural language prompt for Flux models")
    negative_tags: list[str] | None = Field(default=None, description="Negative anti-spoiler tags")
    confidence: Literal["confirmed", "likely", "speculative"] | str = Field(
        default="likely", description="Confidence level of this claim"
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_valid_chapters(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "valid_until_chapter" in data and "valid_to_chapter" not in data:
                data["valid_to_chapter"] = data["valid_until_chapter"]
            elif "valid_to_chapter" in data and "valid_until_chapter" not in data:
                data["valid_until_chapter"] = data["valid_to_chapter"]
        return data


class ResearchImageRef(BaseModel):
    """A visual reference image found during research."""

    image_url: str = Field(description="Original URL of the image")
    local_path: str | None = Field(
        default=None, description="Local filesystem path after download"
    )
    source_url: str = Field(description="Page URL where the image was found")
    entity_name: str | None = Field(
        default=None, description="Entity this image depicts"
    )
    category: Literal[
        "official_cover",
        "official_illustration",
        "community_art",
        "screenshot",
        "reference_photo",
    ] = Field(description="Classification of the image source")
    caption: str | None = Field(
        default=None, description="Caption or alt text for the image"
    )
    artist: str | None = Field(
        default=None, description="Artist or creator attribution"
    )
    license_status: Literal["unknown", "public_domain", "fair_use", "restricted"] = Field(
        default="unknown", description="Known license status of the image"
    )


class SpoilerSafeKnowledge(BaseModel):
    """The filtered knowledge bundle for an entity at a specific chapter boundary."""

    entity_id: str = Field(description="Canonical entity identifier")
    entity_name: str = Field(description="Display name of the entity")
    current_chapter: int = Field(
        description="Chapter boundary for spoiler filtering"
    )
    known_aliases: list[str] = Field(
        description="Only aliases revealed by current_chapter"
    )
    visual_claims: list[ResearchClaim] = Field(
        description="Only claims valid at current_chapter"
    )
    reference_images: list[ResearchImageRef] = Field(
        description="Reference images available at current_chapter"
    )
    novel_descriptions: list[str] = Field(
        description="Passage excerpts from chapters <= current_chapter"
    )


class ArcStructure(BaseModel):
    """Story arc structure parsed from wiki."""

    arc_name: str = Field(description="Human-readable arc name")
    arc_number: int = Field(description="Sequential arc number")
    start_chapter: int = Field(description="First chapter of this arc")
    end_chapter: int = Field(description="Last chapter of this arc")
    description: str | None = Field(
        default=None, description="Brief description of the arc"
    )
    key_characters: list[str] = Field(
        default_factory=list, description="Major characters in this arc"
    )
    key_locations: list[str] = Field(
        default_factory=list, description="Key locations featured in this arc"
    )
