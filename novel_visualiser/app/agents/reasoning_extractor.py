"""
Novel Visualiser — Universal Reasoning Extractor (Phase 1.5+)

Novel-agnostic, multi-era reasoning extractor.
Supports Groq LPU (DeepSeek-R1 / QwQ / Qwen3), Google Gemini (2.5 Flash/Pro Thinking),
OpenRouter, and a universal 0-token offline fallback that cross-references arc timelines.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("novel_visualiser.reasoning_extractor")


class ReasonedAlias(BaseModel):
    alias: str
    alias_type: str = "alternate_name"  # birth_name, title, persona, true_name, epithet
    valid_from_chapter: int = 1
    reveal_chapter: int = 1


class ReasonedVisualEra(BaseModel):
    era_name: str = "Initial Appearance"
    valid_from_chapter: int = 1
    valid_until_chapter: int | None = None
    claims: list[dict[str, str]] = Field(default_factory=list)  # [{"claim_type": "physique", "claim_text": "..."}]
    danbooru_tags: list[str] = Field(default_factory=list)
    flux_natural_prompt: str = ""
    anti_spoiler_negative_tags: list[str] = Field(default_factory=list)


class ReasonedEntityDossier(BaseModel):
    canonical_name: str
    entity_type: str = "character"
    genre_detected: str = "General Fiction"
    first_appearance_chapter: int = 1
    reasoning_trace: str = ""
    aliases: list[ReasonedAlias] = Field(default_factory=list)
    visual_eras: list[ReasonedVisualEra] = Field(default_factory=list)


SYSTEM_REASONING_PROMPT = """You are the Lead Canonical Visual Researcher for an AI Novel-to-Storyboard Visualiser.
Your job is to analyze raw wiki data and/or novel prose for ANY novel of ANY genre (Fantasy, Xianxia, Sci-Fi, Victorian Occult, LitRPG, Romance, etc.) and construct a strictly spoiler-gated, temporally accurate visual profile.

### CRITICAL RULES:
1. ARC-TO-CHAPTER TEMPORAL GROUNDING:
   - Use the provided `ARC_TIMELINE` to map vague volume or arc mentions (e.g., "In Volume 2...", "During the Academy Arc...") to exact integer `valid_from_chapter` and `valid_until_chapter` boundaries.
   - If a trait is permanent from debut, set `valid_from_chapter` to the entity's debut chapter and `valid_until_chapter` to `null`.
   - If a trait changes later (e.g., hair color changes, child -> adult, starter tunic -> endgame armor), set `valid_until_chapter` on the old era (`new_era_chapter - 1`) so old and new traits NEVER collide!
2. SPOILER ALIAS GATING:
   - Assign each alias a `reveal_chapter`. Secret identities, endgame titles, or late-story personas MUST have their true `reveal_chapter` estimated from the text or Arc Timeline.
3. DUAL PROMPT SYNTHESIS (DANBOORU + FLUX):
   - For each visual era, generate:
     a) `danbooru_tags`: Standard snake_case Danbooru tags (`1boy`, `solo`, `black_hair`, `pale_skin`, `victorian_clothes`, etc.).
     b) `flux_natural_prompt`: A vivid, concrete 1-2 sentence natural language visual description suitable for Flux.1 / video models.
     c) `anti_spoiler_negative_tags`: Snake_case tags of late-story transformations/scars/outfits that MUST NOT appear during this era.

Return ONLY valid JSON matching this exact structure (you may think inside <think>...</think> tags first):
{
  "canonical_name": "string",
  "entity_type": "character | location | creature",
  "genre_detected": "string",
  "first_appearance_chapter": 1,
  "aliases": [
    {"alias": "string", "alias_type": "title|persona|birth_name|true_name", "valid_from_chapter": 1, "reveal_chapter": 1}
  ],
  "visual_eras": [
    {
      "era_name": "string",
      "valid_from_chapter": 1,
      "valid_until_chapter": null,
      "claims": [
        {"claim_type": "physique|hair|eyes|attire|equipment|aura|environment|anatomy", "claim_text": "atomic visual fact"}
      ],
      "danbooru_tags": ["tag_1", "tag_2"],
      "flux_natural_prompt": "vivid visual prose",
      "anti_spoiler_negative_tags": ["future_spoiler_tag_1"]
    }
  ]
}"""


class UniversalReasoningExtractor:
    """
    Novel-agnostic Reasoning LLM Extractor.
    Supports Groq (DeepSeek-R1 / QwQ / Qwen3), Google Gemini (2.5 Flash/Pro Thinking), and OpenRouter.
    """

    def __init__(self, offline_mode: bool = False):
        self.offline_mode = offline_mode or os.getenv("USE_OFFLINE_EXTRACTOR", "0") == "1"
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")

        # Default reasoning models per provider
        self.groq_model = os.getenv("GROQ_REASONING_MODEL", "qwen-qwq-32b")
        self.gemini_model = os.getenv("GEMINI_REASONING_MODEL", "gemini-2.5-flash")
        self.openrouter_model = os.getenv("OPENROUTER_REASONING_MODEL", "deepseek/deepseek-r1")

    def analyze_entity(
        self,
        novel_title: str,
        entity_name: str,
        entity_type: str = "character",
        genre_style: str | None = None,
        debut_chapter: int = 1,
        raw_aliases: list[str] | None = None,
        raw_infobox: dict[str, str] | None = None,
        raw_prose: str = "",
        arc_timeline: list[dict[str, Any]] | None = None,
    ) -> ReasonedEntityDossier:
        """Runs a single unified Reasoning pass to build the temporal visual dossier."""
        user_payload = {
            "novel_title": novel_title,
            "genre_style": genre_style or "Auto-Detect from context",
            "entity_name": entity_name,
            "entity_type": entity_type,
            "debut_chapter_hint": debut_chapter,
            "arc_timeline": arc_timeline or [],
            "infobox_attributes": raw_infobox or {},
            "raw_aliases": raw_aliases or [],
            "appearance_and_lore_excerpt": raw_prose[:6000],  # Cap input to ~1,200 tokens max
        }

        if not self.offline_mode:
            raw_response = self._invoke_reasoning_llm(json.dumps(user_payload, ensure_ascii=False))
            if raw_response:
                parsed = self._parse_reasoning_output(raw_response, entity_name, entity_type, debut_chapter)
                if parsed:
                    return parsed

        # Universal Genre-Agnostic Offline Fallback (for 0-token CI/unit tests)
        return self._universal_offline_fallback(
            entity_name=entity_name,
            entity_type=entity_type,
            genre_style=genre_style or "General Fiction",
            debut_chapter=debut_chapter,
            raw_aliases=raw_aliases or [],
            raw_infobox=raw_infobox or {},
            raw_prose=raw_prose,
            arc_timeline=arc_timeline or [],
        )

    def _invoke_reasoning_llm(self, user_content: str) -> str | None:
        # 1. Try Groq Reasoning LPU first (Ultra-fast QwQ / DeepSeek-R1 / Qwen3)
        if self.groq_api_key:
            res = self._call_openai_compatible(
                url="https://api.groq.com/openai/v1/chat/completions",
                api_key=self.groq_api_key,
                model=self.groq_model,
                user_content=user_content,
            )
            if res:
                return res

        # 2. Try Google Gemini 2.5 Thinking
        if self.gemini_api_key:
            res = self._call_gemini(user_content)
            if res:
                return res

        # 3. Try OpenRouter Reasoning (DeepSeek-R1)
        if self.openrouter_api_key:
            res = self._call_openai_compatible(
                url="https://openrouter.ai/api/v1/chat/completions",
                api_key=self.openrouter_api_key,
                model=self.openrouter_model,
                user_content=user_content,
            )
            if res:
                return res

        return None

    def _call_openai_compatible(self, url: str, api_key: str, model: str, user_content: str) -> str | None:
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_REASONING_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.1,
            "max_tokens": 2500,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=35) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                msg = data["choices"][0]["message"]
                reasoning = msg.get("reasoning_content") or ""
                content = msg.get("content") or ""
                if reasoning and "<think>" not in content:
                    return f"<think>{reasoning}</think>\n{content}"
                return content
        except Exception as exc:
            logger.warning("Reasoning call failed for %s: %s", url, exc)
            return None

    def _call_gemini(self, user_content: str) -> str | None:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        payload = json.dumps({
            "systemInstruction": {"parts": [{"text": SYSTEM_REASONING_PROMPT}]},
            "contents": [{"parts": [{"text": user_content}]}],
            "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=35) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                parts = data["candidates"][0]["content"]["parts"]
                return "\n".join(p.get("text", "") for p in parts)
        except Exception as exc:
            logger.warning("Gemini reasoning call failed: %s", exc)
            return None

    def _parse_reasoning_output(
        self, raw_text: str, default_name: str, default_type: str, default_debut: int
    ) -> ReasonedEntityDossier | None:
        think_match = re.search(r"<think>(.*?)</think>", raw_text, re.DOTALL)
        reasoning_trace = think_match.group(1).strip() if think_match else "Direct structured reasoning completed."
        clean_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if not json_match:
            return None

        try:
            obj = json.loads(json_match.group(0))
            obj.setdefault("canonical_name", default_name)
            obj.setdefault("entity_type", default_type)
            obj.setdefault("first_appearance_chapter", default_debut)
            obj["reasoning_trace"] = reasoning_trace[:1500]
            return ReasonedEntityDossier.model_validate(obj)
        except Exception as exc:
            logger.warning("Failed to parse reasoning JSON: %s", exc)
            return None

    def _universal_offline_fallback(
        self,
        entity_name: str,
        entity_type: str,
        genre_style: str,
        debut_chapter: int,
        raw_aliases: list[str],
        raw_infobox: dict[str, str],
        raw_prose: str,
        arc_timeline: list[dict[str, Any]],
    ) -> ReasonedEntityDossier:
        """
        Universal, novel-agnostic rule engine used when running offline tests (0 API tokens).
        Uses Arc Timeline matching + inline chapter regex (`Ch. X`, `Volume Y`) for any novel.
        """
        vol_map: dict[int, tuple[int, int]] = {}
        arc_name_map: dict[str, tuple[int, int]] = {}
        for arc in arc_timeline:
            v_num = int(arc.get("arc_number", 1))
            s_ch = int(arc.get("start_chapter", 1))
            e_ch = int(arc.get("end_chapter", 9999))
            vol_map[v_num] = (s_ch, e_ch)
            if arc.get("name"):
                arc_name_map[str(arc["name"]).lower()] = (s_ch, e_ch)

        # 1. Parse aliases with inline chapter or volume annotations (e.g., "Gehrman Sparrow (Vol. 3)" or "Mongrel (Ch. 25)")
        reasoned_aliases: list[ReasonedAlias] = []
        for raw_a in raw_aliases:
            ch_reveal = debut_chapter
            m_ch = re.search(r"(?:ch(?:apter)?\.?\s*)(\d+)", raw_a, re.IGNORECASE)
            m_vol = re.search(r"(?:vol(?:ume)?\.?\s*)(\d+)", raw_a, re.IGNORECASE)
            if m_ch:
                ch_reveal = int(m_ch.group(1))
            elif m_vol and int(m_vol.group(1)) in vol_map:
                ch_reveal = vol_map[int(m_vol.group(1))][0]

            clean_a = re.sub(r"\s*\(.*?\)", "", raw_a).strip()
            if clean_a:
                reasoned_aliases.append(
                    ReasonedAlias(alias=clean_a, alias_type="alternate_name", valid_from_chapter=ch_reveal, reveal_chapter=ch_reveal)
                )

        # 2. Split prose into sentences and group by detected temporal markers (Chapter X or Volume Y)
        eras_by_chapter: dict[int, list[str]] = {debut_chapter: []}
        for sentence in re.split(r"(?<=[.!?])\s+", raw_prose):
            s_clean = re.sub(r"\[\d+\]", "", sentence).strip()
            if len(s_clean) < 12:
                continue

            detected_ch = debut_chapter
            m_ch = re.search(r"\bchapter\s+(\d+)\b", s_clean, re.IGNORECASE)
            m_vol = re.search(r"\bvolume\s+(\d+)\b", s_clean, re.IGNORECASE)
            if m_ch:
                detected_ch = int(m_ch.group(1))
            elif m_vol and int(m_vol.group(1)) in vol_map:
                detected_ch = vol_map[int(m_vol.group(1))][0]
            else:
                for arc_title, (s_ch, _) in arc_name_map.items():
                    if arc_title in s_clean.lower():
                        detected_ch = s_ch
                        break

            eras_by_chapter.setdefault(detected_ch, []).append(s_clean)

        # Build chronological visual eras
        sorted_chapters = sorted(eras_by_chapter.keys())
        visual_eras: list[ReasonedVisualEra] = []

        for idx, ch in enumerate(sorted_chapters):
            sentences = eras_by_chapter[ch]
            until_ch = (sorted_chapters[idx + 1] - 1) if idx + 1 < len(sorted_chapters) else None

            claims: list[dict[str, str]] = []
            danbooru: list[str] = ["1boy" if entity_type == "character" else "scenery", "solo"]
            flux_parts: list[str] = []

            for s in sentences:
                claims.append({"claim_type": "appearance", "claim_text": s})
                flux_parts.append(s)

                s_lower = s.lower()
                for kw, tag in [
                    ("black hair", "black_hair"),
                    ("silver hair", "silver_hair"),
                    ("white hair", "white_hair"),
                    ("pale skin", "pale_skin"),
                    ("black eyes", "black_eyes"),
                    ("dark eyes", "black_eyes"),
                    ("brown eyes", "brown_eyes"),
                    ("emerald eyes", "emerald_eyes"),
                    ("emerald eyes", "green_eyes"),
                    ("silver eyes", "grey_eyes"),
                    ("top hat", "top_hat"),
                    ("frock coat", "frock_coat"),
                    ("windbreaker", "black_windbreaker"),
                    ("spectacles", "gold_rimmed_glasses"),
                    ("glasses", "glasses"),
                    ("white robes", "white_robes"),
                    ("white robes", "white_robe"),
                    ("white robe", "white_robe"),
                    ("robe", "robe"),
                    ("armor", "armor"),
                    ("mask", "mask"),
                ]:
                    if kw in s_lower and tag not in danbooru:
                        danbooru.append(tag)

            if raw_infobox and idx == 0:
                for k, v in raw_infobox.items():
                    claims.append({"claim_type": "core_trait", "claim_text": f"{k}: {v}"})

            era_title = f"Debut Era (Ch. {ch}+)" if ch == debut_chapter else f"Evolution Phase (Ch. {ch}+)"
            visual_eras.append(
                ReasonedVisualEra(
                    era_name=era_title,
                    valid_from_chapter=ch,
                    valid_until_chapter=until_ch,
                    claims=claims,
                    danbooru_tags=danbooru,
                    flux_natural_prompt=" ".join(flux_parts[:3]) if flux_parts else f"Visual description for {entity_name}",
                    anti_spoiler_negative_tags=[],
                )
            )

        # Invert future tags into earlier era negative tags
        all_future_tags = set()
        for era in reversed(visual_eras):
            era.anti_spoiler_negative_tags = list(all_future_tags - set(era.danbooru_tags))
            all_future_tags.update(era.danbooru_tags)

        return ReasonedEntityDossier(
            canonical_name=entity_name,
            entity_type=entity_type,
            genre_detected=genre_style,
            first_appearance_chapter=debut_chapter,
            reasoning_trace=f"Universal offline heuristic extracted {len(visual_eras)} eras across {len(sorted_chapters)} chapter boundaries.",
            aliases=reasoned_aliases,
            visual_eras=visual_eras,
        )
