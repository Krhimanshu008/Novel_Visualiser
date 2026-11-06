"""Novel Visualiser — Biological Evolution Auditor (Main Agent Reasoning Subsystem).

Responsible for:
1. Detecting character biological, physical, and supernatural transformations across arcs.
2. Promoting outgrown physical traits (e.g. 'scrawny', 'sallow skin') into negative constraint tags.
3. Activating new evolutionary physical traits (e.g. 'porcelain skin', 'alabaster skin', 'athletic build', 'ethereal glow').
4. Minting new BiologicalEpoch specifications and updating SQLite asset_versions.
5. Powered by high-end frontier reasoning AI via Kaggle Model Proxy.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Optional
from sqlalchemy.orm import Session

from app.database.models import AssetLevel, AssetVersion, Entity
from app.schemas.character_subagent import (
    ArchetypalDNA,
    BiologicalEpoch,
    CharacterMatrixSpec,
    DanbooruIdentityLock,
)

logger = logging.getLogger("novel_visualiser.evolution_auditor")


class BiologicalEvolutionAuditor:
    """Main Agent's high-end reasoning auditor for character physical maturation and growth."""

    def __init__(self, db: Optional[Session] = None, use_ai_thinking: bool = True):
        self.db = db
        self.use_ai_thinking = use_ai_thinking
        self._kaggle_client = None

        if self.use_ai_thinking:
            try:
                workspace_root = Path(__file__).resolve().parent.parent.parent.parent
                if str(workspace_root) not in sys.path:
                    sys.path.insert(0, str(workspace_root))
                from kaggle_ai_engine.proxy_client import KaggleModelProxyClient
                self._kaggle_client = KaggleModelProxyClient()
                logger.info("BiologicalEvolutionAuditor initialized with Kaggle Model Proxy client.")
            except Exception as exc:
                logger.warning("Could not initialize Kaggle Model Proxy: %s", exc)
                self._kaggle_client = None

    def audit_evolution(
        self,
        character_name: str,
        current_chapter: int,
        passage_text: str,
        current_spec: CharacterMatrixSpec,
    ) -> Optional[BiologicalEpoch]:
        """
        Audit novel text to determine if a character has undergone physical growth,
        dermal transformation (e.g. porcelain/alabaster glow), muscle gain, or aging.
        """
        current_epoch_name = (
            current_spec.active_epoch.epoch_name if current_spec.active_epoch else "Initial Slum State"
        )

        if self._kaggle_client:
            try:
                system_prompt = (
                    "You are the Lead Visual Character Supervisor and World Director for a major anime adaptation.\n"
                    "Analyze the novel passage for character physical development, growth spurts, aspect awakenings, and bodily transformations.\n\n"
                    "EVALUATION CRITERIA:\n"
                    "1. Did their age or maturation advance? (e.g., from 16yo malnourished boy to 18yo young adult man)\n"
                    "2. Did their nutrition and physique change? (e.g. from scrawny/malnourished to lean sinewy or athletic build)\n"
                    "3. Did their skin or complexion transform? (e.g. from sallow/grey outskirts skin to smooth porcelain/alabaster with an ethereal glow)\n"
                    "4. Did they gain supernatural manifestations? (e.g. obsidian shadow eyes, shadow mist, subtle ethereal presence)\n\n"
                    "CRITICAL NEGATIVE PROMOTION RULE:\n"
                    "Any physical trait that the character has OUTGROWN must be explicitly retired and placed into 'retired_negative_tags'\n"
                    "(e.g., if he is no longer scrawny, 'scrawny' and 'sallow skin' MUST be retired to negative tags so diffusion never draws them!)."
                )

                user_prompt = f"""
Character Name: {character_name}
Current Chapter: {current_chapter}
Current Active Epoch: {current_epoch_name}
Current Biology Tags: {current_spec.identity_lock.core_danbooru_tags}
Novel Passage Text:
\"\"\"{passage_text[:3000]}\"\"\"

Respond ONLY in valid JSON matching this exact structure:
{{
  "evolution_detected": true,
  "transformation_summary": "...",
  "new_epoch_id": "epoch_03_shadow_awakened",
  "new_epoch_name": "Forgotten Shore Shadow Awakened",
  "age_appearance": "17-18",
  "gender_tag": "1boy",
  "physique_tags": ["athletic slender build", "defined muscle tone"],
  "skin_complexion_tags": ["porcelain skin", "alabaster skin", "subtle ethereal glow", "smooth pale skin"],
  "ocular_fatigue_tags": ["sharp dark onyx eyes", "obsidian depth"],
  "supernatural_manifestation_tags": ["subtle shadow mist", "shadow affinity"],
  "retired_negative_tags": ["scrawny", "sallow skin", "sunken eyes", "malnourished", "chubby", "muscular giant"]
}}
"""
                resp_text = self._kaggle_client.chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                )

                clean_text = resp_text.strip()
                if "```json" in clean_text:
                    clean_text = clean_text.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_text:
                    clean_text = clean_text.split("```")[1].split("```")[0].strip()
                else:
                    start_idx = clean_text.find("{")
                    end_idx = clean_text.rfind("}")
                    if start_idx != -1 and end_idx != -1:
                        clean_text = clean_text[start_idx:end_idx + 1].strip()

                clean_text = re.sub(r",\s*([\]\}])", r"\1", clean_text)
                data = json.loads(clean_text)

                if not data.get("evolution_detected", False):
                    logger.info("Evolution audit complete: No biological changes detected for %s in Ch %d", character_name, current_chapter)
                    return None

                new_epoch = BiologicalEpoch(
                    epoch_id=data.get("new_epoch_id", f"epoch_ch{current_chapter}_{character_name.lower()}"),
                    epoch_name=data.get("new_epoch_name", f"Evolved State (Ch {current_chapter})"),
                    valid_from_chapter=current_chapter,
                    age_appearance=data.get("age_appearance", "17-18"),
                    physique_tags=data.get("physique_tags", ["athletic slender build"]),
                    skin_complexion_tags=data.get("skin_complexion_tags", ["porcelain skin", "alabaster skin", "subtle ethereal glow"]),
                    ocular_fatigue_tags=data.get("ocular_fatigue_tags", ["sharp dark onyx eyes"]),
                    supernatural_manifestation_tags=data.get("supernatural_manifestation_tags", ["subtle shadow mist"]),
                    retired_negative_tags=data.get("retired_negative_tags", ["scrawny", "sallow skin", "sunken eyes"]),
                )
                logger.info("Evolution audit DETECTED transformation: %s for %s", new_epoch.epoch_name, character_name)
                return new_epoch
            except Exception as exc:
                logger.warning("Kaggle AI biological evolution audit failed: %s; using heuristic check", exc)

        # Heuristic Rule-Based Check (e.g. keywords for awakening, forgotten shore, rank-up)
        lower_text = passage_text.lower()
        if "awakened" in lower_text or "divine shadow" in lower_text or "porcelain" in lower_text or "forgotten shore" in lower_text or current_chapter >= 19:
            return BiologicalEpoch(
                epoch_id=f"epoch_03_shadow_awakened_ch{current_chapter}",
                epoch_name="Forgotten Shore Shadow Awakened",
                valid_from_chapter=current_chapter,
                age_appearance="17-18",
                physique_tags=["athletic slender build", "defined muscle tone"],
                skin_complexion_tags=["porcelain skin", "alabaster skin", "subtle ethereal glow", "smooth flawless skin"],
                ocular_fatigue_tags=["sharp dark onyx eyes", "obsidian depth"],
                supernatural_manifestation_tags=["subtle shadow mist", "shadow affinity"],
                retired_negative_tags=["scrawny", "sallow skin", "sunken eyes", "malnourished", "chubby", "muscular giant"],
            )
        elif "slave" in lower_text or "nightmare" in lower_text or current_chapter >= 3:
            return BiologicalEpoch(
                epoch_id=f"epoch_02_slave_survivor_ch{current_chapter}",
                epoch_name="First Nightmare Slave Survivor",
                valid_from_chapter=current_chapter,
                age_appearance="16",
                physique_tags=["lean sinewy build", "hardened muscles"],
                skin_complexion_tags=["pale skin", "whip scars", "scratched skin"],
                ocular_fatigue_tags=["hollow eyes", "fierce survivalist glare"],
                supernatural_manifestation_tags=[],
                retired_negative_tags=["scrawny", "chubby", "fat"],
            )

        return None

    def apply_evolution_to_spec(
        self,
        current_spec: CharacterMatrixSpec,
        new_epoch: BiologicalEpoch,
        new_outfit: Optional[Any] = None,
    ) -> CharacterMatrixSpec:
        """Apply the newly audited biological epoch to the character matrix specification."""
        dna = current_spec.archetypal_dna

        # Assemble new Core Danbooru Biology Tags: [DNA] + [Epoch Age/Physique] + [Epoch Skin/Glow] + [Epoch Eyes/Aura]
        core_tags = [
            dna.gender_tag,
            dna.hair_color_dna,
            dna.eye_color_dna,
            dna.facial_bone_structure,
        ]
        core_tags.extend(new_epoch.physique_tags)
        core_tags.extend(new_epoch.skin_complexion_tags)
        core_tags.extend(new_epoch.ocular_fatigue_tags)
        core_tags.extend(new_epoch.supernatural_manifestation_tags)

        # Assemble new Negative Tags: [Permanent DNA Negatives] + [Retired Outgrown Tags]
        neg_tags = list(dna.permanent_negative_tags)
        for ret in new_epoch.retired_negative_tags:
            if ret not in neg_tags:
                neg_tags.append(ret)

        # Update Identity Lock
        current_spec.identity_lock.core_danbooru_tags = core_tags
        current_spec.identity_lock.negative_identity_tags = neg_tags
        current_spec.identity_lock.active_epoch = new_epoch
        current_spec.active_epoch = new_epoch

        if new_outfit:
            current_spec.active_outfit = new_outfit

        return current_spec

