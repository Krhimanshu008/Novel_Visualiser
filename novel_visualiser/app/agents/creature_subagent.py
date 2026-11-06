"""Novel Visualiser — Creature, Monster & Power System Continuity Subagent.

Autonomous Subagent responsible for:
1. Identifying nightmare creatures, beasts, and hero memories/items in novel chapters.
2. Auditing Power System classification across the 2-axis hierarchy:
   - Axis 1: Soul Rank (Dormant, Awakened, Fallen, Corrupted, Great, Cursed, Unholy)
   - Axis 2: Class Hierarchy (Beast, Monster, Demon, Devil, Tyrant, Terror, Titan)
3. Conducting 5-dimension Deep Observational Reasoning on creature biology, soul cores, and scale.
4. Compiling the standardized 4-Sheet Bestiary Reference Matrix:
   - Sheet 1: Canonical Orthographic Turnaround
   - Sheet 2: 4-Threat Action & Roar Matrix
   - Sheet 3: Carapace, Weak Points & Soul Core Detail
   - Sheet 4: Human-to-Monster Scale Comparison Plate
5. Auto-cropping predatory head reference chips for ComfyUI IP-Adapter Plus consistency.
6. Managing Hero Memories, Weapons, and Artifact item continuity cards.
7. Resolving runtime battle anchors and spatial positioning for combat shots.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.database.models import (
    AssetLevel,
    AssetVersion,
    Entity,
    EntityType,
    RefType,
    VisualRef,
)
from app.schemas.creature_subagent import (
    CreatureCard,
    CreatureClassEnum,
    CreatureDeepObservation,
    CreatureRankEnum,
    CreatureScaleSpec,
    CreatureTaxonomyEnum,
    ItemPropCard,
    ShotCreatureAnchor,
    SingleBestiaryPlateSpec,
)
from app.schemas.location_subagent import ArcHierarchyTag
from app.services.comfyui_client import ComfyUIWorkflowCompiler, DualGPUComfyUIDispatcher

logger = logging.getLogger("novel_visualiser.creature_subagent")


class CreatureMonsterSubagent:
    """Autonomous Subagent dedicated to monsters, nightmare creatures, and power system canon."""

    def __init__(
        self,
        db: Optional[Session] = None,
        comfy_dispatcher: Optional[DualGPUComfyUIDispatcher] = None,
        use_ai_thinking: bool = True,
    ):
        """Initialize CreatureMonsterSubagent with database, dispatcher, and reasoning engine."""
        self.db = db
        self.comfy_dispatcher = comfy_dispatcher
        self.use_ai_thinking = use_ai_thinking
        self._kaggle_client = None

        if self.use_ai_thinking:
            try:
                workspace_root = Path(__file__).resolve().parent.parent.parent.parent
                if str(workspace_root) not in sys.path:
                    sys.path.insert(0, str(workspace_root))
                from kaggle_ai_engine.proxy_client import KaggleModelProxyClient
                self._kaggle_client = KaggleModelProxyClient()
                logger.info("CreatureMonsterSubagent initialized with Kaggle Model Proxy reasoning engine.")
            except Exception as exc:
                logger.warning("Could not initialize Kaggle Model Proxy: %s. Using heuristic analysis.", exc)
                self._kaggle_client = None

    # =========================================================================
    # 1. HISTORICAL CANON & BESTIARY MEMORY LOOKUP
    # =========================================================================

    def query_historical_creature(
        self,
        project_id: str,
        creature_name: str,
        chapter_number: int = 1,
    ) -> Optional[CreatureCard]:
        """Query SQLite memory to fetch an existing creature card valid at this chapter."""
        if not self.db:
            return None

        clean_name = creature_name.strip().lower()

        entity = (
            self.db.query(Entity)
            .filter(
                Entity.project_id == project_id,
                Entity.entity_type == EntityType.CREATURE,
                Entity.canonical_name.ilike(f"%{clean_name}%"),
            )
            .first()
        )

        if not entity:
            return None

        latest_version = (
            self.db.query(AssetVersion)
            .filter(
                AssetVersion.entity_id == entity.id,
                AssetVersion.valid_from_chapter <= chapter_number,
            )
            .order_by(AssetVersion.valid_from_chapter.desc(), AssetVersion.created_at.desc())
            .first()
        )

        if not latest_version or not latest_version.visual_attributes:
            return None

        try:
            attrs = latest_version.visual_attributes
            if "creature_card" in attrs:
                return CreatureCard.model_validate(attrs["creature_card"])
        except Exception as exc:
            logger.warning("Failed to parse cached creature card: %s", exc)

        return None

    def query_historical_item(
        self,
        project_id: str,
        item_name: str,
        chapter_number: int = 1,
    ) -> Optional[ItemPropCard]:
        """Query SQLite memory to fetch an existing hero item / memory card."""
        if not self.db:
            return None

        clean_name = item_name.strip().lower()

        entity = (
            self.db.query(Entity)
            .filter(
                Entity.project_id == project_id,
                Entity.entity_type == EntityType.ITEM,
                Entity.canonical_name.ilike(f"%{clean_name}%"),
            )
            .first()
        )

        if not entity:
            return None

        latest_version = (
            self.db.query(AssetVersion)
            .filter(
                AssetVersion.entity_id == entity.id,
                AssetVersion.valid_from_chapter <= chapter_number,
            )
            .order_by(AssetVersion.valid_from_chapter.desc())
            .first()
        )

        if not latest_version or not latest_version.visual_attributes:
            return None

        try:
            attrs = latest_version.visual_attributes
            if "item_card" in attrs:
                return ItemPropCard.model_validate(attrs["item_card"])
        except Exception as exc:
            logger.warning("Failed to parse cached item card: %s", exc)

        return None

    # =========================================================================
    # 2. 5-DIMENSION EXTENDED OBSERVATIONAL REASONING
    # =========================================================================

    def observe_and_synthesize_creature(
        self,
        creature_name: str,
        passage_text: str,
        arc_hierarchy: ArcHierarchyTag,
        chapter_number: int = 1,
        creature_id: Optional[str] = None,
    ) -> CreatureCard:
        """
        Conduct deep 5-dimension observational reasoning on creature biology, soul cores, and scale.
        """
        import re
        clean_creature = re.sub(r"[^a-zA-Z0-9_]+", "_", creature_name.lower()).strip("_")
        cid = creature_id or f"creature_{clean_creature[:32]}"

        if self._kaggle_client:
            try:
                system_prompt = (
                    "You are an Elite Anime Visual Director, Creature Designer, and Power System Architect for a dark fantasy production.\n"
                    "Analyze the novel scene text and perform a deep 5-dimension observational analysis on the monster:\n"
                    "1. Power System & Soul Anatomy: Determine Soul Rank (Dormant, Awakened, Fallen, Corrupted, Great, Cursed, Unholy), "
                    "Class (Beast=1 core, Monster=2 cores, Demon=3 cores, Devil=4 cores, Tyrant=5 cores, Terror=6 cores, Titan=7 cores), "
                    "and visible soul aura / miasma.\n"
                    "2. Skeletal Scale & Mass: Physical length/height in meters, mass density, and visual ratio against a 1.75m human silhouette (e.g. '1:4', '1:10').\n"
                    "3. Biomaterial & Integument: Carapace, obsidian scales, bone armor, rot, multiple eyes, claws, stingers.\n"
                    "4. Locomotion & Weaponry: Kinetic attack speed, striking mandibles, coiled posture, predatory behavior.\n"
                    "5. Environmental Corruption: Acid melting stone, freezing frost aura, black shadow tendrils.\n\n"
                    "Then compile the standardized 4-Sheet Bestiary Reference Matrix (Danbooru tags, dark fantasy anime aesthetic):\n"
                    "- Sheet 1: Canonical Orthographic Turnaround (front, 3/4, profile, back on solid neutral grey #2b2b2b, 1024x1024)\n"
                    "- Sheet 2: 4-Threat Action & Roar Matrix (neutral stalk, roaring maw, striking attack, coiled dormant, 1024x1024)\n"
                    "- Sheet 3: Carapace, Weak Points & Soul Core Detail (close-up chitin plates, glowing soul core, 1024x1024)\n"
                    "- Sheet 4: Scale Comparison Plate (monster beside a 1.75m human silhouette with height markers, 1024x1024)\n"
                    "CRITICAL: Include strict negative identity tags to prevent humanoid, anime girl, or cute pet drift."
                )

                user_prompt = f"""
Creature Name: {creature_name}
Arc Hierarchy: Main Arc={arc_hierarchy.main_arc_id}, Sub Arc={arc_hierarchy.sub_arc_id}, Mini Arc={arc_hierarchy.mini_arc_id}
Novel Passage Text:
\"\"\"{passage_text[:3000]}\"\"\"

Allowed 'rank' values: ['dormant', 'awakened', 'fallen', 'corrupted', 'great', 'cursed', 'unholy']
Allowed 'creature_class' values: ['beast', 'monster', 'demon', 'devil', 'tyrant', 'terror', 'titan']
Allowed 'taxonomy' values: ['insectoid', 'reptilian_draconic', 'abyssal_aquatic', 'undead_necrotic', 'mammalian_beast', 'eldritch_aberration', 'elemental']

Respond ONLY in valid JSON matching this exact structure:
{{
  "observation": {{
    "power_system_soul_anatomy": "...",
    "skeletal_scale_mass": "...",
    "biomaterial_integument": "...",
    "locomotion_kinesics_weaponry": "...",
    "environmental_corruption": "..."
  }},
  "rank": "awakened",
  "creature_class": "beast",
  "taxonomy": "insectoid",
  "soul_core_count": 1,
  "scale_spec": {{
    "estimated_length_meters": 6.5,
    "estimated_height_meters": 2.2,
    "human_scale_ratio": "1:3",
    "danbooru_scale_tags": ["large creature", "towering", "massive centipede"]
  }},
  "core_danbooru_biology_tags": ["segmented black chitin", "multiple legs", "mandibles", "cluster of red eyes"],
  "supernatural_aura_tags": ["dark mist", "oozing green acid", "faint soul flame"],
  "negative_identity_tags": ["1boy", "1girl", "human face", "cute", "fluffy", "anime girl", "bipedal human"],
  "sheet_1_turnaround_prompt": "masterpiece, best quality, bestiary reference sheet, orthographic turnaround, ...",
  "sheet_2_action_threat_prompt": "masterpiece, best quality, threat matrix, 4-panel roar and attack sheet, ...",
  "sheet_3_carapace_detail_prompt": "masterpiece, best quality, anatomical detail sheet, close-up chitin armor, ...",
  "sheet_4_scale_comparison_prompt": "masterpiece, best quality, scale comparison sheet, monster beside human silhouette, ..."
}}
"""

                raw_resp = self._kaggle_client.chat_completion(
                    messages=[
                        {"role": "user", "content": f"System: {system_prompt}\nUser: {user_prompt}"}
                    ],
                    temperature=0.2,
                )

                clean_text = raw_resp.strip()
                if clean_text.startswith("```"):
                    clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
                    clean_text = re.sub(r"\s*```$", "", clean_text)
                clean_text = re.sub(r",\s*([\]\}])", r"\1", clean_text)

                parsed = json.loads(clean_text)

                # Robust Enums
                rank_str = str(parsed.get("rank", "awakened")).lower().strip()
                matched_rank = CreatureRankEnum.AWAKENED
                for r in CreatureRankEnum:
                    if r.value == rank_str or r.name.lower() == rank_str:
                        matched_rank = r
                        break

                class_str = str(parsed.get("creature_class", "beast")).lower().strip()
                matched_class = CreatureClassEnum.BEAST
                for c in CreatureClassEnum:
                    if c.value == class_str or c.name.lower() == class_str:
                        matched_class = c
                        break

                tax_str = str(parsed.get("taxonomy", "insectoid")).lower().strip()
                matched_tax = CreatureTaxonomyEnum.INSECTOID
                for t in CreatureTaxonomyEnum:
                    if t.value == tax_str or t.name.lower() == tax_str:
                        matched_tax = t
                        break

                obs = CreatureDeepObservation(
                    power_system_soul_anatomy=parsed.get("observation", {}).get("power_system_soul_anatomy", ""),
                    skeletal_scale_mass=parsed.get("observation", {}).get("skeletal_scale_mass", ""),
                    biomaterial_integument=parsed.get("observation", {}).get("biomaterial_integument", ""),
                    locomotion_kinesics_weaponry=parsed.get("observation", {}).get("locomotion_kinesics_weaponry", ""),
                    environmental_corruption=parsed.get("observation", {}).get("environmental_corruption", ""),
                )

                scale_data = parsed.get("scale_spec", {})
                scale_spec = CreatureScaleSpec(
                    estimated_length_meters=float(scale_data.get("estimated_length_meters", 4.0)),
                    estimated_height_meters=float(scale_data.get("estimated_height_meters", 2.0)),
                    human_scale_ratio=str(scale_data.get("human_scale_ratio", "1:2")),
                    danbooru_scale_tags=scale_data.get("danbooru_scale_tags", ["large creature", "towering"]),
                )

                neg_default = "worst quality, low quality, blurry, 1boy, 1girl, human, cute, deformed"

                sheet_1 = SingleBestiaryPlateSpec(
                    sheet_index=1,
                    title="Orthographic Turnaround",
                    role="TURNAROUND",
                    prompt=parsed.get("sheet_1_turnaround_prompt", f"masterpiece, bestiary turnaround, {creature_name}"),
                    negative_prompt=neg_default,
                )
                sheet_2 = SingleBestiaryPlateSpec(
                    sheet_index=2,
                    title="Threat & Roar Matrix",
                    role="ACTION_MATRIX",
                    prompt=parsed.get("sheet_2_action_threat_prompt", f"masterpiece, roar and attack matrix, {creature_name}"),
                    negative_prompt=neg_default,
                )
                sheet_3 = SingleBestiaryPlateSpec(
                    sheet_index=3,
                    title="Carapace & Weak Point Detail",
                    role="CARAPACE_DETAIL",
                    prompt=parsed.get("sheet_3_carapace_detail_prompt", f"masterpiece, chitin detail, {creature_name}"),
                    negative_prompt=neg_default,
                )
                sheet_4 = SingleBestiaryPlateSpec(
                    sheet_index=4,
                    title="Human Scale Comparison",
                    role="SCALE_COMPARISON",
                    prompt=parsed.get("sheet_4_scale_comparison_prompt", f"masterpiece, scale comparison, {creature_name} beside human silhouette"),
                    negative_prompt=neg_default,
                )

                card = CreatureCard(
                    creature_id=cid,
                    creature_name=creature_name,
                    novel_arc=arc_hierarchy,
                    valid_from_chapter=chapter_number,
                    valid_to_chapter=None,
                    rank=matched_rank,
                    creature_class=matched_class,
                    taxonomy=matched_tax,
                    soul_core_count=int(parsed.get("soul_core_count", 1)),
                    observation=obs,
                    scale_spec=scale_spec,
                    core_danbooru_biology_tags=parsed.get("core_danbooru_biology_tags", []),
                    supernatural_aura_tags=parsed.get("supernatural_aura_tags", []),
                    negative_identity_tags=parsed.get("negative_identity_tags", []),
                    sheet_1_turnaround=sheet_1,
                    sheet_2_action_threat=sheet_2,
                    sheet_3_carapace_detail=sheet_3,
                    sheet_4_scale_comparison=sheet_4,
                    head_reference_chip_path=None,
                )
                logger.info("Successfully synthesized CreatureCard for %s via AI Thinking.", creature_name)
                return card

            except Exception as exc:
                logger.warning("AI Creature analysis failed (%s); falling back to heuristic engine.", exc)

        return self._heuristic_creature_synthesis(
            creature_name=creature_name,
            passage_text=passage_text,
            arc_hierarchy=arc_hierarchy,
            cid=cid,
            chapter_number=chapter_number,
        )

    # =========================================================================
    # 3. HEURISTIC FALLBACK SYNTHESIS
    # =========================================================================

    def _heuristic_creature_synthesis(
        self,
        creature_name: str,
        passage_text: str,
        arc_hierarchy: ArcHierarchyTag,
        cid: str,
        chapter_number: int,
    ) -> CreatureCard:
        """Deterministic rule-based fallback for creature and power system modeling."""
        text = passage_text.lower()

        # Deduce Rank
        if "unholy" in text or "titan" in text:
            rank = CreatureRankEnum.UNHOLY
            cls_ = CreatureClassEnum.TITAN
            cores = 7
        elif "corrupted" in text or "calamity" in text:
            rank = CreatureRankEnum.CORRUPTED
            cls_ = CreatureClassEnum.TYRANT
            cores = 5
        elif "fallen" in text:
            rank = CreatureRankEnum.FALLEN
            cls_ = CreatureClassEnum.DEMON
            cores = 3
        elif "awakened" in text:
            rank = CreatureRankEnum.AWAKENED
            cls_ = CreatureClassEnum.BEAST
            cores = 1
        else:
            rank = CreatureRankEnum.AWAKENED
            cls_ = CreatureClassEnum.BEAST
            cores = 1

        # Deduce Taxonomy
        if any(w in text for w in ["centipede", "spider", "larva", "chitin", "mandible", "insect"]):
            tax = CreatureTaxonomyEnum.INSECTOID
            length, height = 5.0, 1.8
            ratio = "1:3"
        elif any(w in text for w in ["dragon", "lizard", "scale", "serpent"]):
            tax = CreatureTaxonomyEnum.REPTILIAN_DRACONIC
            length, height = 8.0, 3.5
            ratio = "1:5"
        elif any(w in text for w in ["bone", "skeleton", "decay", "rot"]):
            tax = CreatureTaxonomyEnum.UNDEAD_NECROTIC
            length, height = 3.0, 2.5
            ratio = "1:2"
        else:
            tax = CreatureTaxonomyEnum.ELDRITCH_ABERRATION
            length, height = 4.0, 2.0
            ratio = "1:2"

        obs = CreatureDeepObservation(
            power_system_soul_anatomy=f"Power Rank: {rank.value.title()}, Class: {cls_.value.title()} ({cores} Soul Cores). Emits a necrotic dark miasma.",
            skeletal_scale_mass=f"Estimated length: {length}m, height: {height}m. Scale ratio approx {ratio} against human warrior.",
            biomaterial_integument="Heavy interlocking segmented obsidian chitin carapace with serrated bone spurs and multiple glowing red ocular clusters.",
            locomotion_kinesics_weaponry="Multipedal rapid scuttling; predatory lunging strikes utilizing triple-jointed serrated mandibles.",
            environmental_corruption="Corrosive acidic secretions sizzling against stone surfaces; dark shadow mist drifting from joint plates.",
        )

        scale = CreatureScaleSpec(
            estimated_length_meters=length,
            estimated_height_meters=height,
            human_scale_ratio=ratio,
            danbooru_scale_tags=["large monster", "towering", "massive creature"],
        )

        bio_tags = [
            "monster", "dark fantasy creature", "segmented black chitin", "multiple legs",
            "sharp mandibles", "glowing red eyes", "spikes", "carapace"
        ]
        aura_tags = ["dark mist", "shadow aura", "acid droplets", "menacing aura"]

        neg = "worst quality, low quality, 1boy, 1girl, human, cute, fluffy, deformed limbs"

        sheet_1 = SingleBestiaryPlateSpec(
            sheet_index=1,
            title="Orthographic Turnaround",
            role="TURNAROUND",
            prompt=f"masterpiece, best quality, bestiary reference sheet, orthographic turnaround, front, side, back, {creature_name}, solid neutral grey background #2b2b2b, {', '.join(bio_tags)}",
            negative_prompt=neg,
        )
        sheet_2 = SingleBestiaryPlateSpec(
            sheet_index=2,
            title="Threat & Roar Matrix",
            role="ACTION_MATRIX",
            prompt=f"masterpiece, best quality, threat matrix, 4-panel action grid, roaring maw, striking attack, {creature_name}, {', '.join(aura_tags)}",
            negative_prompt=neg,
        )
        sheet_3 = SingleBestiaryPlateSpec(
            sheet_index=3,
            title="Carapace & Weak Point Detail",
            role="CARAPACE_DETAIL",
            prompt=f"masterpiece, best quality, anatomical detail plate, macro close-up chitin armor, weak spot, soul core, {creature_name}",
            negative_prompt=neg,
        )
        sheet_4 = SingleBestiaryPlateSpec(
            sheet_index=4,
            title="Human Scale Comparison",
            role="SCALE_COMPARISON",
            prompt=f"masterpiece, best quality, scale comparison sheet, colossal {creature_name} standing beside standard 1.75m human silhouette, height markers",
            negative_prompt=neg,
        )

        return CreatureCard(
            creature_id=cid,
            creature_name=creature_name,
            novel_arc=arc_hierarchy,
            valid_from_chapter=chapter_number,
            valid_to_chapter=None,
            rank=rank,
            creature_class=cls_,
            taxonomy=tax,
            soul_core_count=cores,
            observation=obs,
            scale_spec=scale,
            core_danbooru_biology_tags=bio_tags,
            supernatural_aura_tags=aura_tags,
            negative_identity_tags=["1boy", "1girl", "human face", "cute", "fluffy", "anime girl", "bipedal human"],
            sheet_1_turnaround=sheet_1,
            sheet_2_action_threat=sheet_2,
            sheet_3_carapace_detail=sheet_3,
            sheet_4_scale_comparison=sheet_4,
            head_reference_chip_path=None,
        )

    # =========================================================================
    # 4. HERO PROP & MEMORY WEAPON SYNTHESIS
    # =========================================================================

    def observe_and_synthesize_item(
        self,
        item_name: str,
        passage_text: str,
        arc_hierarchy: ArcHierarchyTag,
        item_type: str = "weapon",
        chapter_number: int = 1,
    ) -> ItemPropCard:
        """Synthesize a canonical ItemPropCard for Memories, weapons, and relics."""
        item_id = f"prop_{item_name.lower().replace(' ', '_').replace('-', '_')[:32]}"
        text = passage_text.lower()

        # Deducing material & rank
        if "obsidian" in text or "black glass" in text or "midnight" in text:
            mat = "Mirror-polished razor obsidian volcanic glass with etched silver runic channels"
            tags = ["holding sword", "black blade", "obsidian sword", "single-edged blade", "dark mist", "glowing runes"]
        elif "iron" in text or "steel" in text:
            mat = "Heavy folded carbon steel with battle notches and weathered brass guard"
            tags = ["holding sword", "steel sword", "crossguard", "metallic sheen"]
        else:
            mat = "Enchanted alloy infused with ethereal soul essence"
            tags = ["holding weapon", "magical weapon", "ethereal glow"]

        rank = "Awakened" if "awakened" in text else ("Dormant" if "dormant" in text else "Ascended")

        return ItemPropCard(
            item_id=item_id,
            item_name=item_name,
            item_type=item_type,
            rank_tier=rank,
            material_physics=mat,
            dimensions="85cm single-edged straight blade, 3.5cm width, 0.9kg mass",
            danbooru_item_tags=tags,
            negative_item_tags=["broken blade", "rust", "wooden sword", "blunt", "cracked"],
            reference_plate_path=None,
        )

    # =========================================================================
    # 5. RENDERING BESTIARY MATRIX & PREDATORY HEAD CHIP
    # =========================================================================

    def render_bestiary_matrix_and_head_chip(
        self,
        card: CreatureCard,
        output_dir: Path,
        deterministic_seed: int = 2000,
    ) -> CreatureCard:
        """
        Generate bestiary visual plates and auto-crop the predatory head reference chip for IP-Adapter Plus.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate / Render Sheet 1 (Turnaround) Canvas
        sheet1_path = output_dir / f"{card.creature_id}_sheet1_turnaround.png"
        canvas = np.full((1024, 1024, 3), (43, 43, 43), dtype=np.uint8)  # Studio grey #2b2b2b

        # Render stylistic creature silhouette / wireframe
        cv2.ellipse(canvas, (512, 540), (280, 160), 0, 0, 360, (20, 20, 20), -1)
        cv2.circle(canvas, (320, 480), 90, (30, 30, 30), -1)  # Predatory Head
        # Glowing crimson eyes
        cv2.circle(canvas, (290, 460), 12, (30, 30, 220), -1)
        cv2.circle(canvas, (340, 460), 12, (30, 30, 220), -1)
        cv2.circle(canvas, (315, 490), 8, (30, 30, 255), -1)
        # Mandibles / Claws
        cv2.line(canvas, (280, 520), (250, 570), (10, 10, 180), 8)
        cv2.line(canvas, (350, 520), (380, 570), (10, 10, 180), 8)

        # Header metadata
        cv2.putText(
            canvas,
            f"BESTIARY CANON: {card.creature_name.upper()} [{card.rank.value.upper()} {card.creature_class.value.upper()}]",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Taxonomy: {card.taxonomy.value.upper()} | Cores: {card.soul_core_count} | Scale: {card.scale_spec.human_scale_ratio} Human",
            (40, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

        cv2.imwrite(str(sheet1_path), canvas)
        card.sheet_1_turnaround.image_path = str(sheet1_path)

        # 2. Auto-Crop Predatory Head Reference Chip for IP-Adapter Plus
        head_chip_path = output_dir / f"{card.creature_id}_predatory_head_chip.png"
        head_crop = canvas[360:600, 200:440]
        cv2.imwrite(str(head_chip_path), head_crop)
        card.head_reference_chip_path = str(head_chip_path)
        logger.info("Auto-cropped predatory head reference chip to %s", head_chip_path)

        # 3. Generate Sheet 4 (Scale Comparison Plate)
        sheet4_path = output_dir / f"{card.creature_id}_sheet4_scale_comparison.png"
        scale_canvas = np.full((1024, 1024, 3), (35, 35, 35), dtype=np.uint8)

        # Human silhouette (1.75m standard reference)
        human_x, human_y = 180, 650
        cv2.circle(scale_canvas, (human_x, human_y - 140), 20, (180, 180, 180), -1) # Head
        cv2.rectangle(scale_canvas, (human_x - 15, human_y - 120), (human_x + 15, human_y - 40), (180, 180, 180), -1) # Torso
        cv2.line(scale_canvas, (human_x - 8, human_y - 40), (human_x - 12, human_y + 40), (180, 180, 180), 6) # Leg L
        cv2.line(scale_canvas, (human_x + 8, human_y - 40), (human_x + 12, human_y + 40), (180, 180, 180), 6) # Leg R
        cv2.putText(scale_canvas, "Human (1.75m)", (human_x - 55, human_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Monster massive silhouette
        monster_center_x = 650
        cv2.ellipse(scale_canvas, (monster_center_x, 500), (280, 220), 0, 0, 360, (15, 15, 15), -1)
        cv2.putText(
            scale_canvas,
            f"{card.creature_name} ({card.scale_spec.estimated_length_meters:.1f}m)",
            (monster_center_x - 140, 750),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 100, 100),
            2,
        )

        cv2.imwrite(str(sheet4_path), scale_canvas)
        card.sheet_4_scale_comparison.image_path = str(sheet4_path)

        return card

    # =========================================================================
    # 6. DATABASE PERSISTENCE
    # =========================================================================

    def persist_creature_card(
        self,
        project_id: str,
        card: CreatureCard,
    ) -> bool:
        """Persist CreatureCard to SQLite Entity and AssetVersion tables."""
        if not self.db:
            return False

        try:
            # Query or Create Creature Entity
            entity = (
                self.db.query(Entity)
                .filter(
                    Entity.project_id == project_id,
                    Entity.canonical_name == card.creature_name,
                )
                .first()
            )

            if not entity:
                entity = Entity(
                    id=card.creature_id,
                    project_id=project_id,
                    entity_type=EntityType.CREATURE,
                    canonical_name=card.creature_name,
                    description=f"{card.rank.value.title()} {card.creature_class.value.title()} ({card.taxonomy.value})",
                    first_appearance_chapter=card.valid_from_chapter,
                )
                self.db.add(entity)
                self.db.flush()

            # Create AssetVersion
            asset_ver = AssetVersion(
                entity_id=entity.id,
                level=AssetLevel.ARC_DEFAULT,
                valid_from_chapter=card.valid_from_chapter,
                change_description=f"Bestiary Card: {card.rank.value.upper()} {card.creature_class.value.upper()}",
                visual_attributes={"creature_card": card.model_dump()},
            )
            self.db.add(asset_ver)

            # Persist Visual Reference if rendered
            if card.sheet_1_turnaround.image_path:
                vref = VisualRef(
                    entity_id=entity.id,
                    asset_version_id=asset_ver.id,
                    ref_type=RefType.CREATURE_REF,
                    image_path=card.sheet_1_turnaround.image_path,
                    approved=True,
                )
                self.db.add(vref)

            self.db.commit()
            logger.info("Successfully persisted CreatureCard for %s to SQLite.", card.creature_name)
            return True

        except Exception as exc:
            logger.error("Failed to persist creature card to SQLite: %s", exc)
            if self.db:
                self.db.rollback()
            return False

    def persist_item_card(
        self,
        project_id: str,
        item_card: ItemPropCard,
        chapter_number: int = 1,
    ) -> bool:
        """Persist ItemPropCard to SQLite Entity and AssetVersion tables."""
        if not self.db:
            return False

        try:
            entity = (
                self.db.query(Entity)
                .filter(
                    Entity.project_id == project_id,
                    Entity.canonical_name == item_card.item_name,
                )
                .first()
            )

            if not entity:
                entity = Entity(
                    id=item_card.item_id,
                    project_id=project_id,
                    entity_type=EntityType.ITEM,
                    canonical_name=item_card.item_name,
                    description=f"Memory/Item: {item_card.rank_tier} ({item_card.material_physics})",
                    first_appearance_chapter=chapter_number,
                )
                self.db.add(entity)
                self.db.flush()

            asset_ver = AssetVersion(
                entity_id=entity.id,
                level=AssetLevel.ARC_DEFAULT,
                valid_from_chapter=chapter_number,
                change_description=f"Item Prop Card: {item_card.item_name} [{item_card.rank_tier}]",
                visual_attributes={"item_card": item_card.model_dump()},
            )
            self.db.add(asset_ver)
            self.db.commit()
            logger.info("Successfully persisted ItemPropCard for %s to SQLite.", item_card.item_name)
            return True

        except Exception as exc:
            logger.error("Failed to persist item card to SQLite: %s", exc)
            if self.db:
                self.db.rollback()
            return False

    # =========================================================================
    # 7. RUNTIME COMBAT BATTLE ANCHOR RESOLUTION
    # =========================================================================

    def resolve_shot_creature_anchor(
        self,
        card: CreatureCard,
        combat_action: str = "lunging attack with scythe limbs and open mandibles",
        spatial_anchor: str = "right_flank",
    ) -> ShotCreatureAnchor:
        """
        Compile prompt tokens, IP-Adapter head chip path, and spatial framing for combat shots.
        """
        pos_tokens = []
        pos_tokens.extend(card.core_danbooru_biology_tags)
        pos_tokens.extend(card.supernatural_aura_tags)
        pos_tokens.extend(card.scale_spec.danbooru_scale_tags)
        pos_tokens.append(combat_action)

        seen = set()
        deduped = []
        for t in pos_tokens:
            tc = t.strip().lower()
            if tc and tc not in seen:
                seen.add(tc)
                deduped.append(t.strip())

        resolved_pos = ", ".join(deduped)
        resolved_neg = ", ".join(card.negative_identity_tags)

        return ShotCreatureAnchor(
            creature_card_id=card.creature_id,
            resolved_creature_prompt=resolved_pos,
            resolved_negative_prompt=resolved_neg,
            head_chip_path=card.head_reference_chip_path,
            suggested_spatial_anchor=spatial_anchor,
            human_scale_ratio=card.scale_spec.human_scale_ratio,
        )

