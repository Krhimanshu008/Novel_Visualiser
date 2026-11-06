"""Novel Visualiser — Character Continuity Subagent.

Autonomous Subagent responsible for:
1. Identifying characters, wardrobe, and hero items in novel chapters.
2. Querying historical SQLite canon & arc memory across past chapters and arcs.
3. Conducting 5-dimension Deep Observational Reasoning on biology, poverty wear, and expression.
4. Decoupling permanent facial/body geometry (Identity Lock) from temporal outfits and items.
5. Synthesizing and rendering the standardized 4-Sheet Character Reference Matrix (Turnaround, Expression Grid, Outfit, Items).
6. Auto-cropping canonical face references for ComfyUI IP-Adapter Plus and FaceDetailer.
7. Resolving locked prompt blocks and inpainting anchors to place characters into Location Stage canvases.
"""

from __future__ import annotations

import json
import logging
import os
import random
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
    EntityAlias,
    EntityType,
    RefType,
    VisualRef,
)
from app.schemas.character_subagent import (
    CharacterDeepObservation,
    CharacterMatrixSpec,
    CharacterSheetRole,
    DanbooruIdentityLock,
    ItemCard,
    OutfitCard,
    ShotCharacterAnchor,
    SingleCharacterSheetSpec,
)
from app.schemas.location_subagent import ArcHierarchyTag
from app.services.comfyui_client import ComfyUIWorkflowCompiler, DualGPUComfyUIDispatcher

logger = logging.getLogger("novel_visualiser.character_subagent")


class CharacterContinuitySubagent:
    """Autonomous Subagent dedicated to character canon, biology locks, and wardrobe modeling."""

    def __init__(
        self,
        db: Optional[Session] = None,
        comfy_dispatcher: Optional[DualGPUComfyUIDispatcher] = None,
        use_ai_thinking: bool = True,
    ):
        """Initialize CharacterContinuitySubagent with database, ComfyUI, and reasoning engine."""
        self.db = db
        self.comfy_dispatcher = comfy_dispatcher
        self.use_ai_thinking = use_ai_thinking
        self._kaggle_client = None

        if self.use_ai_thinking:
            try:
                # Add workspace root to sys.path if not present
                workspace_root = Path(__file__).resolve().parent.parent.parent.parent
                if str(workspace_root) not in sys.path:
                    sys.path.insert(0, str(workspace_root))
                from kaggle_ai_engine.proxy_client import KaggleModelProxyClient
                self._kaggle_client = KaggleModelProxyClient()
                logger.info("CharacterContinuitySubagent initialized with Kaggle Model Proxy reasoning engine.")
            except Exception as exc:
                logger.warning("Could not initialize Kaggle Model Proxy: %s. Using heuristic analysis.", exc)
                self._kaggle_client = None

    # =========================================================================
    # 1. HISTORICAL MEMORY & CANON LOOKUP
    # =========================================================================

    def query_historical_memory(
        self,
        project_id: str,
        candidate_name: str,
        chapter_number: int = 1,
    ) -> Optional[CharacterMatrixSpec]:
        """Query SQLite memory to fetch the active biological epoch and wardrobe for the requested chapter."""
        if not self.db:
            return None

        clean_name = candidate_name.strip().lower()

        # Query Entity by canonical_name or alias
        entity = (
            self.db.query(Entity)
            .filter(
                Entity.project_id == project_id,
                Entity.entity_type == EntityType.CHARACTER,
            )
            .filter(
                (Entity.canonical_name.ilike(f"%{clean_name}%"))
                | (Entity.id.in_(
                    self.db.query(EntityAlias.entity_id).filter(
                        EntityAlias.alias.ilike(f"%{clean_name}%")
                    )
                ))
            )
            .first()
        )

        if not entity:
            return None

        # Fetch active AssetVersion for this chapter (valid_from_chapter <= chapter_number)
        active_version = (
            self.db.query(AssetVersion)
            .filter(
                AssetVersion.entity_id == entity.id,
                AssetVersion.level.in_([AssetLevel.ARC_DEFAULT, AssetLevel.PROJECT_CANONICAL]),
                AssetVersion.valid_from_chapter <= chapter_number,
            )
            .order_by(AssetVersion.valid_from_chapter.desc())
            .first()
        )

        if not active_version or not active_version.visual_attributes:
            return None

        try:
            matrix_data = active_version.visual_attributes
            if "identity_lock" in matrix_data:
                spec = CharacterMatrixSpec.model_validate(matrix_data)
                spec.is_existing = True
                return spec
        except Exception as exc:
            logger.warning("Failed to parse cached character matrix spec for %s: %s", entity.canonical_name, exc)

        return None

    # =========================================================================
    # 2. DEEP OBSERVATIONAL REASONING ENGINE (AI THINKING)
    # =========================================================================

    def observe_and_synthesize_character(
        self,
        character_name: str,
        passage_text: str,
        arc_hierarchy: ArcHierarchyTag,
        gender: str = "male",
        deterministic_seed: int = 42,
    ) -> CharacterMatrixSpec:
        """
        Conduct deep 5-dimension observational reasoning and decouple biology from wardrobe.
        """
        import re
        clean_char = re.sub(r"[^a-zA-Z0-9_]+", "_", character_name.lower()).strip("_")
        character_id = f"char_{clean_char[:32]}"
        gender_tag = "1boy" if gender.lower() in ["male", "boy", "man"] else "1girl"

        # If frontier reasoning client is available, execute 5-dimension CoT
        if self._kaggle_client:
            try:
                system_prompt = (
                    "You are an Elite Anime Character Director and Character Designer for a high-end anime production.\n"
                    "Analyze the novel scene text and perform a deep 5-dimension observational analysis on the character:\n"
                    "1. Socioeconomic & Nutrition Physics: Malnutrition signs, skeletal build, sallow skin, pale undertones, visible collarbones.\n"
                    "2. Facial Morphology & Ocular Physics: Exact facial contours, messy raven-black hair bangs, sharp jaw, dark hollow onyx eyes.\n"
                    "3. Wardrobe Poverty & Textile Wear: Frayed seams, broken zippers, patched fabrics, grease/rain stains, synthetic weave.\n"
                    "4. Dynamic Emotional Expressions: Cynical smirk, grimace, exhausted drooping eyelids from survival vigil.\n"
                    "5. Posture & Kinesics: Hunched defensive shoulders, guarded posture, clutching items.\n\n"
                    "CRITICAL DECOUPLING RULES:\n"
                    "- Permanent Identity Lock: ONLY biological features (skin, hair, eyes, facial shape, slender build). NEVER clothing or weapons.\n"
                    "- Outfit Card: ONLY clothing, footwear, patches, and fabric condition for this arc.\n"
                    "- Item Card: Hero handheld props (e.g. synth-coffee cup).\n\n"
                    "Compile the 4-Sheet Reference Matrix (Danbooru tags, anime aesthetic):\n"
                    "- Sheet 1: Canonical Neutral Turnaround (front, 3/4, side, back on plain neutral grey, 1024x1024)\n"
                    "- Sheet 2: 4-Expression Emotion Grid (neutral, cynical grimace, heavy eyelids/drowsy, intense, 1024x1024)\n"
                    "- Sheet 3: Arc Outfit Plate (clothing on mannequin/neutral T-pose, 1024x1024)\n"
                    "- Sheet 4: Hero Props Plate (isolated items on white background, 1024x1024)"
                )

                user_prompt = f"""
Character Name: {character_name}
Gender: {gender}
Arc Hierarchy: Main Arc={arc_hierarchy.main_arc_id}, Sub Arc={arc_hierarchy.sub_arc_id}, Mini Arc={arc_hierarchy.mini_arc_id}
Novel Passage Text:
\"\"\"{passage_text[:3000]}\"\"\"

Respond ONLY in valid JSON matching this exact structure:
{{
  "observation": {{
    "socioeconomic_nutrition": "...",
    "facial_morphology": "...",
    "wardrobe_poverty_textile": "...",
    "emotional_expressions": "...",
    "posture_body_language": "..."
  }},
  "core_danbooru_biology_tags": ["pale skin", "messy jet-black hair", "sharp dark onyx eyes", "slender build", "dark circles under eyes"],
  "negative_identity_tags": ["blonde hair", "blue eyes", "muscular build", "beard"],
  "outfit_name": "Outskirts Frayed Utility Jacket & Rags",
  "danbooru_clothing_tags": ["oversized dark grey utility jacket", "frayed cuffs", "patched dark trousers", "worn combat boots"],
  "poverty_wear_tags": ["frayed seams", "tarnished zipper", "dirt stains on sleeves"],
  "negative_clothing_tags": ["clean clothes", "armor", "business suit", "hoodie"],
  "item_name": "Crushed Paper Synth-Coffee Cup",
  "danbooru_item_tags": ["holding paper coffee cup", "dented paper cup", "plastic lid"],
  "sheet_1_turnaround_prompt": "masterpiece, anime aesthetic, character model sheet, turnaround, neutral grey background, ...",
  "sheet_2_expressions_prompt": "masterpiece, anime aesthetic, expression sheet, 4-koma facial matrix, ...",
  "sheet_3_outfit_prompt": "masterpiece, anime aesthetic, costume design sheet, isolated wardrobe, ...",
  "sheet_4_items_prompt": "masterpiece, anime aesthetic, item sheet, isolated prop on white background, ..."
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

                # Clean trailing commas for resilient JSON parsing
                clean_text = re.sub(r",\s*([\]\}])", r"\1", clean_text)

                try:
                    data = json.loads(clean_text)
                except Exception as parse_err:
                    match = re.search(r"(\{.*\})", resp_text, re.DOTALL)
                    if match:
                        repaired = re.sub(r",\s*([\]\}])", r"\1", match.group(1))
                        data = json.loads(repaired)
                    else:
                        raise parse_err

                obs_data = data.get("observation", {})
                obs = CharacterDeepObservation(
                    socioeconomic_nutrition=obs_data.get("socioeconomic_nutrition", "Malnourished lean youth"),
                    facial_morphology=obs_data.get("facial_morphology", "Messy black hair, dark onyx eyes, sharp jaw"),
                    wardrobe_poverty_textile=obs_data.get("wardrobe_poverty_textile", "Frayed dark jacket, patched trousers"),
                    emotional_expressions=obs_data.get("emotional_expressions", "Cynical grimace, heavy drowsy eyelids"),
                    posture_body_language=obs_data.get("posture_body_language", "Defensive guarded posture"),
                )

                identity_lock = DanbooruIdentityLock(
                    entity_id=character_id,
                    entity_name=character_name,
                    gender_tag=gender_tag,
                    core_danbooru_tags=data.get("core_danbooru_biology_tags", [
                        "pale skin", "messy jet-black hair", "sharp dark onyx eyes", "slender build"
                    ]),
                    negative_identity_tags=data.get("negative_identity_tags", [
                        "blonde hair", "blue eyes", "muscular", "beard"
                    ]),
                    canonical_seed=deterministic_seed,
                )

                outfit = OutfitCard(
                    outfit_id=f"outfit_{character_id}_arc01",
                    outfit_name=data.get("outfit_name", "Outskirts Frayed Utility Jacket & Rags"),
                    danbooru_clothing_tags=data.get("danbooru_clothing_tags", [
                        "oversized dark grey utility jacket", "frayed collar", "patched dark trousers"
                    ]),
                    poverty_wear_tags=data.get("poverty_wear_tags", [
                        "frayed seams", "dirt stains", "tarnished zipper"
                    ]),
                    negative_clothing_tags=data.get("negative_clothing_tags", [
                        "clean clothes", "business suit", "armor", "hoodie"
                    ]),
                )

                held_item = ItemCard(
                    item_id=f"item_{character_id}_coffee_cup",
                    item_name=data.get("item_name", "Crushed Paper Synth-Coffee Cup"),
                    danbooru_item_tags=data.get("danbooru_item_tags", [
                        "holding paper coffee cup", "cheap white cup with plastic lid"
                    ]),
                )

                neg_char = "worst quality, low quality, blurry, extra limbs, bad anatomy, deformed fingers, text, watermark"

                return CharacterMatrixSpec(
                    character_id=character_id,
                    canonical_name=character_name,
                    gender=gender,
                    arc_hierarchy=arc_hierarchy,
                    observation=obs,
                    identity_lock=identity_lock,
                    active_outfit=outfit,
                    held_items=[held_item],
                    sheet_1_turnaround=SingleCharacterSheetSpec(
                        role=CharacterSheetRole.CANONICAL_TURNAROUND,
                        title=f"{character_name} — Canonical Turnaround Sheet",
                        prompt=data.get("sheet_1_turnaround_prompt", f"character model sheet, turnaround of {character_name}"),
                        negative_prompt=neg_char,
                        seed=deterministic_seed + 1,
                    ),
                    sheet_2_expressions=SingleCharacterSheetSpec(
                        role=CharacterSheetRole.EXPRESSION_GRID,
                        title=f"{character_name} — 4-Expression Emotion Matrix",
                        prompt=data.get("sheet_2_expressions_prompt", f"expression sheet, 4 expressions of {character_name}"),
                        negative_prompt=neg_char,
                        seed=deterministic_seed + 2,
                    ),
                    sheet_3_outfit=SingleCharacterSheetSpec(
                        role=CharacterSheetRole.OUTFIT_PLATE,
                        title=f"{character_name} — Arc 1 Wardrobe Plate",
                        prompt=data.get("sheet_3_outfit_prompt", f"costume design sheet for {outfit.outfit_name}"),
                        negative_prompt=neg_char,
                        seed=deterministic_seed + 3,
                    ),
                    sheet_4_items=SingleCharacterSheetSpec(
                        role=CharacterSheetRole.ITEM_SHEET,
                        title=f"{character_name} — Hero Props Sheet",
                        prompt=data.get("sheet_4_items_prompt", f"item sheet, isolated {held_item.item_name} on white background"),
                        negative_prompt=neg_char,
                        seed=deterministic_seed + 4,
                    ),
                    is_existing=False,
                )
            except Exception as exc:
                logger.warning("Kaggle AI character observation failed: %s; falling back to heuristic engine", exc)

        # Heuristic Rule-Based Deep Fallback
        obs = CharacterDeepObservation(
            socioeconomic_nutrition="Severe malnutrition, prominent clavicles, sallow pale skin from synth-paste diet",
            facial_morphology="Messy unkempt raven-black bangs over forehead, sharp jawline, hollow dark onyx eyes",
            wardrobe_poverty_textile="Oversized faded charcoal utility jacket, frayed cuff seams, tarnished plastic zipper",
            emotional_expressions="Cynical sneer, weary grimace drinking bitter coffee, heavy drooping eyelids",
            posture_body_language="Hunched defensive shoulders, clutching paper cup with both palms, guarded stance",
        )

        identity_lock = DanbooruIdentityLock(
            entity_id=character_id,
            entity_name=character_name,
            gender_tag=gender_tag,
            core_danbooru_tags=["pale skin", "messy jet-black hair", "sharp dark onyx eyes", "slender build", "dark circles under eyes"],
            negative_identity_tags=["blonde hair", "blue eyes", "muscular", "beard"],
            canonical_seed=deterministic_seed,
        )

        outfit = OutfitCard(
            outfit_id=f"outfit_{character_id}_arc01",
            outfit_name="Outskirts Frayed Utility Jacket & Rags",
            danbooru_clothing_tags=["oversized dark grey utility jacket", "frayed cuffs", "patched dark trousers", "worn combat boots"],
            poverty_wear_tags=["frayed seams", "dirt stains on sleeves", "tarnished zipper"],
            negative_clothing_tags=["clean clothes", "business suit", "armor", "hoodie"],
        )

        held_item = ItemCard(
            item_id=f"item_{character_id}_coffee_cup",
            item_name="Crushed Paper Synth-Coffee Cup",
            danbooru_item_tags=["holding paper coffee cup", "cheap white cup with plastic lid"],
        )

        neg_char = "worst quality, low quality, blurry, extra limbs, bad anatomy, deformed fingers, text, watermark"

        return CharacterMatrixSpec(
            character_id=character_id,
            canonical_name=character_name,
            gender=gender,
            arc_hierarchy=arc_hierarchy,
            observation=obs,
            identity_lock=identity_lock,
            active_outfit=outfit,
            held_items=[held_item],
            sheet_1_turnaround=SingleCharacterSheetSpec(
                role=CharacterSheetRole.CANONICAL_TURNAROUND,
                title=f"{character_name} — Canonical Turnaround Sheet",
                prompt=f"masterpiece, anime aesthetic, character model sheet, turnaround, neutral grey background, full body, front view, side view, back view, 1boy, {', '.join(identity_lock.core_danbooru_tags)}, simple studio background",
                negative_prompt=neg_char,
                seed=deterministic_seed + 1,
            ),
            sheet_2_expressions=SingleCharacterSheetSpec(
                role=CharacterSheetRole.EXPRESSION_GRID,
                title=f"{character_name} — 4-Expression Emotion Matrix",
                prompt=f"masterpiece, anime aesthetic, expression sheet, 4-koma grid, 1boy, {', '.join(identity_lock.core_danbooru_tags)}, neutral expression, cynical smirk, bitter grimace, heavy eyelids drowsy expression, close-up portraits, grey background",
                negative_prompt=neg_char,
                seed=deterministic_seed + 2,
            ),
            sheet_3_outfit=SingleCharacterSheetSpec(
                role=CharacterSheetRole.OUTFIT_PLATE,
                title=f"{character_name} — Arc 1 Wardrobe Plate",
                prompt=f"masterpiece, anime aesthetic, costume design sheet, isolated wardrobe, {', '.join(outfit.danbooru_clothing_tags)}, {', '.join(outfit.poverty_wear_tags)}, mannequin t-pose, white background, no humans",
                negative_prompt=neg_char,
                seed=deterministic_seed + 3,
            ),
            sheet_4_items=SingleCharacterSheetSpec(
                role=CharacterSheetRole.ITEM_SHEET,
                title=f"{character_name} — Hero Props Sheet",
                prompt=f"masterpiece, anime aesthetic, item design sheet, isolated prop, {', '.join(held_item.danbooru_item_tags)}, white background, high detail, no humans",
                negative_prompt=neg_char,
                seed=deterministic_seed + 4,
            ),
            is_existing=False,
        )

    # =========================================================================
    # 3. 4-SHEET CHARACTER REFERENCE FOUNDRY EXECUTION
    # =========================================================================

    def generate_character_matrix(
        self,
        matrix_spec: CharacterMatrixSpec,
        output_dir: Path,
        checkpoint: Optional[str] = None,
    ) -> CharacterMatrixSpec:
        """Render all 4 sheets and auto-extract canonical face chips for IP-Adapter."""
        output_dir.mkdir(parents=True, exist_ok=True)
        sheets = [
            ("sheet_1_turnaround", matrix_spec.sheet_1_turnaround, "turnaround"),
            ("sheet_2_expressions", matrix_spec.sheet_2_expressions, "expressions"),
            ("sheet_3_outfit", matrix_spec.sheet_3_outfit, "outfit"),
            ("sheet_4_items", matrix_spec.sheet_4_items, "items"),
        ]

        if self.comfy_dispatcher and self.comfy_dispatcher.clients:
            logger.info("Executing 4-sheet character generation on ComfyUI Dual-GPU cluster...")
            jobs = []
            for key, sheet, mode in sheets:
                job_id = f"{matrix_spec.character_id}_{sheet.role.value.lower()}"
                if mode in ["turnaround", "expressions"]:
                    workflow = ComfyUIWorkflowCompiler.compile_foundry_char(
                        char_id=job_id,
                        identity_tags=sheet.prompt,
                        mode=mode,
                        seed=sheet.seed or random.randint(1000, 9999),
                        width=sheet.width,
                        height=sheet.height,
                        checkpoint=checkpoint,
                    )
                elif mode == "items":
                    workflow = ComfyUIWorkflowCompiler.compile_foundry_prop(
                        prop_id=job_id,
                        item_tags=sheet.prompt,
                        negative_prompt=sheet.negative_prompt,
                        seed=sheet.seed or random.randint(1000, 9999),
                        width=sheet.width,
                        height=sheet.height,
                        checkpoint=checkpoint,
                    )
                else:  # outfit
                    workflow = ComfyUIWorkflowCompiler.compile_foundry_char(
                        char_id=job_id,
                        identity_tags=sheet.prompt,
                        mode="outfit",
                        seed=sheet.seed or random.randint(1000, 9999),
                        width=sheet.width,
                        height=sheet.height,
                        checkpoint=checkpoint,
                    )
                jobs.append((job_id, workflow))

            results = self.comfy_dispatcher.dispatch_batch_parallel(jobs, output_dir=output_dir)

            for (key, sheet, _), (job_id, img_path) in zip(sheets, results):
                if img_path and img_path.exists():
                    sheet.image_path = str(img_path)
                else:
                    self._create_mock_sheet(sheet, output_dir, matrix_spec.character_id)
        else:
            logger.info("ComfyUI cluster not connected; generating local test character sheets")
            for key, sheet, _ in sheets:
                self._create_mock_sheet(sheet, output_dir, matrix_spec.character_id)

        # Extract Canonical Face Reference for IP-Adapter Plus from Sheet 1 or Sheet 2
        face_chip_path = output_dir / f"{matrix_spec.character_id}_canonical_face_chip.png"
        ref_source = matrix_spec.sheet_2_expressions.image_path or matrix_spec.sheet_1_turnaround.image_path
        if not face_chip_path.exists() and ref_source:
            img = cv2.imread(ref_source)
            if img is not None:
                h, w, _ = img.shape
                face_crop = img[0:int(h * 0.5), 0:int(w * 0.5)]
                cv2.imwrite(str(face_chip_path), face_crop)
                matrix_spec.identity_lock.reference_face_path = str(face_chip_path)

        return matrix_spec

    def _create_mock_sheet(self, sheet: SingleCharacterSheetSpec, out_dir: Path, character_id: str) -> None:
        """Create a placeholder character sheet image for tests."""
        img_path = out_dir / f"{character_id}_{sheet.role.value.lower()}.png"
        img = np.zeros((sheet.height, sheet.width, 3), dtype=np.uint8)
        img[:] = (45, 45, 45)  # neutral grey background
        cv2.rectangle(img, (50, 50), (sheet.width - 50, sheet.height - 50), (100, 100, 100), 2)
        cv2.putText(
            img,
            f"{sheet.title}",
            (60, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
        )
        cv2.imwrite(str(img_path), img)
        sheet.image_path = str(img_path)

    # =========================================================================
    # 4. SHOT CHARACTER RESOLUTION & INPAINTING ANCHOR
    # =========================================================================

    def resolve_shot_character_anchor(
        self,
        matrix_spec: CharacterMatrixSpec,
        shot_action: str,
        emotion: str = "neutral",
        camera_framing: str = "MEDIUM",  # 'CLOSE_UP', 'MEDIUM', 'FULL_BODY'
    ) -> ShotCharacterAnchor:
        """
        Assemble the ordered, locked prompt block and references to inpaint into a location canvas.
        """
        ident = matrix_spec.identity_lock
        outfit = matrix_spec.active_outfit

        # Ordered Danbooru Block Assembly: [Quality] [Gender] [Biology Lock] [Outfit] [Wear] [Action] [Emotion]
        prompt_parts = [
            "masterpiece, best quality, cinematic manhwa illustration",
            ident.gender_tag,
            ", ".join(ident.core_danbooru_tags),
            ", ".join(outfit.danbooru_clothing_tags),
            ", ".join(outfit.poverty_wear_tags),
            shot_action,
            emotion,
        ]

        # Add held item if action mentions holding/drinking/etc
        if matrix_spec.held_items and ("coffee" in shot_action.lower() or "cup" in shot_action.lower()):
            prompt_parts.append(", ".join(matrix_spec.held_items[0].danbooru_item_tags))

        full_prompt = ", ".join([p for p in prompt_parts if p.strip()])

        # Build Negative Prompt
        neg_parts = [
            "worst quality, low quality, blurry, extra limbs, bad anatomy, deformed fingers",
            ", ".join(ident.negative_identity_tags),
            ", ".join(outfit.negative_clothing_tags),
        ]
        full_neg = ", ".join([p for p in neg_parts if p.strip()])

        # Framing-based inpaint box
        framing_upper = camera_framing.upper()
        if "CLOSE" in framing_upper:
            inpaint_box = [0.25, 0.15, 0.75, 0.85]
        elif "FULL" in framing_upper:
            inpaint_box = [0.20, 0.10, 0.80, 0.95]
        else:  # MEDIUM
            inpaint_box = [0.25, 0.25, 0.75, 0.90]

        return ShotCharacterAnchor(
            entity_id=matrix_spec.character_id,
            entity_name=matrix_spec.canonical_name,
            full_positive_prompt=full_prompt,
            full_negative_prompt=full_neg,
            face_reference_path=ident.reference_face_path,
            expression_tag=emotion,
            pose_action_tag=shot_action,
            recommended_inpaint_box=inpaint_box,
        )

    # =========================================================================
    # 5. DATABASE CANON PERSISTENCE
    # =========================================================================

    def save_to_canon(
        self,
        project_id: str,
        matrix_spec: CharacterMatrixSpec,
        current_chapter: int = 1,
    ) -> Entity:
        """Persist character into SQLite entities, asset_versions, and visual_refs."""
        if not self.db:
            logger.warning("No database session provided; skipping canon persistence.")
            return Entity(id=matrix_spec.character_id, canonical_name=matrix_spec.canonical_name)

        # 1. Upsert Entity
        entity = (
            self.db.query(Entity)
            .filter(Entity.id == matrix_spec.character_id)
            .first()
        )
        if not entity:
            entity = Entity(
                id=matrix_spec.character_id,
                project_id=project_id,
                entity_type=EntityType.CHARACTER,
                canonical_name=matrix_spec.canonical_name,
                description=matrix_spec.observation.socioeconomic_nutrition,
                first_appearance_chapter=current_chapter,
                is_active=True,
            )
            self.db.add(entity)
            self.db.flush()

        # 2. Add AssetVersion (Level = ARC_DEFAULT or PROJECT_CANONICAL)
        asset_level = AssetLevel.ARC_DEFAULT if matrix_spec.arc_hierarchy.main_arc_id else AssetLevel.PROJECT_CANONICAL
        asset_version = AssetVersion(
            entity_id=entity.id,
            level=asset_level,
            arc_id=matrix_spec.arc_hierarchy.main_arc_id if self.db.query(Entity).filter(Entity.id == matrix_spec.arc_hierarchy.main_arc_id).first() else None,
            valid_from_chapter=current_chapter,
            change_description=f"Epoch: {matrix_spec.active_epoch.epoch_name if matrix_spec.active_epoch else 'Initial'} | Outfit: {matrix_spec.active_outfit.outfit_name}",
            visual_attributes=matrix_spec.model_dump(),
        )
        self.db.add(asset_version)
        self.db.flush()

        # 3. Add VisualRef records for turnaround, expressions, and face chip
        sheets = [
            (matrix_spec.sheet_1_turnaround, RefType.TURNAROUND_SHEET),
            (matrix_spec.sheet_2_expressions, RefType.EXPRESSION_GRID),
            (matrix_spec.sheet_3_outfit, RefType.LOCATION_PLATE),  # reusable image slot
        ]

        for s, rtype in sheets:
            if s.image_path:
                vref = VisualRef(
                    entity_id=entity.id,
                    asset_version_id=asset_version.id,
                    ref_type=rtype,
                    image_path=s.image_path,
                    seed=s.seed,
                )
                self.db.add(vref)

        if matrix_spec.identity_lock.reference_face_path:
            vref_face = VisualRef(
                entity_id=entity.id,
                asset_version_id=asset_version.id,
                ref_type=RefType.EXPRESSION_GRID,
                image_path=matrix_spec.identity_lock.reference_face_path,
            )
            self.db.add(vref_face)

        self.db.commit()
        logger.info("Successfully persisted character %s to SQLite canon.", matrix_spec.canonical_name)
        return entity

