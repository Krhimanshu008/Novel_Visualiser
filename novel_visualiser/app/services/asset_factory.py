"""Novel Visualiser — Asset Pre-Creation Factory (VCE Asset Lifecycle).

Responsible for:
1. Minting complete reference packages BEFORE shot generation:
   - Character Turnaround & Expression packages
   - Arc Outfit packages with condition variants (pristine, battle_worn, critical_damage)
   - Location Master Plates with A/B/C perspective angles and depth/lineart guides
   - Isolated Item / Weapon sheets
2. Organizing directory hierarchies per project and per arc.
3. Registering assets into SQLite World Canon and VisualRef tables.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional
from sqlalchemy.orm import Session

from app.database.models import (
    AssetLevel,
    AssetVersion,
    Entity,
    EntityType,
    VisualRef,
)
from app.schemas.visual_context import (
    DanbooruIdentityLock,
    OutfitCard,
    LocationPlateSpec,
)

logger = logging.getLogger("novel_visualiser.asset_factory")


class AssetPreCreationFactory:
    """Pre-creation factory for visual reference packages."""

    def __init__(self, base_data_dir: Path | str = "data"):
        self.base_data_dir = Path(base_data_dir)

    def get_character_canon_dir(self, project_id: str, entity_id: str) -> Path:
        """Return path for permanent character biological reference packages."""
        p = self.base_data_dir / "projects" / project_id / "canon_library" / "characters" / entity_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_outfit_dir(self, project_id: str, arc_id: str, outfit_id: str) -> Path:
        """Return path for arc-specific outfit packages."""
        p = self.base_data_dir / "projects" / project_id / "arcs" / arc_id / "outfits" / outfit_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_location_plate_dir(self, project_id: str, arc_id: str, location_id: str) -> Path:
        """Return path for location master plates."""
        p = self.base_data_dir / "projects" / project_id / "arcs" / arc_id / "location_plates" / location_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get_item_dir(self, project_id: str, item_id: str) -> Path:
        """Return path for isolated prop/item packages."""
        p = self.base_data_dir / "projects" / project_id / "canon_library" / "master_props" / item_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def build_character_package_manifest(
        self,
        project_id: str,
        entity_id: str,
        identity_lock: DanbooruIdentityLock,
    ) -> dict[str, Any]:
        """Generate the specification manifest for a character turnaround package."""
        target_dir = self.get_character_canon_dir(project_id, entity_id)
        core_tags_str = ", ".join(identity_lock.core_danbooru_tags)
        neg_tags_str = ", ".join(identity_lock.negative_identity_tags)

        manifest = {
            "entity_id": entity_id,
            "entity_name": identity_lock.entity_name,
            "canonical_seed": identity_lock.canonical_seed,
            "views": {
                "face_front": {
                    "filename": "face_front.png",
                    "path": str(target_dir / "face_front.png"),
                    "prompt": f"masterpiece, best quality, portrait, face close-up, front view, looking at viewer, neutral expression, {core_tags_str}, simple neutral grey background",
                    "negative_prompt": f"profile view, 3/4 view, {neg_tags_str}",
                },
                "face_three_quarter": {
                    "filename": "face_three_quarter.png",
                    "path": str(target_dir / "face_three_quarter.png"),
                    "prompt": f"masterpiece, best quality, portrait, 3/4 angle view, looking towards side, {core_tags_str}, simple neutral grey background",
                    "negative_prompt": f"front view, extreme profile, {neg_tags_str}",
                },
                "face_profile": {
                    "filename": "face_profile.png",
                    "path": str(target_dir / "face_profile.png"),
                    "prompt": f"masterpiece, best quality, portrait, profile view, side angle, sharp jawline, {core_tags_str}, simple neutral grey background",
                    "negative_prompt": f"front view, {neg_tags_str}",
                },
                "expression_matrix": {
                    "filename": "expression_matrix.png",
                    "path": str(target_dir / "expression_matrix.png"),
                    "expressions": ["neutral", "fierce_combat", "wary_shocked", "cunning_smirk"],
                    "prompt": f"masterpiece, best quality, 4-panel expression sheet, neutral, angry shouting, wide eyes shocked, smirk, {core_tags_str}",
                },
                "full_body_turnaround": {
                    "filename": "full_body_turnaround.png",
                    "path": str(target_dir / "full_body_turnaround.png"),
                    "prompt": f"masterpiece, best quality, full body turnaround orthographic sheet, front view, side view, back view, A-pose, {core_tags_str}, neutral studio background",
                },
            },
        }

        # Write manifest file
        manifest_path = target_dir / "character_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def build_outfit_package_manifest(
        self,
        project_id: str,
        arc_id: str,
        outfit_card: OutfitCard,
    ) -> dict[str, Any]:
        """Generate the specification manifest for an arc outfit package."""
        target_dir = self.get_outfit_dir(project_id, arc_id, outfit_card.outfit_id)
        clothing_str = ", ".join(outfit_card.danbooru_clothing_tags)
        neg_str = ", ".join(outfit_card.negative_clothing_tags)

        manifest = {
            "outfit_id": outfit_card.outfit_id,
            "entity_id": outfit_card.entity_id,
            "outfit_name": outfit_card.outfit_name,
            "valid_from_chapter": outfit_card.valid_from_chapter,
            "valid_to_chapter": outfit_card.valid_to_chapter,
            "views": {
                "outfit_front": {
                    "filename": "outfit_front.png",
                    "path": str(target_dir / "outfit_front.png"),
                    "prompt": f"masterpiece, best quality, full body, standing front view, wearing {clothing_str}, neutral grey studio background",
                    "negative_prompt": neg_str,
                },
                "outfit_back": {
                    "filename": "outfit_back.png",
                    "path": str(target_dir / "outfit_back.png"),
                    "prompt": f"masterpiece, best quality, full body, from behind, back view showing mantle and cloak, wearing {clothing_str}, neutral background",
                    "negative_prompt": neg_str,
                },
            },
            "condition_variants": {
                "pristine": {
                    "delta_tags": ["clean clothes", "pristine condition"],
                },
                "battle_worn": {
                    "delta_tags": ["frayed seams", "dirt smudges", "minor tears in fabric", "battle-worn"],
                },
                "critical_damage": {
                    "delta_tags": ["heavily torn clothing", "deep fabric slashes", "blood stains", "charred edges"],
                },
            },
        }

        manifest_path = target_dir / "outfit_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

    def build_location_plates_manifest(
        self,
        project_id: str,
        arc_id: str,
        plate_spec: LocationPlateSpec,
    ) -> dict[str, Any]:
        """Generate the specification manifest for location master plates."""
        target_dir = self.get_location_plate_dir(project_id, arc_id, plate_spec.location_id)
        tags_str = ", ".join(plate_spec.plate_tags)
        neg_str = ", ".join(plate_spec.negative_environment_tags)

        manifest = {
            "location_id": plate_spec.location_id,
            "location_name": plate_spec.location_name,
            "views": {
                "master_plate_angle_A": {
                    "filename": "master_plate_angle_A.png",
                    "path": str(target_dir / "master_plate_angle_A.png"),
                    "prompt": f"masterpiece, best quality, wide shot, establishing landscape shot, no people, no characters, empty scenery, {tags_str}",
                    "negative_prompt": f"people, character, 1boy, 1girl, {neg_str}",
                },
                "master_plate_angle_B": {
                    "filename": "master_plate_angle_B.png",
                    "path": str(target_dir / "master_plate_angle_B.png"),
                    "prompt": f"masterpiece, best quality, medium wide shot, reverse angle view, interior architecture, no people, {tags_str}",
                    "negative_prompt": f"people, character, {neg_str}",
                },
                "controlnet_depth_A": {
                    "filename": "controlnet_depth_A.png",
                    "path": str(target_dir / "controlnet_depth_A.png"),
                    "purpose": "ControlNet Depth map guide for spatial stability across shots",
                },
            },
        }

        manifest_path = target_dir / "location_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest

