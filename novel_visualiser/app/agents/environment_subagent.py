"""Novel Visualiser — Environment & Lighting Continuity Subagent.

Autonomous Subagent responsible for:
1. Identifying chronological time, solar/lunar vectors, weather, and atmospheric phenomena in novel chapters.
2. Querying historical SQLite canon to preserve or evolve lighting/weather across scenes and chapters.
3. Conducting 5-dimension Deep Observational Reasoning on atmospheric physics, lighting rigs, foliage, and surface moisture.
4. Generating physical LightingRigSpecs (Key, Fill, Rim vectors, Kelvin ratings, shadow falloffs).
5. Compiling prompt-ready Danbooru atmospheric token blocks and negative constraints for diffusion/inpainting.
6. Generating color harmony palettes and runtime anchors for compositing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional
from sqlalchemy.orm import Session

from app.database.models import (
    AssetLevel,
    AssetVersion,
    Entity,
    EntityAlias,
    EntityType,
)
from app.schemas.environment_subagent import (
    EnvironmentalAtmosphereCard,
    EnvironmentalDeepObservation,
    LightingRigSpec,
    ShotEnvironmentAnchor,
    TimeOfDayEnum,
    WeatherConditionEnum,
)
from app.schemas.location_subagent import ArcHierarchyTag

logger = logging.getLogger("novel_visualiser.environment_subagent")


class EnvironmentContinuitySubagent:
    """Autonomous Subagent dedicated to environmental physics, lighting rigs, and atmospheric continuity."""

    def __init__(
        self,
        db: Optional[Session] = None,
        use_ai_thinking: bool = True,
    ):
        """Initialize EnvironmentContinuitySubagent with database and frontier reasoning engine."""
        self.db = db
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
                logger.info("EnvironmentContinuitySubagent initialized with Kaggle Model Proxy reasoning engine.")
            except Exception as exc:
                logger.warning("Could not initialize Kaggle Model Proxy: %s. Using heuristic analysis.", exc)
                self._kaggle_client = None

    # =========================================================================
    # 1. HISTORICAL CANON & ATMOSPHERIC MEMORY LOOKUP
    # =========================================================================

    def query_historical_environment(
        self,
        project_id: str,
        location_id: str,
        chapter_number: int = 1,
    ) -> Optional[EnvironmentalAtmosphereCard]:
        """
        Query SQLite memory to fetch the active environmental card for the specified location and chapter.
        """
        if not self.db:
            return None

        # Find location entity
        entity = (
            self.db.query(Entity)
            .filter(
                Entity.project_id == project_id,
                Entity.id == location_id,
            )
            .first()
        )
        if not entity:
            # Try by canonical_name
            entity = (
                self.db.query(Entity)
                .filter(
                    Entity.project_id == project_id,
                    Entity.canonical_name == location_id,
                )
                .first()
            )

        if not entity:
            return None

        # Fetch latest AssetVersion valid at this chapter
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
            if "environment_card" in attrs:
                return EnvironmentalAtmosphereCard.model_validate(attrs["environment_card"])
        except Exception as exc:
            logger.warning("Failed to parse cached environmental card: %s", exc)

        return None

    # =========================================================================
    # 2. 5-DIMENSION EXTENDED OBSERVATIONAL REASONING
    # =========================================================================

    def observe_and_synthesize_environment(
        self,
        location_name: str,
        passage_text: str,
        arc_hierarchy: ArcHierarchyTag,
        location_id: Optional[str] = None,
        chapter_number: int = 1,
    ) -> EnvironmentalAtmosphereCard:
        """
        Conduct deep 5-dimension observational reasoning on lighting, weather, foliage, and surface physics.
        """
        import re
        clean_loc = re.sub(r"[^a-zA-Z0-9_]+", "_", location_name.lower()).strip("_")
        loc_id = location_id or f"loc_{clean_loc[:32]}"
        env_id = f"env_{loc_id.replace('loc_', '')}_ch{chapter_number:03d}"

        # If frontier reasoning client is available, execute 5-dimension CoT
        if self._kaggle_client:
            try:
                system_prompt = (
                    "You are an Elite Anime Visual Director, Cinematographer, and Lighting Artist for a high-end production.\n"
                    "Analyze the novel scene text and conduct a deep 5-dimension observational analysis on the environment:\n"
                    "1. Chronological Time & Solar Geometry: Celestial light, solar/lunar altitude, azimuth, sky color gradient, narrative time cues.\n"
                    "2. Lighting Physics: Key light source (e.g. flickering sodium vapor lamp, hearth, sun), direction vector, Kelvin color temperature (e.g. 2700K amber, 8000K twilight), fill bounce, rim/kicker backlighting, shadow hardness/diffusion.\n"
                    "3. Atmospheric Particles & Weather: Smog density, suspended mist droplets, airborne ash, crepuscular rays (God rays), wind velocity.\n"
                    "4. Foliage & Micro-Ecosystems: Specific plant species, dead weeds in fissures, withered branches, moss, wind deflection.\n"
                    "5. Surface Weathering & Moisture: Wet pavement, puddles mirroring streetlights, frost rime, mud, specular reflection.\n\n"
                    "Then compile prompt conditioning token blocks for Stable Diffusion / SDXL anime manhwa aesthetic:\n"
                    "- danbooru_lighting_tags: Lighting tokens (e.g. 'volumetric lighting', 'rim light', 'backlighting', 'sodium vapor streetlamp', 'warm amber glow')\n"
                    "- danbooru_weather_tags: Weather tokens (e.g. 'overcast', 'smog', 'crepuscular rays', 'foggy ambient')\n"
                    "- danbooru_foliage_tags: Vegetation tokens (e.g. 'dead weeds in sidewalk cracks', 'withered branch')\n"
                    "- danbooru_surface_tags: Ground moisture tokens (e.g. 'wet asphalt', 'puddle reflections')\n"
                    "- negative_environment_tags: Anti-features that contradict this mood (e.g. 'sunny blue sky', 'bright daylight', 'clean dry ground')\n"
                    "- color_palette_hex: 4 hex codes (dominant darks, key light accent, ambient fill, atmospheric haze)"
                )

                user_prompt = f"""
Location Name: {location_name}
Arc Hierarchy: Main Arc={arc_hierarchy.main_arc_id}, Sub Arc={arc_hierarchy.sub_arc_id}, Mini Arc={arc_hierarchy.mini_arc_id}
Narrative Time/Weather Hint: {arc_hierarchy.time_of_day}, {arc_hierarchy.weather}
Novel Passage Text:
\"\"\"{passage_text[:3000]}\"\"\"

Allowed 'time_of_day' values: ['dawn', 'morning', 'noon_harsh', 'afternoon', 'golden_hour', 'twilight_blue_hour', 'night', 'midnight_pitch', 'timeless_void']
Allowed 'weather_condition' values: ['clear', 'overcast_smog', 'acid_drizzle', 'heavy_rain', 'dense_fog', 'ash_fall', 'blizzard', 'supernatural_darkness']

Respond ONLY in valid JSON matching this exact structure:
{{
  "observation": {{
    "chronological_solar": "...",
    "lighting_physics": "...",
    "atmospheric_particles": "...",
    "foliage_ecosystem": "...",
    "surface_weathering": "..."
  }},
  "time_of_day": "twilight_blue_hour",
  "weather_condition": "overcast_smog",
  "lighting_rig": {{
    "key_light_source": "...",
    "key_light_direction": "...",
    "key_color_temperature_k": 2700,
    "fill_light_source": "...",
    "rim_light_active": true,
    "rim_light_color": "...",
    "shadow_hardness": "soft_diffuse"
  }},
  "danbooru_lighting_tags": ["volumetric lighting", "rim light", "backlighting", "sodium vapor streetlamp", "warm amber glow"],
  "danbooru_weather_tags": ["overcast", "smog", "crepuscular rays", "foggy ambient"],
  "danbooru_foliage_tags": ["dead weeds in sidewalk cracks", "withered tree branch"],
  "danbooru_surface_tags": ["wet asphalt", "puddle reflections", "damp ground"],
  "negative_environment_tags": ["sunny blue sky", "bright daylight", "clear sky", "lush green grass", "clean dry ground"],
  "color_palette_hex": ["#14161d", "#d47828", "#3b4f63", "#7a8a99"]
}}
"""

                raw_resp = self._kaggle_client.chat_completion(
                    messages=[
                        {"role": "user", "content": f"System: {system_prompt}\nUser: {user_prompt}"}
                    ],
                    temperature=0.2,
                )

                # Clean markdown backticks and trailing commas
                clean_text = raw_resp.strip()
                if clean_text.startswith("```"):
                    clean_text = re.sub(r"^```(?:json)?\s*", "", clean_text)
                    clean_text = re.sub(r"\s*```$", "", clean_text)
                clean_text = re.sub(r",\s*([\]\}])", r"\1", clean_text)

                parsed = json.loads(clean_text)

                # Robust enum mapping with substring fallback
                tod_val = str(parsed.get("time_of_day", "twilight_blue_hour")).lower().strip()
                matched_tod = None
                for candidate in TimeOfDayEnum:
                    if candidate.value == tod_val or candidate.name.lower() == tod_val:
                        matched_tod = candidate
                        break
                if not matched_tod:
                    if "midnight" in tod_val or "pitch" in tod_val:
                        matched_tod = TimeOfDayEnum.MIDNIGHT_PITCH
                    elif "twilight" in tod_val or "blue" in tod_val or "dusk" in tod_val:
                        matched_tod = TimeOfDayEnum.TWILIGHT_BLUE_HOUR
                    elif "dawn" in tod_val or "sunrise" in tod_val:
                        matched_tod = TimeOfDayEnum.DAWN
                    elif "golden" in tod_val or "sunset" in tod_val:
                        matched_tod = TimeOfDayEnum.GOLDEN_HOUR
                    elif "noon" in tod_val or "midday" in tod_val:
                        matched_tod = TimeOfDayEnum.NOON_HARSH
                    elif "night" in tod_val:
                        matched_tod = TimeOfDayEnum.NIGHT
                    elif "morning" in tod_val:
                        matched_tod = TimeOfDayEnum.MORNING
                    else:
                        matched_tod = TimeOfDayEnum.TWILIGHT_BLUE_HOUR

                weather_val = str(parsed.get("weather_condition", "overcast_smog")).lower().strip()
                matched_weather = None
                for candidate in WeatherConditionEnum:
                    if candidate.value == weather_val or candidate.name.lower() == weather_val:
                        matched_weather = candidate
                        break
                if not matched_weather:
                    if "blizzard" in weather_val or "snow" in weather_val:
                        matched_weather = WeatherConditionEnum.BLIZZARD
                    elif "acid" in weather_val and "drizzle" in weather_val:
                        matched_weather = WeatherConditionEnum.ACID_DRIZZLE
                    elif "rain" in weather_val or "storm" in weather_val or "downpour" in weather_val:
                        matched_weather = WeatherConditionEnum.HEAVY_RAIN
                    elif "acid" in weather_val:
                        matched_weather = WeatherConditionEnum.ACID_DRIZZLE
                    elif "fog" in weather_val or "mist" in weather_val:
                        matched_weather = WeatherConditionEnum.DENSE_FOG
                    elif "smog" in weather_val or "overcast" in weather_val or "haze" in weather_val:
                        matched_weather = WeatherConditionEnum.OVERCAST_SMOG
                    elif "ash" in weather_val:
                        matched_weather = WeatherConditionEnum.ASH_FALL
                    elif "clear" in weather_val:
                        matched_weather = WeatherConditionEnum.CLEAR
                    else:
                        matched_weather = WeatherConditionEnum.OVERCAST_SMOG

                rig_data = parsed.get("lighting_rig", {})
                lighting_rig = LightingRigSpec(
                    key_light_source=rig_data.get("key_light_source", "Overhead flickering sodium-vapor streetlamp"),
                    key_light_direction=rig_data.get("key_light_direction", "top-left 45-degree elevation"),
                    key_color_temperature_k=int(rig_data.get("key_color_temperature_k", 2700)),
                    fill_light_source=rig_data.get("fill_light_source", "Ambient sky bounce"),
                    rim_light_active=bool(rig_data.get("rim_light_active", True)),
                    rim_light_color=rig_data.get("rim_light_color", "Pale cold cyan"),
                    shadow_hardness=rig_data.get("shadow_hardness", "soft_diffuse"),
                )

                observation = EnvironmentalDeepObservation(
                    chronological_solar=parsed.get("observation", {}).get("chronological_solar", ""),
                    lighting_physics=parsed.get("observation", {}).get("lighting_physics", ""),
                    atmospheric_particles=parsed.get("observation", {}).get("atmospheric_particles", ""),
                    foliage_ecosystem=parsed.get("observation", {}).get("foliage_ecosystem", ""),
                    surface_weathering=parsed.get("observation", {}).get("surface_weathering", ""),
                )

                card = EnvironmentalAtmosphereCard(
                    environment_id=env_id,
                    location_id=loc_id,
                    environment_name=f"{location_name} Atmosphere (Ch {chapter_number})",
                    novel_arc=arc_hierarchy,
                    valid_from_chapter=chapter_number,
                    valid_to_chapter=None,
                    time_of_day=matched_tod,
                    weather_condition=matched_weather,
                    observation=observation,
                    lighting_rig=lighting_rig,
                    danbooru_lighting_tags=parsed.get("danbooru_lighting_tags", []),
                    danbooru_weather_tags=parsed.get("danbooru_weather_tags", []),
                    danbooru_foliage_tags=parsed.get("danbooru_foliage_tags", []),
                    danbooru_surface_tags=parsed.get("danbooru_surface_tags", []),
                    negative_environment_tags=parsed.get("negative_environment_tags", []),
                    color_palette_hex=parsed.get("color_palette_hex", ["#14161d", "#d47828", "#3b4f63", "#7a8a99"]),
                )
                logger.info("Successfully synthesized environmental atmosphere for %s via AI Thinking.", location_name)
                return card

            except Exception as exc:
                logger.warning("AI Environmental analysis failed (%s); falling back to heuristic engine.", exc)

        # Fallback Heuristic Synthesis
        return self._heuristic_observation_and_card(
            location_name=location_name,
            passage_text=passage_text,
            arc_hierarchy=arc_hierarchy,
            loc_id=loc_id,
            env_id=env_id,
            chapter_number=chapter_number,
        )

    # =========================================================================
    # 3. HEURISTIC FALLBACK SYNTHESIS
    # =========================================================================

    def _heuristic_observation_and_card(
        self,
        location_name: str,
        passage_text: str,
        arc_hierarchy: ArcHierarchyTag,
        loc_id: str,
        env_id: str,
        chapter_number: int,
    ) -> EnvironmentalAtmosphereCard:
        """Rule-based heuristic parsing of environmental, lighting, and weather tokens."""
        text = passage_text.lower()

        # Deduce Time of Day
        if any(w in text for w in ["twilight", "dusk", "sunset", "evening"]):
            tod = TimeOfDayEnum.TWILIGHT_BLUE_HOUR
            key_temp = 3200
        elif any(w in text for w in ["midnight", "pitch black", "dead of night"]):
            tod = TimeOfDayEnum.MIDNIGHT_PITCH
            key_temp = 6500
        elif any(w in text for w in ["night", "darkness", "shadow"]):
            tod = TimeOfDayEnum.NIGHT
            key_temp = 2700
        elif any(w in text for w in ["dawn", "sunrise", "early morning"]):
            tod = TimeOfDayEnum.DAWN
            key_temp = 2800
        elif any(w in text for w in ["noon", "midday", "harsh sun"]):
            tod = TimeOfDayEnum.NOON_HARSH
            key_temp = 5500
        else:
            tod = TimeOfDayEnum.TWILIGHT_BLUE_HOUR
            key_temp = 3000

        # Deduce Weather
        if any(w in text for w in ["rain", "drizzle", "wet", "puddle"]):
            weather = WeatherConditionEnum.ACID_DRIZZLE if "acid" in text else WeatherConditionEnum.HEAVY_RAIN
        elif any(w in text for w in ["smog", "smoke", "pollut"]):
            weather = WeatherConditionEnum.OVERCAST_SMOG
        elif any(w in text for w in ["fog", "mist"]):
            weather = WeatherConditionEnum.DENSE_FOG
        elif any(w in text for w in ["snow", "frost", "blizzard"]):
            weather = WeatherConditionEnum.BLIZZARD
        else:
            weather = WeatherConditionEnum.OVERCAST_SMOG

        observation = EnvironmentalDeepObservation(
            chronological_solar="Subdued diffuse twilight sky obscured by industrial smog layers. Low solar angle creates elongated soft shadows.",
            lighting_physics=f"Primary illumination from artificial street fixtures ({key_temp}K). Ambient cold cyan fill from upper atmospheric haze.",
            atmospheric_particles="Suspended micro-particulates of industrial smog, light moisture condensation, and visible volumetric beam falloff.",
            foliage_ecosystem="Gnarled hardy weeds penetrating cracked asphalt seams; withered leafless branches overhead.",
            surface_weathering="Damp asphalt surfaces showing subtle specular sheen; localized pooling in pavement depressions.",
        )

        lighting_rig = LightingRigSpec(
            key_light_source="Overhead sodium vapor fixture" if key_temp < 4000 else "Diffuse overcast sky",
            key_light_direction="Top-left 45-degree angle",
            key_color_temperature_k=key_temp,
            fill_light_source="Ambient smog sky bounce",
            rim_light_active=True,
            rim_light_color="Cold pale cyan backlight",
            shadow_hardness="soft_diffuse",
        )

        lighting_tags = ["volumetric lighting", "rim light", "backlighting", "cinematic lighting"]
        if key_temp <= 3200:
            lighting_tags.extend(["warm amber glow", "sodium vapor streetlamp"])
        else:
            lighting_tags.extend(["cold ambient lighting", "moonlight"])

        weather_tags = ["overcast", "atmospheric haze"]
        if "smog" in text or weather == WeatherConditionEnum.OVERCAST_SMOG:
            weather_tags.extend(["smog", "gloomy sky"])
        if "rain" in text or "drizzle" in text:
            weather_tags.extend(["drizzle", "rain droplets"])

        foliage_tags = ["dead weeds in cracks", "withered tree"]
        surface_tags = ["damp ground", "wet pavement reflections"]

        return EnvironmentalAtmosphereCard(
            environment_id=env_id,
            location_id=loc_id,
            environment_name=f"{location_name} Heuristic Atmosphere",
            novel_arc=arc_hierarchy,
            valid_from_chapter=chapter_number,
            valid_to_chapter=None,
            time_of_day=tod,
            weather_condition=weather,
            observation=observation,
            lighting_rig=lighting_rig,
            danbooru_lighting_tags=lighting_tags,
            danbooru_weather_tags=weather_tags,
            danbooru_foliage_tags=foliage_tags,
            danbooru_surface_tags=surface_tags,
            negative_environment_tags=[
                "sunny blue sky", "bright daylight", "clear sky", "lush green grass",
                "clean dry ground", "lens glare"
            ],
            color_palette_hex=["#14161d", "#d47828", "#3b4f63", "#7a8a99"],
        )

    # =========================================================================
    # 4. DATABASE PERSISTENCE
    # =========================================================================

    def persist_environment_card(
        self,
        project_id: str,
        card: EnvironmentalAtmosphereCard,
    ) -> bool:
        """Persist the environmental card into SQLite AssetVersion records."""
        if not self.db:
            return False

        try:
            # Query Location Entity
            entity = (
                self.db.query(Entity)
                .filter(
                    Entity.project_id == project_id,
                    Entity.id == card.location_id,
                )
                .first()
            )

            # If not found by ID, look up by canonical_name
            if not entity:
                entity = (
                    self.db.query(Entity)
                    .filter(
                        Entity.project_id == project_id,
                        Entity.canonical_name.ilike(f"%{card.location_id.replace('loc_', '')}%"),
                    )
                    .first()
                )

            if not entity:
                logger.warning("Could not find location entity %s to link environment card.", card.location_id)
                return False

            # Create or update AssetVersion
            asset_ver = AssetVersion(
                entity_id=entity.id,
                level=AssetLevel.ARC_DEFAULT,
                valid_from_chapter=card.valid_from_chapter,
                change_description=f"Atmosphere Card: {card.environment_name} ({card.time_of_day.value}, {card.weather_condition.value})",
                visual_attributes={"environment_card": card.model_dump()},
            )
            self.db.add(asset_ver)
            self.db.commit()
            logger.info("Successfully persisted environment card %s to SQLite.", card.environment_id)
            return True

        except Exception as exc:
            logger.error("Failed to persist environment card to SQLite: %s", exc)
            if self.db:
                self.db.rollback()
            return False

    # =========================================================================
    # 5. RUNTIME SHOT ANCHOR RESOLUTION
    # =========================================================================

    def resolve_shot_environment_anchor(
        self,
        card: EnvironmentalAtmosphereCard,
    ) -> ShotEnvironmentAnchor:
        """
        Compile prompt tokens, directional vectors, and moisture ratings for diffusion inpainting.
        """
        # Combine all positive atmospheric tokens
        all_positive_tags = []
        all_positive_tags.extend(card.danbooru_lighting_tags)
        all_positive_tags.extend(card.danbooru_weather_tags)
        all_positive_tags.extend(card.danbooru_foliage_tags)
        all_positive_tags.extend(card.danbooru_surface_tags)

        # De-duplicate while preserving order
        seen = set()
        deduped_pos = []
        for tag in all_positive_tags:
            tag_clean = tag.strip().lower()
            if tag_clean and tag_clean not in seen:
                seen.add(tag_clean)
                deduped_pos.append(tag.strip())

        resolved_positive = ", ".join(deduped_pos)
        resolved_negative = ", ".join(card.negative_environment_tags)

        # Vector mapping
        dir_clean = card.lighting_rig.key_light_direction.lower()
        if "top-left" in dir_clean:
            key_vec = "top_left"
            shadow_vec = "bottom_right"
        elif "top-right" in dir_clean:
            key_vec = "top_right"
            shadow_vec = "bottom_left"
        elif "overhead" in dir_clean or "zenith" in dir_clean:
            key_vec = "overhead"
            shadow_vec = "directly_beneath"
        else:
            key_vec = "top_left"
            shadow_vec = "bottom_right"

        # Moisture mapping
        moisture = 0.2
        if card.weather_condition in [WeatherConditionEnum.HEAVY_RAIN, WeatherConditionEnum.ACID_DRIZZLE]:
            moisture = 0.85
        elif "wet" in resolved_positive or "puddle" in resolved_positive:
            moisture = 0.6
        elif card.weather_condition == WeatherConditionEnum.BLIZZARD:
            moisture = 0.5

        return ShotEnvironmentAnchor(
            environment_card_id=card.environment_id,
            time_of_day=card.time_of_day,
            weather_condition=card.weather_condition,
            resolved_lighting_block=resolved_positive,
            resolved_negative_block=resolved_negative,
            key_light_vector=key_vec,
            shadow_direction=shadow_vec,
            surface_moisture_level=moisture,
        )
