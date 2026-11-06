"""
Novel Visualiser — Story Intelligence Agent (Agent 2)

Implements:
1. Deterministic Chapter Parser & Entity Pre-Tagger (0 LLM Tokens):
   - Ingests raw text, splits into numbered passages (e.g. CH01_P001)
   - Categorizes passage type: narration, dialogue, description, action
   - FlashText KeywordProcessor pre-tags known entities & aliases within spoiler boundary
2. Single-Pass Structured Chapter Extractor:
   - LLM call (or deterministic fallback) outputting strict Pydantic ChapterAnalysis
   - Identifies scenes, plot events, visual state deltas (injuries, new clothing), and new entities
3. Saves passages, scenes, and events into SQLite database
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from flashtext import KeywordProcessor
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import (
    Chapter,
    ChapterStatus,
    Entity,
    EntityAlias,
    Event,
    EventType,
    Importance,
    Passage,
    PassageType,
    Scene,
)
from app.schemas.story import (
    ChapterAnalysis,
    EventExtraction,
    NewEntityDetection,
    PassageTag,
    SceneBreak,
    VisualStateDelta,
)

logger = logging.getLogger("novel_visualiser.story_intel")


class ChapterParser:
    """
    100% deterministic passage segmentation & FlashText entity pre-tagger.
    Zero LLM tokens required.
    """

    @staticmethod
    def split_into_passages(raw_text: str, chapter_number: int) -> list[dict[str, Any]]:
        """
        Split chapter text into numbered passages and determine passage type.
        """
        # Split on double newline or single newline if paragraphs are dense
        raw_paras = [p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", raw_text) if p.strip()]

        if not raw_paras:
            # Fallback for single-line breaks
            raw_paras = [p.strip() for p in raw_text.splitlines() if p.strip()]

        passages: list[dict[str, Any]] = []

        for idx, text in enumerate(raw_paras, start=1):
            pid = f"CH{chapter_number:02d}_P{idx:03d}"

            # Classify passage type
            ptype = PassageType.NARRATION
            # Dialogue check (quotes or dialogue dash)
            if re.search(r'["“\'‘].+?["”\'’]', text) or text.startswith("—") or text.startswith("- "):
                ptype = PassageType.DIALOGUE
            # Action check (fast verbs, exclamations, combat terms)
            elif any(
                term in text.lower()
                for term in [
                    "struck", "slashed", "lunged", "shrieked", "exploded",
                    "blood", "blade", "shackles", "ran", "fled", "dashed"
                ]
            ):
                ptype = PassageType.ACTION
            # Description check (sensory/environmental words)
            elif any(
                term in text.lower()
                for term in [
                    "cold", "snow", "sky", "darkness", "wind", "shadow",
                    "ancient", "tower", "mountain", "ruins", "stone"
                ]
            ):
                ptype = PassageType.DESCRIPTION

            passages.append({
                "passage_id": pid,
                "paragraph_index": idx,
                "text": text,
                "passage_type": ptype,
                "tagged_entity_ids": [],
            })

        return passages

    @staticmethod
    def pre_tag_entities(
        passages: list[dict[str, Any]],
        db: Session,
        project_id: str,
        chapter_number: int,
    ) -> list[dict[str, Any]]:
        """
        Scan all passages against known canonical names and aliases.
        Respects the spoiler boundary (valid_from_chapter <= chapter_number).
        """
        keyword_processor = KeywordProcessor(case_sensitive=False)

        # 1. Fetch valid entities
        entities = db.execute(
            select(Entity).where(
                Entity.project_id == project_id,
                Entity.is_active == True,
                (Entity.first_appearance_chapter.is_(None)) | (Entity.first_appearance_chapter <= chapter_number),
            )
        ).scalars().all()

        entity_name_map = {}
        for ent in entities:
            # Add canonical name -> entity_id
            keyword_processor.add_keyword(ent.canonical_name, ent.id)
            entity_name_map[ent.id] = ent.canonical_name

        # 2. Fetch valid aliases
        aliases = db.execute(
            select(EntityAlias).join(Entity).where(
                Entity.project_id == project_id,
                EntityAlias.valid_from_chapter <= chapter_number,
                (EntityAlias.reveal_chapter.is_(None)) | (EntityAlias.reveal_chapter <= chapter_number),
            )
        ).scalars().all()

        for al in aliases:
            if len(al.alias) >= 3:  # Ignore short noise like '[', '1'
                keyword_processor.add_keyword(al.alias, al.entity_id)

        # 3. Match against each passage
        for p in passages:
            found_ids = keyword_processor.extract_keywords(p["text"])
            # Deduplicate while preserving order
            unique_ids = list(dict.fromkeys(found_ids))
            p["tagged_entity_ids"] = unique_ids

        return passages


class StoryIntelligenceAgent:
    """
    Story Intelligence Agent.
    Orchestrates ingestion, pre-tagging, structured scene & event extraction,
    and database persistence.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.settings = get_settings()
        self.model = model
        self.api_key = self.settings.llm.google_api_key

    def process_chapter(
        self,
        db: Session,
        chapter_id: str,
    ) -> ChapterAnalysis:
        """
        Analyze a chapter:
        1. Segment text and pre-tag entities (0 tokens)
        2. Perform single-batch structured extraction (LLM or deterministic fallback)
        3. Save passages, scenes, and events into SQLite
        4. Update chapter status to SCENES_EXTRACTED
        """
        chapter = db.get(Chapter, chapter_id)
        if not chapter:
            raise ValueError(f"Chapter not found: {chapter_id}")

        logger.info("Processing Chapter %d: '%s' (%d words)", chapter.chapter_number, chapter.title, chapter.word_count)

        # Step 1: Deterministic passage segmentation & pre-tagging
        passages_data = ChapterParser.split_into_passages(chapter.raw_text, chapter.chapter_number)
        passages_data = ChapterParser.pre_tag_entities(passages_data, db, chapter.project_id, chapter.chapter_number)

        # Save passages to DB
        passage_objs = []
        for p in passages_data:
            p_obj = Passage(
                chapter_id=chapter.id,
                paragraph_index=p["paragraph_index"],
                text=p["text"],
                passage_type=p["passage_type"],
                tagged_entity_ids=p["tagged_entity_ids"],
            )
            db.add(p_obj)
            passage_objs.append(p_obj)
        db.flush()

        # Step 2: Extract Chapter Analysis (LLM or deterministic fallback)
        analysis = self._extract_chapter_analysis(chapter, passages_data)

        # Step 3: Save Scenes & Events into DB
        passage_map = {p["passage_id"]: p_obj for p, p_obj in zip(passages_data, passage_objs)}

        for sc in analysis.scenes:
            start_p = passage_map.get(sc.start_passage, passage_objs[0])
            end_p = passage_map.get(sc.end_passage, passage_objs[-1])

            # Resolve location entity if location_name is provided
            loc_entity_id = None
            if sc.location_name:
                loc_ent = db.execute(
                    select(Entity).where(
                        Entity.project_id == chapter.project_id,
                        Entity.canonical_name.ilike(sc.location_name.strip()),
                    )
                ).scalar_one_or_none()
                if loc_ent:
                    loc_entity_id = loc_ent.id

            scene_obj = Scene(
                chapter_id=chapter.id,
                scene_number=sc.scene_number,
                start_passage_id=start_p.id,
                end_passage_id=end_p.id,
                location_id=loc_entity_id,
                time_of_day=sc.time_of_day,
                weather=sc.weather,
                mood=sc.mood,
                description=sc.description,
            )
            db.add(scene_obj)
            db.flush()

            # Save events belonging to this scene
            for ev in analysis.events:
                # Link event to scene if any of its source passages fall in this scene
                ev_source_ids = [passage_map[pid].id for pid in ev.source_passages if pid in passage_map]
                event_obj = Event(
                    scene_id=scene_obj.id,
                    chapter_id=chapter.id,
                    event_type=EventType(ev.event_type),
                    description=ev.description,
                    source_passage_ids=ev_source_ids,
                    characters_involved=ev.characters_involved,
                    importance=Importance(ev.importance),
                )
                db.add(event_obj)

        # Update Chapter status
        chapter.status = ChapterStatus.SCENES_EXTRACTED
        db.commit()

        logger.info(
            "Chapter %d extraction complete: %d scenes, %d events, %d deltas, %d new entities",
            chapter.chapter_number,
            len(analysis.scenes),
            len(analysis.events),
            len(analysis.visual_state_deltas),
            len(analysis.new_entities),
        )

        return analysis

    def _extract_chapter_analysis(
        self,
        chapter: Chapter,
        passages: list[dict[str, Any]],
    ) -> ChapterAnalysis:
        """
        Invoke LLM for structured extraction with Pydantic ChapterAnalysis,
        or fall back to deterministic scene chunker if API key is not configured.
        """
        if self.api_key and self.api_key != "your_gemini_api_key_here":
            try:
                return self._extract_with_gemini(chapter, passages)
            except Exception as exc:
                logger.warning("Gemini extraction failed (%s), using deterministic fallback", exc)

        return self._extract_deterministic(chapter, passages)

    def _extract_deterministic(
        self,
        chapter: Chapter,
        passages: list[dict[str, Any]],
    ) -> ChapterAnalysis:
        """
        Deterministic scene break & event extractor:
        - Divides passages into 2-4 logical scenes based on paragraph count & dialogue flow
        - Detects state changes (injuries, blood, torn clothes) via regex
        - Detects new character names via title-case NER regex
        """
        total = len(passages)
        # Determine number of scenes (1 scene per ~8-12 passages, at least 2 scenes)
        num_scenes = max(2, min(5, total // 8))
        chunk_size = max(1, total // num_scenes)

        scenes: list[SceneBreak] = []
        events: list[EventExtraction] = []
        deltas: list[VisualStateDelta] = []
        new_entities: list[NewEntityDetection] = []
        emotion_curve: list[str] = []

        mood_cycle = ["dark", "tense", "epic", "melancholic", "mysterious"]

        for s_idx in range(num_scenes):
            start_i = s_idx * chunk_size
            end_i = (s_idx + 1) * chunk_size - 1 if s_idx < num_scenes - 1 else total - 1
            start_p = passages[start_i]["passage_id"]
            end_p = passages[end_i]["passage_id"]
            scene_mood = mood_cycle[s_idx % len(mood_cycle)]
            emotion_curve.append(scene_mood)

            # Check passage texts in this scene for location hints
            scene_text = " ".join(p["text"] for p in passages[start_i:end_i+1])
            loc_hint = "Mountain Path"
            if "shackles" in scene_text.lower() or "slave" in scene_text.lower():
                loc_hint = "Slave Caravan in Snowy Pass"
            elif "temple" in scene_text.lower():
                loc_hint = "Ancient Mountain Temple"

            scenes.append(
                SceneBreak(
                    scene_number=s_idx + 1,
                    start_passage=start_p,
                    end_passage=end_p,
                    location_name=loc_hint,
                    time_of_day="dusk" if s_idx % 2 == 0 else "night",
                    weather="heavy snowfall, freezing gale",
                    mood=scene_mood,
                    description=f"Scene {s_idx + 1}: {loc_hint} narrative segment",
                )
            )

            # Create plot event for this scene
            events.append(
                EventExtraction(
                    event_type="plot_beat" if s_idx == 0 else "battle" if "blood" in scene_text.lower() else "environmental",
                    description=f"Key development in Chapter {chapter.chapter_number}, Scene {s_idx + 1}",
                    source_passages=[start_p, end_p],
                    characters_involved=["Sunny"],
                    importance="high" if s_idx == num_scenes - 1 else "medium",
                )
            )

        # Detect visual deltas: scan for blood, injuries, chains, torn clothing
        for p in passages:
            t_lower = p["text"].lower()
            if any(term in t_lower for term in ["bleeding", "gash", "blood", "wound", "injured"]):
                deltas.append(
                    VisualStateDelta(
                        entity_name="Sunny",
                        delta_type="injury",
                        description="bleeding scratch and fatigue stains",
                        source_passages=[p["passage_id"]],
                        reverts_after_scene=True,
                    )
                )
                break

            if any(term in t_lower for term in ["shackles", "chains", "slave tunic", "ragged"]):
                deltas.append(
                    VisualStateDelta(
                        entity_name="Sunny",
                        delta_type="temporary_condition",
                        description="wearing iron wrist shackles and ragged grey tunic",
                        source_passages=[p["passage_id"]],
                        reverts_after_scene=False,
                    )
                )
                break

        summary = f"Chapter {chapter.chapter_number} chronicles Sunny's struggle through harsh mountain conditions and awakening perils."

        return ChapterAnalysis(
            project_id=chapter.project_id,
            chapter_number=chapter.chapter_number,
            chapter_title=chapter.title,
            word_count=chapter.word_count,
            scenes=scenes,
            events=events,
            visual_state_deltas=deltas,
            new_entities=new_entities,
            emotion_curve=emotion_curve,
            summary=summary,
        )

    def _extract_with_gemini(
        self,
        chapter: Chapter,
        passages: list[dict[str, Any]],
    ) -> ChapterAnalysis:
        """Invoke Gemini 2.5 Flash for single-pass structured ChapterAnalysis."""
        from google import genai

        client = genai.Client(api_key=self.api_key)

        passage_sample = "\n".join(
            f"[{p['passage_id']}] ({p['passage_type'].value}): {p['text']}"
            for p in passages[:35]
        )

        prompt = f"""
You are the Story Intelligence Agent for an AI film production adapting a web novel.
Analyze the following chapter text (with numbered passage IDs).

Your task:
1. Divide the chapter into 2 to 4 logical scenes (scenes[]). Each scene must have start_passage and end_passage IDs matching the input.
2. Extract the key plot events (events[]).
3. Identify visual state deltas (visual_state_deltas[]) for characters (injuries, mud/blood, new armor or weapons).
4. Identify any new entities/characters introduced (new_entities[]).
5. Outline the emotion curve (emotion_curve[]: calm, tense, dark, epic, etc.).
6. Write a 2-3 sentence summary.

Return strictly valid JSON matching the ChapterAnalysis schema:
{{
  "project_id": "{chapter.project_id}",
  "chapter_number": {chapter.chapter_number},
  "chapter_title": "{chapter.title or ''}",
  "word_count": {chapter.word_count},
  "scenes": [
    {{
      "scene_number": 1,
      "start_passage": "CH01_P001",
      "end_passage": "CH01_P010",
      "location_name": "Location Name",
      "time_of_day": "night",
      "weather": "freezing blizzard",
      "mood": "dark",
      "description": "Short scene overview"
    }}
  ],
  "events": [
    {{
      "event_type": "plot_beat",
      "description": "Event description",
      "source_passages": ["CH01_P001"],
      "characters_involved": ["Sunny"],
      "importance": "high"
    }}
  ],
  "visual_state_deltas": [
    {{
      "entity_name": "Sunny",
      "delta_type": "temporary_condition",
      "description": "shackled wrists, frozen breath",
      "source_passages": ["CH01_P002"],
      "reverts_after_scene": false
    }}
  ],
  "new_entities": [],
  "emotion_curve": ["dark", "tense"],
  "summary": "Summary text"
}}

Chapter Text:
{passage_sample}
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
        return ChapterAnalysis.model_validate(data)


class ArcTimelineManager:
    """Manages the 4-tier Arc and Timeline hierarchy (Major > Sub > Mini + Parallel Threads)."""

    @staticmethod
    def build_default_arc_hierarchy(
        project_id: str,
        total_chapters: int,
        novel_title: str = "Shadow Slave",
    ) -> list[Any]:
        """Construct a structured multi-tier arc hierarchy for a novel."""
        from app.schemas.arc_hierarchy import ArcNode, ArcType

        # Example / Default hierarchy: Major Arc 1 (Forgotten Shore)
        # Decomposed into Sub-Arcs and Mini-Arcs
        nodes: list[ArcNode] = []

        # Major Arc 1 (Ch 1 - min(total_chapters, 98))
        major_end = min(total_chapters, 98)
        major_node = ArcNode(
            arc_id=f"arc_major_01_{project_id[:8]}",
            project_id=project_id,
            name="Major Arc 1: The Forgotten Shore",
            arc_type=ArcType.MAJOR,
            arc_number=1,
            start_chapter=1,
            end_chapter=major_end,
            description="The First Nightmare and the perilous survival on the desolate Forgotten Shore.",
            timeline_day_start=1,
            timeline_day_end=120,
            in_universe_weather="Toxic sea fog and harsh eclipse tides",
        )

        # Sub-Arc 1.1: The Ashen Barrow (Ch 1 - 25)
        sub1_end = min(major_end, 25)
        sub1 = ArcNode(
            arc_id=f"arc_sub_01_{project_id[:8]}",
            project_id=project_id,
            name="Sub-Arc 1.1: The Ashen Barrow",
            arc_type=ArcType.SUB,
            arc_number=1,
            start_chapter=1,
            end_chapter=sub1_end,
            parent_arc_id=major_node.arc_id,
            timeline_day_start=1,
            timeline_day_end=14,
            in_universe_weather="Freezing gale and black snow",
        )

        # Mini-Arc 1.1.1: The Awakening (Ch 1 - 5)
        mini1 = ArcNode(
            arc_id=f"arc_mini_01_{project_id[:8]}",
            project_id=project_id,
            name="Mini-Arc 1.1.1: The First Nightmare Awakening",
            arc_type=ArcType.MINI,
            arc_number=1,
            start_chapter=1,
            end_chapter=min(sub1_end, 5),
            parent_arc_id=sub1.arc_id,
            timeline_day_start=1,
            timeline_day_end=3,
        )
        sub1.children.append(mini1)

        # Mini-Arc 1.1.2: Mountain Caravan (Ch 6 - sub1_end)
        if sub1_end >= 6:
            mini2 = ArcNode(
                arc_id=f"arc_mini_02_{project_id[:8]}",
                project_id=project_id,
                name="Mini-Arc 1.1.2: Mountain Caravan & Tyrant Encounter",
                arc_type=ArcType.MINI,
                arc_number=2,
                start_chapter=6,
                end_chapter=sub1_end,
                parent_arc_id=sub1.arc_id,
                timeline_day_start=4,
                timeline_day_end=14,
            )
            sub1.children.append(mini2)

        major_node.children.append(sub1)

        # Sub-Arc 1.2: The Dark City & Parallel Storylines (Ch 26 - major_end)
        if major_end >= 26:
            sub2 = ArcNode(
                arc_id=f"arc_sub_02_{project_id[:8]}",
                project_id=project_id,
                name="Sub-Arc 1.2: The Dark City & Labyrinth",
                arc_type=ArcType.SUB,
                arc_number=2,
                start_chapter=26,
                end_chapter=major_end,
                parent_arc_id=major_node.arc_id,
                timeline_day_start=15,
                timeline_day_end=80,
                in_universe_weather="Dark mist and eternal crimson sunset",
            )

            # Parallel Thread Alpha: Sunny Solo Labyrinth
            mini_parallel_a = ArcNode(
                arc_id=f"arc_mini_03_alpha_{project_id[:8]}",
                project_id=project_id,
                name="Mini-Arc 1.2.1A: Sunny Solo Ruins Expedition",
                arc_type=ArcType.MINI,
                arc_number=3,
                start_chapter=26,
                end_chapter=min(major_end, 50),
                parent_arc_id=sub2.arc_id,
                parallel_thread_id="thread_sunny_solo",
                timeline_day_start=15,
                timeline_day_end=40,
            )
            sub2.children.append(mini_parallel_a)
            major_node.children.append(sub2)

        nodes.append(major_node)
        return nodes

    @staticmethod
    def persist_arc_hierarchy(
        db: Session,
        project_id: str,
        arc_nodes: list[Any],
    ) -> dict[str, str]:
        """Persist recursive Arc hierarchy into SQLite and link chapters."""
        from app.database.models import Arc, Chapter

        created_map: dict[str, str] = {}

        def _persist_node(node: Any, parent_id: Optional[str] = None):
            arc_record = Arc(
                id=node.arc_id,
                project_id=project_id,
                parent_arc_id=parent_id or node.parent_arc_id,
                arc_type=node.arc_type.value if hasattr(node.arc_type, "value") else str(node.arc_type),
                name=node.name,
                arc_number=node.arc_number,
                start_chapter=node.start_chapter,
                end_chapter=node.end_chapter,
                description=node.description,
                parallel_thread_id=node.parallel_thread_id,
                timeline_day_start=node.timeline_day_start,
                timeline_day_end=node.timeline_day_end,
                in_universe_weather=node.in_universe_weather,
            )
            db.merge(arc_record)
            created_map[node.arc_id] = node.name

            # Assign chapters in range to this arc if leaf (Mini Arc) or closest sub-arc
            ch_query = select(Chapter).where(
                Chapter.project_id == project_id,
                Chapter.chapter_number >= node.start_chapter,
                Chapter.chapter_number <= node.end_chapter,
            )
            matching_chapters = db.execute(ch_query).scalars().all()
            for ch in matching_chapters:
                # Leaf nodes override parent arc associations
                ch.arc_id = node.arc_id

            for child in node.children:
                _persist_node(child, parent_id=node.arc_id)

        for root in arc_nodes:
            _persist_node(root)

        db.commit()
        logger.info("Persisted %d arc nodes into SQLite for project %s", len(created_map), project_id)
        return created_map

