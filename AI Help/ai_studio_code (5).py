import tempfile
import unittest
import zipfile
from pathlib import Path

from app.database.session import init_db, get_session
from app.database.models import Project, Arc
from app.agents.reasoning_extractor import (
    UniversalReasoningExtractor,
    ReasonedEntityDossier,
    ReasonedAlias,
    ReasonedVisualEra,
)
from app.agents.research_agent import ResearchAgent
from app.orchestrator.resolver import get_safe_knowledge


class MockLotMReasoningExtractor(UniversalReasoningExtractor):
    """Simulates DeepSeek-R1 / QwQ output for 'Lord of the Mysteries' (Victorian Occult genre)."""

    def analyze_entity(self, **kwargs) -> ReasonedEntityDossier:
        return ReasonedEntityDossier(
            canonical_name="Klein Moretti",
            entity_type="character",
            genre_detected="Victorian Occult / Steampunk",
            first_appearance_chapter=1,
            reasoning_trace=(
                "Cross-referenced Volume 1 (Ch 1-213) and Volume 3 (Ch 483-732) from Arc Timeline. "
                "Assigned Tingen Nighthawk attire to Ch 1-213 and Gehrman Sparrow persona to Ch 483-732."
            ),
            aliases=[
                ReasonedAlias(alias="Zhou Mingrui", alias_type="birth_name", valid_from_chapter=1, reveal_chapter=1),
                ReasonedAlias(alias="The Fool", alias_type="tarot_persona", valid_from_chapter=5, reveal_chapter=5),
                ReasonedAlias(alias="Gehrman Sparrow", alias_type="persona", valid_from_chapter=483, reveal_chapter=483),
                ReasonedAlias(alias="Merlin Hermes", alias_type="persona", valid_from_chapter=1267, reveal_chapter=1267),
            ],
            visual_eras=[
                ReasonedVisualEra(
                    era_name="Volume 1: Tingen Nighthawk",
                    valid_from_chapter=1,
                    valid_until_chapter=213,
                    claims=[
                        {"claim_type": "physique", "claim_text": "scholarly young man with black hair and brown eyes"},
                        {"claim_type": "attire", "claim_text": "wearing a black double-breasted frock coat and half top hat"},
                    ],
                    danbooru_tags=["1boy", "solo", "black_hair", "brown_eyes", "frock_coat", "top_hat", "cane"],
                    flux_natural_prompt="A scholarly young man with black hair and dark brown eyes in a Victorian black frock coat and half top hat.",
                    anti_spoiler_negative_tags=["monocle", "tentacles"],
                ),
                ReasonedVisualEra(
                    era_name="Volume 3: Crazy Adventurer Gehrman Sparrow",
                    valid_from_chapter=483,
                    valid_until_chapter=732,
                    claims=[
                        {"claim_type": "physique", "claim_text": "cold sharp facial features and thin lips"},
                        {"claim_type": "attire", "claim_text": "wearing gold-rimmed spectacles and a black windbreaker"},
                    ],
                    danbooru_tags=["1boy", "solo", "black_hair", "brown_eyes", "gold_rimmed_glasses", "sharp_eyes", "black_windbreaker"],
                    flux_natural_prompt="A cold, sharp-featured bounty hunter wearing gold-rimmed spectacles and a black windbreaker.",
                    anti_spoiler_negative_tags=["tentacles"],
                ),
            ],
        )


class TestUniversalReasoningArchitecture(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        init_db(f"sqlite:///{(self.root / 'universal_test.db').as_posix()}")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_gate1_multi_era_expiration_and_anti_spoiler_negatives(self):
        """Proves traits expire across eras AND future traits automatically enter the negative prompt."""
        with get_session() as session:
            proj = Project(
                name="lotm_proj",
                novel_title="Lord of the Mysteries",
                author="Cuttlefish That Loves Diving",
                wiki_url="https://lordofthemysteries.fandom.com",
                genre_style="Victorian Occult",
            )
            session.add(proj)
            session.flush()

            session.add_all([
                Arc(project_id=proj.id, arc_number=1, name="Volume 1: Clown", start_chapter=1, end_chapter=213),
                Arc(project_id=proj.id, arc_number=2, name="Volume 2: Faceless", start_chapter=214, end_chapter=482),
                Arc(project_id=proj.id, arc_number=3, name="Volume 3: Traveler", start_chapter=483, end_chapter=732),
            ])
            session.commit()

            agent = ResearchAgent(
                reasoning_extractor=MockLotMReasoningExtractor(),
                export_dir=self.root / "exports",
                ref_dir=self.root / "refs",
            )
            summary = agent.research_character(session, proj.id, "Klein Moretti")
            entity_id = summary["entity_id"]

            # Query at Chapter 10 (Volume 1)
            ch10 = get_safe_knowledge(session, entity_id, current_chapter=10)
            self.assertIn("The Fool", ch10["safe_aliases"])
            self.assertNotIn("Gehrman Sparrow", ch10["safe_aliases"])
            self.assertIn("frock_coat", ch10["danbooru_prompt"])
            self.assertNotIn("gold_rimmed_glasses", ch10["danbooru_prompt"])
            # Future Ch 483 glasses MUST be automatically placed in the anti-spoiler negative prompt at Ch 10!
            self.assertIn("gold_rimmed_glasses", ch10["anti_spoiler_negative_prompt"])

            # Query at Chapter 500 (Volume 3: Gehrman Sparrow Era)
            ch500 = get_safe_knowledge(session, entity_id, current_chapter=500)
            self.assertIn("Gehrman Sparrow", ch500["safe_aliases"])
            self.assertNotIn("Merlin Hermes", ch500["safe_aliases"])
            # Volume 1 'frock_coat' expired at Ch 213, replaced by 'gold_rimmed_glasses'
            self.assertNotIn("frock_coat", ch500["danbooru_prompt"])
            self.assertIn("gold_rimmed_glasses", ch500["danbooru_prompt"])
            print("[PASS] Universal Gate 1: Multi-Era Trait Expiration & Auto Anti-Spoiler Negative Prompts")

    def test_gate2_epub_only_xianxia_novel_and_sha256_cache(self):
        """Proves an unpublished/indie Xianxia EPUB with NO wiki works and caches with 0 tokens."""
        epub_path = self.root / "Azure_Immortal.epub"
        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr(
                "OEBPS/ch1.xhtml",
                "<html><body><h1>Chapter 1: Qi Condensation</h1>"
                "<p>Lin Feng was a tall young man with long silver hair, emerald eyes, and flowing white robes.</p>"
                "</body></html>",
            )

        with get_session() as session:
            proj = Project(
                name="xianxia_epub",
                novel_title="Record of the Azure Immortal",
                author="IndieAuthor",
                wiki_url=None,  # No Wiki exists for this novel!
                epub_path=epub_path.as_posix(),
                genre_style="Xianxia Cultivation",
            )
            session.add(proj)
            session.commit()

            agent = ResearchAgent(
                reasoning_extractor=UniversalReasoningExtractor(offline_mode=True),
                export_dir=self.root / "exports",
                ref_dir=self.root / "refs",
                offline_mode=True,
            )
            res1 = agent.research_character(session, proj.id, "Lin Feng")
            self.assertGreaterEqual(res1["claims_count"], 1)

            ch1 = get_safe_knowledge(session, res1["entity_id"], current_chapter=1)
            self.assertIn("silver_hair", ch1["danbooru_prompt"])
            self.assertIn("emerald_eyes", ch1["danbooru_prompt"])
            self.assertIn("white_robes", ch1["danbooru_prompt"])

            # Re-running must hit the SHA-256 content_hash cache (0 LLM tokens!)
            res2 = agent.research_character(session, proj.id, "Lin Feng")
            self.assertEqual(res2.get("status"), "cached_hit")
            print("[PASS] Universal Gate 2: EPUB-Only Novel Discovery & SHA-256 Zero-Token Cache")


if __name__ == "__main__":
    unittest.main(verbosity=2)