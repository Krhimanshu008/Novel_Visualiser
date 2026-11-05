"""
Novel Visualiser — Run Research & Populate Database

Fetches and downloads all research for a novel into:
1. SQLite Database: `novel_visualiser.db` (arcs, entities, aliases, claims, images)
2. Local Reference Images: `data/references/<character_name>/...`
3. Human-Readable Dossiers: `data/research_exports/<character_name>_dossier.md`
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import select
from app.config import get_settings
from app.database.models import Project, Arc, Entity, EntityAlias, ResearchClaim, ResearchImage
from app.database.session import init_db, get_session
from app.agents.research_agent import ResearchAgent


def main():
    print("=========================================================")
    print("   NOVEL VISUALISER — RESEARCH POPULATION RUNNER        ")
    print("=========================================================")

    settings = get_settings()
    settings.project.ensure_directories()
    db_file = os.path.abspath(settings.db.database_path)
    print(f"[*] Target SQLite Database: {db_file}")

    # Initialize tables
    init_db(db_file)

    novel_title = "Shadow Slave"
    author = "Guiltythree"
    subdomain = "shadowslave"

    with get_session() as session:
        # 1. Create or load Project
        project = session.execute(
            select(Project).where(Project.novel_title == novel_title)
        ).scalar_one_or_none()

        if not project:
            project = Project(
                name="Shadow Slave Production",
                novel_title=novel_title,
                author=author,
                description="High-accuracy anime/cinematic production of Shadow Slave",
            )
            session.add(project)
            session.commit()
            print(f"[+] Created Project: '{project.name}' (ID: {project.id})")
        else:
            print(f"[i] Using Existing Project: '{project.name}' (ID: {project.id})")

        proj_id = project.id

    agent = ResearchAgent(wiki_subdomain=subdomain)

    # 2. Research Story Volumes / Arcs
    print("\n[*] Step 1: Harvesting Story Volumes & Arcs from Wiki...")
    with get_session() as session:
        arcs = agent.research_arcs(session, proj_id)
        print(f"[+] Successfully saved {len(arcs)} Volumes into SQLite 'arcs' table:")
        for a in arcs[:5]:
            print(f"    - {a['name']} (Chapters {a['start']}-{a['end']})")
        if len(arcs) > 5:
            print(f"    - ... and {len(arcs) - 5} more volumes")

    # 3. Research Core Characters
    target_characters = ["Sunny", "Nephis"]
    print(f"\n[*] Step 2: Harvesting Characters, Aliases, Claims & Reference Images...")
    for char in target_characters:
        print(f"\n--- Researching: {char} ---")
        with get_session() as session:
            res = agent.research_character(session, proj_id, char)
            print(f"[+] Canonical Name: {res['canonical_name']}")
            print(f"[+] Aliases ({len(res['aliases_added'])}): {res['aliases_added'][:6]}...")
            print(f"[+] Visual Claims: {res['claims_count']}")
            print(f"[+] Reference Images Downloaded: {res['images_count']}")
            print(f"[+] Human-Readable Dossier Exported: {res['dossier_path']}")

    # 4. Final summary of disk contents
    print("\n=========================================================")
    print("   RESEARCH POPULATION COMPLETE! WHERE DATA IS STORED:   ")
    print("=========================================================")
    print(f"1. Database File:")
    print(f"   -> {db_file}")
    print(f"2. Local Reference Images (Open in File Explorer):")
    for char in target_characters:
        char_dir = PROJECT_ROOT / "data" / "references" / char
        if char_dir.exists():
            files = list(char_dir.glob("*.*"))
            print(f"   -> {char_dir} ({len(files)} files)")
            for f in files[:3]:
                print(f"      * {f.name} ({f.stat().st_size // 1024} KB)")
    print(f"3. Human-Readable Research Dossiers (Markdown / JSON):")
    exports_dir = PROJECT_ROOT / "data" / "research_exports"
    if exports_dir.exists():
        for doc in exports_dir.glob("*.md"):
            print(f"   -> {doc}")


if __name__ == "__main__":
    main()

