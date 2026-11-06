
Here is the complete execution package to upgrade your Researcher Agent into a 100% Novel-Agnostic, Reasoning-Driven Research Engine.
You can feed these 4 steps sequentially to your IDE agent (or drop the code directly into your files).
Step 1: Upgrade Database Schema & Anti-Spoiler Resolver (models.py & resolver.py)
What This Solves
Novel-Agnostic Projects: Project now stores wiki_url, epub_path, and genre_style (so any novel—whether it has a Fandom wiki, a custom MediaWiki, or only an EPUB file—works seamlessly).
Trait Expiration (valid_until_chapter): When a character grows older, changes outfits, or transforms in a later arc, old traits expire automatically instead of conflicting with new ones.
Automatic Anti-Spoiler Negative Prompts: In resolver.py, any visual claim or transformation that happens in a future chapter (valid_from_chapter > current_chapter) automatically has its tags injected into the ComfyUI Negative Prompt for earlier chapters so LoRAs or fanart bias never leak future spoilers.
Copy-Paste Prompt for Your IDE Agent
"Update novel_visualiser/app/database/models.py and novel_visualiser/app/orchestrator/resolver.py. In models.py, add wiki_url, epub_path, and genre_style to Project; add content_hash to ResearchSource; and add valid_until_chapter, era_name, danbooru_tags, flux_prompt, and negative_tags to ResearchClaim. In resolver.py, update get_safe_knowledge(db, entity_id, current_chapter) so it filters active claims where valid_from_chapter <= current_chapter AND (valid_until_chapter is None OR valid_until_chapter >= current_chapter), and automatically compiles future claims (valid_from_chapter > current_chapter) into anti_spoiler_negative_tags."
1A. novel_visualiser/app/database/models.py
Python
1B. novel_visualiser/app/orchestrator/resolver.py
Python
Step 2: Build the UniversalReasoningExtractor (reasoning_extractor.py)
This replaces all novel-specific static dictionaries with a Single-Pass Reasoning AI Engine that works across Groq (deepseek-r1-distill-llama-70b / qwen-qwq-32b), Google Gemini (gemini-2.5-flash / gemini-2.5-pro), and OpenRouter (deepseek/deepseek-r1), plus a universal offline parser for zero-token unit tests.
Copy-Paste Prompt for Your IDE Agent
"Create novel_visualiser/app/agents/reasoning_extractor.py. Implement UniversalReasoningExtractor that takes novel_title, genre_style, entity_name, entity_type, arc_timeline (from the arcs table), raw_infobox, and raw_prose. Prompt a Reasoning LLM (Groq / Gemini / OpenRouter) to map vague arc/volume references to exact chapter boundaries (valid_from_chapter and valid_until_chapter), classify spoiler aliases, and synthesize both Danbooru tags and natural Flux prompts alongside anti-spoiler negative tags in a single JSON response. Capture the <think> chain-of-thought summary for the markdown dossier."
novel_visualiser/app/agents/reasoning_extractor.py
Python
Step 3: Update ResearchAgent to Support Any Novel (Wiki + EPUB + Reasoning)
Now we update ResearchAgent so it:
Pulls the Arc Timeline from SQLite (Arc table) to feed into the Reasoning Model.
Scrapes any MediaWiki/Fandom URL configured on Project.wiki_url AND/OR harvests paragraphs from Project.epub_path (so even novels without a Wiki work out of the box!).
Computes a SHA-256 hash of the combined raw inputs (content_hash) so re-running research on an unchanged character costs 0 tokens.
Saves the Reasoned Eras, Claims, Danbooru Tags, Flux Prompts, and Reasoning Trace into SQLite and data/research_exports/<Name>_dossier.json & .md.
Copy-Paste Prompt for Your IDE Agent
"Update ResearchAgent in novel_visualiser/app/agents/research_agent.py to use UniversalReasoningExtractor. Load the project's Arc rows first so the Reasoning Model has the full novel chapter timeline. Support both Project.wiki_url (for any Fandom/MediaWiki novel) and Project.epub_path (via EPUBAppearanceHarvester for EPUB-only or hybrid novels). Store era_name, valid_from_chapter, valid_until_chapter, danbooru_tags, flux_prompt, and negative_tags on ResearchClaim, and write the <think> reasoning trace into the exported Markdown dossier."
novel_visualiser/app/agents/research_agent.py (Universal ResearchAgent Class)
Python
Step 4: Verify Across Multiple Novel Genres (test_universal_reasoning_agent.py)
This verification test proves that your Researcher Agent is 100% Novel-Agnostic by testing:
A Victorian Occult Novel (Lord of the Mysteries): Verifies that Klein Moretti's Volume 1 appearance (black frock coat) expires when his Volume 3 persona (Gehrman Sparrow, Ch. 483) begins, and that querying at Chapter 10 automatically puts his future gold-rimmed spectacles and pirate_admiral traits into the ComfyUI Anti-Spoiler Negative Prompt!
An EPUB-Only Xianxia/Cultivation Novel (No Wiki): Verifies that a novel with wiki_url=None and only an .epub file extracts canonical visual claims, caches the SHA-256 hash, and spends 0 tokens on repeat queries.
Copy-Paste Prompt for Your IDE Agent
"Create and run novel_visualiser/tests/test_universal_reasoning_agent.py to verify that the Universal Reasoning Architecture handles multi-era trait expiration (valid_until_chapter), automatic anti-spoiler negative prompts for future transformations, Arc-to-Chapter grounding across non-Shadow-Slave novels, and EPUB-only novels with SHA-256 idempotency caching."
novel_visualiser/tests/test_universal_reasoning_agent.py
Python
How to Execute & Test in PowerShell
Tell your IDE agent to apply Steps 1, 2, 3, and 4 above.
Run the verification suite in PowerShell:
Powershell
$env:PYTHONPATH='novel_visualiser'; python novel_visualiser/tests/test_universal_reasoning_agent.py
When running live against any novel, set your preferred Reasoning API key in .env:
Env
USE_OFFLINE_EXTRACTOR=0
GROQ_API_KEY=gsk_your_key_here
GROQ_REASONING_MODEL=qwen-qwq-32b
# Or GEMINI_API_KEY=... (gemini-2.5-flash)
# Or OPENROUTER_API_KEY=... (deepseek/deepseek-r1)