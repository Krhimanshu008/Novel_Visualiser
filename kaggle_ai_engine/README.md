# Novel Visualiser — Kaggle AI Engine ($10/Day Frontier AI)

This module implements the **Kaggle Benchmarks & Model Proxy** architecture specified in `benchmarks.md`, enabling Novel Visualiser to utilize Kaggle's **$10/day AI spend quota** for frontier models (**Gemini 2.5 Pro**, **Claude Sonnet 4**, **DeepSeek R1**, and **Qwen 2.5 72B**).

---

## 1. Why Kaggle Model Proxy?

Instead of paying for multiple third-party API subscriptions (Anthropic, OpenAI) or relying exclusively on base tier keys, Kaggle provides authenticated users with a **daily spend allowance** (e.g. **$10.00/day**) through its Model Proxy.

This subsystem leverages that allowance for the most computationally and cognitively demanding tasks in Novel Visualiser:

| Task | Traditional Limitation | Kaggle AI Engine Solution |
|---|---|---|
| **Visual Consistency** | Prose descriptions drift between shots; faces morph across angles. | **Claude Sonnet 4 / Gemini 2.5 Pro** synthesizes immutable **Danbooru Locked Identity Cards** and separate **Modular Outfit Blocks**. |
| **Story Extraction** | Local 14B models miss complex narrative subtext and micro visual deltas. | **Gemini 2.5 Pro** parses up to 1M token novel chapters with exact passage referencing and visual state tracking. |
| **Gate 3 QC** | OpenCV histograms only measure color palettes, not actual faces or outfits. | **Multimodal VLM inspection** inspects generated image pixels directly against canonical character references. |
| **Batch Processing** | Local machines overheat processing 50+ chapters. | **Kaggle Benchmark Tasks** run batch novel extraction asynchronously on Kaggle cloud servers. |

---

## 2. Directory Structure

```
kaggle_ai_engine/
├── __init__.py                # Package exports
├── config.py                  # Token management & auto-refresh logic
├── quota_manager.py           # $10/day spend tracker & safety guards
├── proxy_client.py            # Unified HTTP Model Proxy client (OpenAI & Gemini format)
├── consistency_architect.py   # Danbooru identity locking & outfit decomposition
├── story_extractor.py         # Frontier chapter extraction & visual deltas
├── semantic_qc.py             # Multimodal Gate 3 QC (pixel verification)
├── task_runner.py             # Cloud batch execution on Kaggle
├── integration.py             # Non-breaking adapter bridge for Novel Visualiser
├── cli.py                     # Command-line utility
├── outputs/                   # Downloaded task outputs
├── tasks/                     # Generated benchmark task scripts
└── README.md                  # This documentation
```

---

## 3. Quick Start & Authentication

### Step 1: Ensure Kaggle CLI is authenticated
If you haven't authenticated the Kaggle CLI on this machine:
```bash
# Recommended: One-time browser login
kaggle auth login
```
Alternatively, save your token to `~/.kaggle/access_token` or export `KAGGLE_API_TOKEN`.

### Step 2: Mint your Model Proxy token
Run the built-in auth command:
```bash
python -m kaggle_ai_engine.cli auth
```
*(This automatically calls `kaggle benchmarks auth -y` and writes `MODEL_PROXY_URL`, `MODEL_PROXY_API_KEY`, and `MODEL_PROXY_EXPIRY_TIME` to `.env`)*

### Step 3: Check your $10/day spend quota
```bash
python -m kaggle_ai_engine.cli quota
```
Output:
```
===========================================================
       KAGGLE MODEL PROXY AI SPEND QUOTA ($10/DAY)         
===========================================================
Period     | Used       | Remaining  | Total      | Refill Window
-----------------------------------------------------------------
Daily      | $0.15      | $9.85      | $10.00     | 2026-11-06T00:00:00+00:00
Monthly    | $1.20      | $98.80     | $100.00    | 2026-12-01T00:00:00+00:00
===========================================================
```

---

## 4. Key Capabilities & Usage

### A. Danbooru Character Identity Locking
Prevents facial and anatomy drift across shots by separating immutable traits from mutable outfits:

```python
from kaggle_ai_engine import VisualConsistencyArchitect

vca = VisualConsistencyArchitect()
card = vca.build_character_identity(
    character_name="Sunny",
    novel_context="Shadow Slave by Guiltythree",
    prose_descriptions=[
        "A delicate, pale young man with jet-black messy hair and pitch-dark eyes.",
        "Slender and gaunt from living in the outskirts.",
    ],
)
print(card.danbooru_identity_tags)
# Output: ['1boy', 'pale skin', 'messy black hair', 'sharp dark eyes', 'slender youth']
```

### B. Ordered Block Prompt Compilation
Assembles prompts strictly according to Illustrious / SDXL best practices:
```python
prompt = vca.compile_locked_prompt(
    identity=card,
    outfit=outfit_card,
    location=location_plate,
    shot_framing="wide shot",
    camera_angle="low angle dramatic",
    action_pose="holding coffee cup",
    expression="grimace, bitter expression",
)
```

### C. Multimodal Gate 3 QC (Pixel Inspection)
Verifies that the generated image actually contains the requested character and outfit:
```python
from kaggle_ai_engine import KaggleSemanticQCInspector

inspector = KaggleSemanticQCInspector()
verdict = inspector.evaluate_shot(
    generated_image_path="output/generated_images/shot_01.png",
    reference_image_path="data/references/Sunny/Sunny_byDr._Nero.jpg",
    expected_character="Sunny",
    expected_outfit="worn dark tunic",
    expected_elements=["coffee cup", "police station bench"],
)

if not verdict.overall_passed:
    print(f"QC Rejected! Defects: {verdict.detected_defects}")
    print(f"Suggestion: {verdict.remediation_suggestion}")
```

### D. Asynchronous Cloud Batch Processing
Offload heavy novel volumes to Kaggle's cloud runners:
```python
from kaggle_ai_engine import KaggleBenchmarkTaskRunner

runner = KaggleBenchmarkTaskRunner()
task_file = runner.create_batch_chapter_task(
    task_name="shadow-slave-vol1-storyboard",
    novel_title="Shadow Slave",
    chapter_range=list(range(1, 21)),
    sample_prompt="Extract storyboard breakdowns",
)

# Push & run on Kaggle
runner.push_task("shadow-slave-vol1-storyboard", task_file)
runner.run_task("shadow-slave-vol1-storyboard", models=["gemini-2.5-pro", "claude-sonnet-4"])
```

---

## 5. Seamless Pipeline Integration

In `novel_visualiser/app/agents/story_intel.py` or your execution scripts, you can simply import the bridge:

```python
from kaggle_ai_engine import KagglePipelineBridge

bridge = KagglePipelineBridge()

# If Kaggle Model Proxy is authenticated, use it; otherwise fallback
if bridge.is_operational():
    analysis = bridge.enhance_chapter_analysis(
        chapter_number=1,
        passages=chapter_passages,
        novel_title="Shadow Slave",
    )
```

