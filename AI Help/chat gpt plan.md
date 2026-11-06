Novel Visualiser – Ultimate AI workflow architecture
Your Kaggle model library is now a good foundation for a single-checkpoint, multi-stage image-generation pipeline. The important thing is not to load every model into every generation. Each model should have a specific responsibility, and the agents should decide which workflow to use for each scene.

I inspected your uploaded model inventory. It lists a 14.08 GB library with one main SDXL checkpoint, two IP-Adapter models, one CLIP Vision encoder, one ControlNet Union model, an SDXL VAE and an anime upscaler. ai_studio_code.txt

1. Your actual model inventory
Stored asset	Intended responsibility
waiNSFWIllustrious_v110.safetensors	Main image-generation checkpoint
sdxl_vae_fp16_fix.safetensors	SDXL image encoding and decoding
ip-adapter-plus_sdxl_vit-h.safetensors	Visual references for character appearance and style
ip-adapter-plus-face_sdxl_vit-h.safetensors	Face-reference conditioning
CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors	Image encoder required by your IP-Adapter setup
xinsir-controlnet-union-sdxl-1.0-promax.safetensors	Structural guidance: pose, depth, edges, line art and tile
4x-AnimeSharp.pth	Final illustration upscaling

Important inventory distinction: Your Pony, other-checkpoint, FaceID, LoRA and general-upscaler folders are currently empty in the supplied inventory. I will not design the core system around models that aren't confirmed as downloaded. ai_studio_code.txt

2. The overall architecture
I recommend separating Novel Visualiser into six connected systems.
Novel Visualiser production pipeline
1. Story Intelligence

Source ingestion · lore · arcs · chapters · scenes


2. Canon & Asset Intelligence

Characters · locations · items · creatures · effects


3. Visual Director

Scene composition · camera · lighting · visual references


4. Workflow Router

Select checkpoint · IP-Adapter · ControlNet · generation settings


5. ComfyUI Generation Engine

Generate · revise · inpaint · upscale


6. Visual QA & Continuity

Character checks · composition checks · approval · asset storage

Approved assets feed back into the Canon & Asset Library


The most important architectural decision is that ComfyUI should be the image-generation engine, not the brain of the application.

Your agents should make decisions, create structured prompts, select workflow templates and manage references. ComfyUI should execute the generation.

3. Map every model to a specific workflow
This is how I would assign responsibilities to your current library.









A. WAI Illustrious – Main generation engine

Used for:
- Character creation
- Environment creation
- Items and creatures
- Action scenes
- Final illustrations
This is the central checkpoint shared by all your SDXL workflows.












B. IP-Adapter Plus – Visual identity and reference

Used for:
- Reusing a character's appearance
- Maintaining outfit and visual design
- Reusing location appearance
- Maintaining an art direction across scenes
This is your primary visual-reference system.












C. IP-Adapter Plus Face – Facial reference

Used for:
- Character portraits
- Close-ups
- Repeated facial appearance
- Expression variations
It is not equivalent to a dedicated FaceID identity model.












D. Xinsir ControlNet Union ProMax – Composition control

Used for:
- OpenPose: body positions
- Depth: foreground and background structure
- Canny/Lineart: outlines and architecture
- Tile: detail preservation and enhancement
- Supported editing modes for more advanced refinement
One model covers multiple structural-control requirements. 
Hugging Face+1













E. 4x AnimeSharp – Final image enhancement

Used for:
- Increasing image resolution
- Improving illustration sharpness
- Exporting final artwork
Keep this as the final processing stage rather than applying it to every draft.



The IP-Adapter implementation supports transferring subject and style references, and its documented SDXL Plus models use the ViT-H image encoder that you have stored. 
cubiq/ComfyUI_IPAdapter_plus · GitHub+1

The workflow-selection matrix
This should become the routing logic inside your application.
Image requirement	IP-Adapter	ControlNet	Upscaler
New character design	None	Optional	Optional
Character portrait	Plus Face	Optional	Yes
Character full-body	Plus	OpenPose	Yes
Location creation	Optional	Depth/Lineart	Yes
Architecture	Optional	Canny/Depth	Yes
Battle scene	Plus	OpenPose + Depth	Yes
Weapon or artefact	Plus	Canny/Lineart	Yes
Outfit variation	Plus	OpenPose	Yes
Image restoration	Optional	Tile/Repaint	Optional
Large cinematic illustration	Plus	Depth + Pose	Yes

A major advantage is that you don't need a new model for every asset category.

One checkpoint can create everything, while reference images and structural controls determine what you are trying to create.

4. The eight ComfyUI workflows I recommend building
Don't create one giant workflow. Create independent workflow templates that the Workflow Router can combine according to the scene requirements.
Workflow 00 – Model diagnostic and base generation
Build first
Purpose: Confirm that every model works correctly before introducing agents.
Node architecture
CheckpointLoaderSimple

CLIP Text Encode – Positive / Negative

Empty Latent Image

KSampler

VAEDecode → SaveImage

Test these separately:
1. Character portrait.
2. Full-body character.
3. Dark fantasy environment.
4. Large-scale creature.
5. Action scene.
6. Architectural location.
Use the external SDXL VAE you downloaded and compare its output with the checkpoint's built-in VAE.
Workflow 01 – Character creation and identity locking
High priority
The biggest requirement of Novel Visualiser is that a character should remain recognizable across hundreds of illustrations.

I recommend a three-stage process.









Stage 1 – Character discovery

Generate 6–12 variations using WAI Illustrious without IP-Adapter.

Select the best overall face, appearance, silhouette and clothing.












Stage 2 – Canonical character sheet

Create:
- Face portrait
- Front full-body view
- Side view
- Back view
- Outfit details
- Special features
Save approved images as permanent references.












Stage 3 – Character state generation

Use IP-Adapter Plus and Plus Face with the canonical references.

Generate new expressions, poses, clothing states and action variations without redesigning the character from scratch.



Important: Keep identity and character state separate.

For example:
Permanent identity	Variable state
Face structure	Expression
Eye colour	Eye glow
Hair structure	Hair movement
Body proportions	Body posture
Permanent markings	Temporary wounds
Species	Transformation form

This will prevent your system from mistakenly treating a new outfit or injury as a completely new character.
Workflow 02 – Location and environment generation
Purpose: Create a reusable world rather than random backgrounds.

Generation process:
1. Create initial architectural concepts.
2. Select a canonical design.
3. Create multiple viewpoints.
4. Save depth and line-art references where useful.
5. Generate arc-specific environmental states.
6. Reuse those references for subsequent scenes.









Establishing view










Interior










Ground level










Changed environment


Use Depth for spatial construction and Lineart/Canny for more rigid architectural structures.
Workflow 03 – Scene composition and character placement
Core production workflow
This workflow will generate actual novel illustrations.

The input should include:
- Scene description.
- Characters appearing in the scene.
- Character reference images.
- Location reference images.
- Required body positions.
- Camera angle.
- Lighting and atmosphere.
- Emotional context.
The workflow:
Scene Specification
       |
       v
Load WAI Illustrious
       |
       +---- IP-Adapter Plus (Character)
       |
       +---- IP-Adapter Plus (Location, if needed)
       |
       +---- CLIP Vision
       |
       +---- ControlNet Union
                    |
                    +-- OpenPose
                    |
                    +-- Depth
       |
       v
Positive/Negative Conditioning
       |
       v
KSampler
       |
       v
VAE Decode
       |
       v
Scene Illustration


This is where you can significantly improve your previous generation quality.

A prompt saying "a warrior fighting a monster" does not adequately define the composition. Your Visual Director must specify where the warrior is, where the monster is, their relative scale, their movements, the camera and the environmental depth.
Workflow 04 – Items, weapons and artefacts
Every important weapon or artefact should have its own identity.

Create:
- Front view.
- Side view.
- Close-up.
- Material details.
- Magical effects.
- Scale reference.
- Damaged or upgraded versions.
Use IP-Adapter Plus for reference appearance and Canny/Lineart for preserving important contours.
Workflow 05 – Battle and cinematic illustrations
This deserves a specialised workflow because it will probably be one of the most demanding types of generation in dark fantasy.




















12













The recommended combination:
- WAI Illustrious for image generation.
- IP-Adapter Plus for the primary characters.
- IP-Adapter Plus Face for selected face-focused shots.
- OpenPose for body positions.
- Depth for foreground, middle ground and background.
- Canny for important structural details.
Don't force all controls into every scene. Use OpenPose when body positioning matters and Depth when spatial relationships matter. Combine them for particularly demanding scenes.
Workflow 06 – Refinement, editing and repair
Use this for correcting generated artwork without restarting everything.

Examples:
- Correct facial details.
- Change clothing.
- Repair hands.
- Extend the environment.
- Change composition.
- Restore background detail.
Your ProMax model card also documents tile deblur, tile variation, tile super-resolution, inpainting and outpainting capabilities. However, these need a compatible ComfyUI implementation and correct mode selection; the weight file alone doesn't guarantee those features are exposed in your current installation. 
Hugging Face+1

Workflow 07 – Final upscaling and export
Approved 1024px-class illustration

4x-AnimeSharp

Final high-resolution illustration
Save alongside image:
- Original generation
- Upscaled image
- Prompt and negative prompt
- Seed and generation settings
- Character and location references
- Model and workflow version


Keep an original non-upscaled copy. It will be useful for future image editing and alternative upscaling experiments.

5. How to organise the story and visual assets
This is where your earlier idea of inheritance across series, arcs, chapters and scenes becomes extremely valuable.

I recommend this hierarchy:
Series-level canon

Permanent character identities, universe, art direction, world rules

Arc-level variation

New outfits, transformations, locations, weapons and world changes

Chapter-level state

Character injuries, equipment, relationships, current environment

Scene-level state

Position, expression, action, lighting, camera and temporary effects





The inheritance system should work as follows:
- Global canon is inherited by every arc.
- Arc-level changes override global values only within that arc.
- Chapter-level changes override arc values for that chapter.
- Scene-level properties control the actual illustration.
Never overwrite the global character definition just because the character has changed appearance during one arc.
Database design using SQLite
I recommend starting with these tables:
Table	Responsibility
series	Novel/universe
sources	Research sources and references
arcs	Story arcs
chapters	Chapter records
scenes	Scene descriptions and order
entities	Characters, locations, items, creatures
entity_versions	Canonical and inherited states
entity_relationships	Character-item-location connections
reference_assets	Approved visual references
scene_entities	Entities involved in each scene
prompt_templates	Reusable prompt structures
generation_jobs	Generation requests and status
generation_outputs	Generated images
quality_reviews	Approval and rejection records

One essential addition: store the generation configuration with every image, including seed, checkpoint, workflow version, IP-Adapter weights, ControlNet mode and reference asset IDs. That gives you reproducibility and lets you trace why an image looks different from another.

6. Agentic AI architecture
Your earlier idea of adding an agent that researches the novel on the web should be incorporated into the system, not kept as a separate project.

I would divide the application into nine specialised agents.
Agent	What it does	Output
1. Research Agent	Researches the novel, available public illustrations, lore and world information	Research records
2. Story Analyst	Breaks the narrative into arcs, chapters and scenes	Story structure
3. Canon Manager	Maintains character, location and item facts	Canon database
4. Asset Designer	Designs initial character, location and artefact references	Asset concepts
5. Continuity Agent	Tracks changes across arcs and chapters	Character and world state
6. Scene Director	Determines camera, composition, lighting and action	Scene blueprint
7. Prompt Engineer	Converts the blueprint into model-ready prompts	Structured prompt
8. Workflow Router	Chooses the appropriate ComfyUI workflow	Generation configuration
9. Visual QA Agent	Checks generated illustrations against the scene and canon	Approval/rejection

How the agents should communicate
Research Agent + Story Analyst

Understand the source and narrative

Canon Manager + Continuity Agent

Resolve established facts and character states

Asset Designer + Scene Director

Plan visual references and illustration composition

Prompt Engineer + Workflow Router

Build prompts and choose model workflows

ComfyUI Execution

Generate one or more visual candidates

Visual QA Agent

Compare scene, character and image quality

Approved assets return to the reference library


An important design rule
Agents should not directly overwrite canonical character designs.

For example, suppose the system generates a new image where a character has a different eye colour.

The Visual QA Agent should compare it with the canonical record. The image should be marked for correction or review rather than allowing the Canon Manager to change the character's permanent eye colour.

The same applies to weapons, armour, injuries, transformations and locations.

Keep a human approval step for new canonical reference assets.

7. Prompt engineering system
I strongly recommend that you stop generating long prompts directly from the story text.

Instead, build a structured prompt compiler.

Every scene should have these fields:
Field	Example
Subject	Main character
Action	Fighting a giant creature
Location	Ruined underground citadel
Composition	Character in foreground, creature in background
Camera	Low-angle cinematic wide shot
Lighting	Cold moonlight, dark shadows
Atmosphere	Ancient, oppressive, mysterious
Character references	Character reference IDs
Location references	Location reference IDs
Control requirements	OpenPose + Depth
Art direction	Dark fantasy anime illustration
Negative constraints	No incorrect clothing, no extra limbs, no unwanted text

The compiler should then generate three separate things:

1. Positive prompt: What the model should render.

2. Negative prompt: What the model should avoid.

3. Workflow configuration: Which checkpoint, reference images, ControlNet types, weights, resolution and generation parameters to use.

This separation will make it possible to experiment with different workflows without rewriting the story intelligence.
Example of a scene request
Suppose a scene has a warrior standing in front of an enormous supernatural creature.

The compiler should resolve the following before submitting it:
{
  "scene_id": "ARC02_CH014_SC006",
  "scene_type": "cinematic_battle",
  "characters": [
    {
      "entity_id": "CHAR_001",
      "state": "battle_form",
      "reference": "CHAR_001_ARC02_FULLBODY"
    }
  ],
  "location": {
    "entity_id": "LOC_004",
    "state": "destroyed"
  },
  "composition": {
    "camera": "low_angle",
    "shot": "wide",
    "foreground": "warrior",
    "background": "giant_creature"
  },
  "workflow": "battle_scene_v1",
  "controls": [
    "openpose",
    "depth"
  ],
  "output": {
    "candidates": 4,
    "upscale": true
  }
}


The values here are illustrative, but this is the type of structured object your application should produce. The ComfyUI workflow then consumes it.

8. Kaggle and ComfyUI implementation strategy
Now we get to the actual technical implementation.

I recommend maintaining three independent layers.
Layer A – Model Dataset
Persistent input
Your existing 14.08 GB model collection. Keep the dataset as an immutable model library, with catalog.json as its inventory.


Layer B – ComfyUI Engine
Execution
Custom nodes, workflow JSONs, image processors, model loaders and API execution.


Layer C – Novel Visualiser
Application
SQLite, agents, story structure, prompt compiler, asset registry, generation queue and QA.


A. Custom nodes to verify
Your model inventory doesn't show which custom nodes have been installed. These are the components I would verify before building the actual workflows.
Component	Purpose
ComfyUI_IPAdapter_plus	IP-Adapter loading and application
ControlNet Union-compatible nodes	Loading ProMax and selecting its control mode
ComfyUI ControlNet Aux	OpenPose, depth, canny and lineart preprocessing
Standard ComfyUI nodes	Checkpoint, VAE, sampling, image loading/saving
Image upscale nodes	AnimeSharp processing

The official IP-Adapter implementation provides example workflows and requires the corresponding ViT-H encoder and SDXL weights. It has been placed in maintenance mode, so I recommend pinning a tested version rather than automatically updating it during production. 
cubiq/ComfyUI_IPAdapter_plus · GitHub+1


One technical warning: Xinsir Union requires the correct control-mode selection for its input map. Verify that your ComfyUI installation exposes a union-mode selector, such as SetUnionControlNetType or an equivalent implementation. A regular ControlNet loader alone may not be sufficient. 
GitHub+1


Also, your CLIP Vision file is an image encoder for IP-Adapter. It is not a replacement for the text-conditioning encoder provided by the SDXL checkpoint.
B. Starting generation settings
These are initial experimental settings, not guaranteed optimal settings for every scene.
Parameter	Starting point
Resolution	1024 × 1024
Steps	28–32
CFG	4.5–6
Sampler	DPM++ 2M SDE
Scheduler	Karras
Batch size on T4	1
Text-to-image denoise	1.0
IP-Adapter Plus weight	0.55–0.75
IP-Adapter Plus Face weight	0.60–0.80

For ControlNet, start with:
Control type	Initial strength
OpenPose	0.8–0.95
Depth	0.55–0.75
Canny/Lineart	0.4–0.65

Run small comparisons and save the settings with each output.
C. Memory management
Do not load every reference, IP-Adapter, VAE and ControlNet combination into one workflow by default.

Use workflow-specific loading:
- Character discovery: checkpoint only.
- Character portrait: checkpoint + Face IP-Adapter.
- Location: checkpoint + selected ControlNet.
- Battle: checkpoint + character reference + selected ControlNet.
- Upscaling: image + AnimeSharp.
For large VAE decodes, ComfyUI's tiled decoding can help reduce memory requirements. The VAE should always match the SDXL architecture of the checkpoint. 
GitHub


9. Proposed project folder structure
Keep your model dataset separate from your project data.
NovelVisualizer/
│
├── 00_Config/
│   ├── catalog.json
│   ├── model_profiles.json
│   └── workflow_registry.json
│
├── 01_Story/
│   ├── Sources/
│   ├── Arcs/
│   ├── Chapters/
│   └── Scenes/
│
├── 02_Database/
│   └── novel_visualiser.sqlite
│
├── 03_Entity_Bibles/
│   ├── Characters/
│   ├── Locations/
│   ├── Items/
│   ├── Creatures/
│   └── Effects/
│
├── 04_Reference_Assets/
│   ├── Canonical/
│   ├── Arc_Overrides/
│   ├── Chapter_States/
│   └── Scene_References/
│
├── 05_ComfyUI/
│   ├── Workflows/
│   │   ├── 00_Diagnostic/
│   │   ├── 01_Character/
│   │   ├── 02_Locations/
│   │   ├── 03_Items/
│   │   ├── 04_Scenes/
│   │   ├── 05_Battle/
│   │   ├── 06_Refinement/
│   │   └── 07_Upscaling/
│   │
│   └── Custom_Node_Manifest/
│
├── 06_Prompt_Engine/
│   ├── Templates/
│   ├── Compiled/
│   └── Negative_Prompts/
│
├── 07_Generation/
│   ├── Queue/
│   ├── Processing/
│   ├── Failed/
│   └── Completed/
│
├── 08_Quality_Control/
│   ├── Pending/
│   ├── Approved/
│   └── Rejected/
│
└── 09_Exports/
    ├── Original/
    ├── Upscaled/
    └── Final/


The actual model weights remain in the Kaggle model dataset; don't duplicate the 14 GB library inside the project.

10. Implementation roadmap – build in this order
I would develop the project in six phases. Don't start with the agents or a 100-image batch. First establish a repeatable image-generation foundation.
1. Phase 1 – Model validation and benchmarking
   First priority
   - Confirm every model loads successfully.
   - Test IP-Adapter Plus and Plus Face independently.
   - Test each ControlNet mode.
   - Compare the two VAE options.
   - Benchmark image quality, generation time and GPU memory.
   Deliverable: A working model library and diagnostic workflow.
2. Phase 2 – Build the eight ComfyUI workflow templates
   - Character discovery and identity.
   - Location generation.
   - Item generation.
   - Scene composition.
   - Battle illustrations.
   - Refinement.
   - Upscaling.
   Deliverable: Eight independently tested workflow JSONs.
3. Phase 3 – Build the SQLite visual asset system
   - Entity registry.
   - Character reference library.
   - Location and object registry.
   - Arc and chapter inheritance.
   - Version history.
   - Reference image mapping.
   Deliverable: A working visual asset database.
4. Phase 4 – Build the ComfyUI orchestration engine
   - Connect using the ComfyUI API.
   - Submit workflow JSONs.
   - Replace prompt and reference inputs dynamically.
   - Queue generation jobs.
   - Track seeds, settings and outputs.
   - Handle failed jobs and retries.
   Deliverable: Generate illustrations directly from structured scene data.
5. Phase 5 – Develop the agentic system
   - Research Agent.
   - Story Analyst.
   - Canon and Continuity Agents.
   - Asset Designer.
   - Visual Director.
   - Prompt Engineer.
   - Workflow Router.
   - Visual QA Agent.
   Deliverable: Automated story-to-illustration pipeline.
6. Phase 6 – Batch production and visual quality management
   - Process complete chapters.
   - Generate multiple candidates.
   - Compare images against approved references.
   - Human approval and rejection.
   - Automated upscaling.
   - Export final illustrations.
   Deliverable: Production-ready Novel Visualiser.
11. What should we add later?
Your current library is sufficient to develop the first production prototype. I would avoid downloading another large checkpoint until the existing combination has been properly benchmarked.

There are three possible future additions:
Future addition	Why it matters	Priority
Florence-2 or another vision model	Visual inspection, image descriptions and asset tagging	High
Dedicated SDXL FaceID package	Stronger identity-conditioning experiments	Medium
Character/style LoRAs	More specialised appearance and art direction	Medium

These are not part of the supplied inventory. In particular, your FaceID directory is empty. The IP-Adapter implementation documents that FaceID models generally need matching LoRA components and InsightFace support. 
cubiq/ComfyUI_IPAdapter_plus · GitHub+1


Also, image-generation models alone will not provide the Story Analyst, Research Agent or Visual QA reasoning. Those require an LLM/VLM layer, which can initially use an external API rather than another local GPU model.

12. My recommended final development priorities
What we should build first
Working prototype
First milestone: a complete chapter-to-image pipeline with reliable reference reuse.

ComfyUI workflow library
Priority 1

Eight independent workflows.

Visual asset database
Priority 2

Character, location, item and state inheritance.

ComfyUI API controller
Priority 3

Automated generation and batch processing.

Agentic intelligence
Priority 4

Research, story analysis, visual direction and QA.

Full chapter automation
Final milestone

Generate, review, approve and export.

One thing I want to establish before writing the actual workflow JSONs
The model inventory is available, but it doesn't tell me which ComfyUI custom nodes and preprocessors are installed in your Kaggle environment.

I would make the next step an environment audit followed by building the eight executable workflows, rather than writing JSONs that might reference missing nodes.

Please share the output of these commands from your Kaggle notebook:
import osCOMFY_DIR = "/kaggle/working/ComfyUI"print("CUSTOM NODES:")for folder in os.listdir(os.path.join(COMFY_DIR, "custom_nodes")):    print(folder)print("\nCOMFYUI VERSION:")!cd /kaggle/working/ComfyUI && git rev-parse HEADprint("\nGPU:")!nvidia-smi



We can then make every workflow compatible with your actual installation.