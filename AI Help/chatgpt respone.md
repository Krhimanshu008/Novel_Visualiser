I've read the complete Novel Visualiser — Comprehensive Project Analysis. My conclusion is that we should not rebuild Novel Visualiser. We should strengthen the visual generation architecture that you have already built.

You have actually solved several difficult problems already: story segmentation, research, spoiler protection, asset versioning, hierarchical resolution, scene planning, narration, and film assembly.

Your architecture also already contains the beginnings of a consistency engine:

- SQLite world canon with `entities`, `asset_versions`, `visual_refs` and `entity_relationships`.
- A hierarchical asset resolver.
- A Jinja2 prompt compiler.
- IP-Adapter Plus and ControlNet.
- Multiple ComfyUI workflow templates.
- A three-gate quality-control system.&#x20;

  Pasted text(4).txt



But there is a difference between having these components and having a reliable system that makes them work together.

I believe the problem is that the consistency requirements are not yet sufficiently formalised across the database, resolver, prompt compiler, ComfyUI workflow and quality-control stages.

I want to address this as an engineering problem, rather than downloading another checkpoint or trying random prompts.

# 1. Where I think the current architecture is failing

Based on your project analysis, here is my diagnosis. These are architectural risks inferred from the description, not confirmed bugs in the underlying Python code.

| Component       | What you have                          | What needs strengthening                                     |
| --------------- | -------------------------------------- | ------------------------------------------------------------ |
| Database        | World canon and asset versions         | Explicit visual identity and appearance inheritance          |
| Resolver        | Hierarchical asset resolution          | Resolving all visual dependencies for each shot              |
| Prompt compiler | Jinja2 templates                       | Structured prompt construction and constraint management     |
| ComfyUI         | 11 workflow templates                  | Standardised workflow contracts and validation               |
| References      | IP-Adapter and visual refs             | Reference selection, roles and prioritisation                |
| QC              | OpenCV, histograms, semantic checklist | Actual identity, outfit, object and environment verification |

There are five particularly important problems.

Problem A — Character identity and character appearance are different things.

Consider Sunny from Shadow Slave.

His permanent identity includes:

- Face structure.
- Hair.
- Eye characteristics.
- Body structure.
- Recognisable features.

But his appearance changes over time:

- Different clothing.
- Injuries.
- Dirt and blood.
- Different transformations.
- Different equipment.

These shouldn't overwrite one another.

Problem B — Clothing and accessories need independent identities.

A character wearing a particular jacket in Chapter 1 should not automatically wear that jacket in Chapter 50.

Likewise, an object such as a watch, weapon, necklace or ring needs its own visual reference and properties.

Problem C — A prompt should not be a large collection of descriptions.

The prompt compiler should know which parts of the prompt are immutable, inherited, scene-specific or optional.

Problem D — ComfyUI should not decide how to interpret your story.

It should receive an already-resolved generation specification. Your Python application should decide which models, references, adapters and control inputs are required.

Problem E — Your QC needs to measure the right things.

Your existing HSV histogram and structural similarity checks can identify some visual differences, but they cannot reliably establish that a character is the same person with the correct watch, clothing and facial features across different camera angles.

These five problems should be addressed together.

# 2. The architecture I recommend implementing

I want to introduce a dedicated subsystem:

# Visual Consistency Engine (VCE)

A new production subsystem within Novel Visualiser

01\. Visual Canon Manager

Defines permanent visual identities and approved references.

02\. Visual Version Manager

Manages arc, chapter and scene-level changes.

03\. Visual Context Resolver

Resolves characters, outfits, accessories, environments and their relationships for every shot.

04\. Structured Prompt Compiler

Combines permanent style, visual identity, references, scene action and camera instructions.

05\. ComfyUI Workflow Orchestrator

Selects and validates the appropriate workflow and its model dependencies.

06\. Visual Consistency Validator

Checks identity, outfit, objects, environment, art style and technical quality.

Approved Visual Asset

Stored with references, version history and generation metadata.

This should operate inside your existing LangGraph and database architecture rather than become a separate application.

# 3. First major implementation: Visual Canon

This is where I would start.

Your current database already has `entities`, `asset_versions`, `visual_refs` and `entity_relationships`. We should investigate their actual columns before creating additional tables.&#x20;

Pasted text(4).txt



I recommend organising every visual asset into the following categories.

A. Character Canon

[「VAZZROCK THE ANIMATION」第5話場面カット公開！阿座上洋平・前野智昭・江口拓也が演じるゲストキャラにも注目！ | 株式会社アニメイトホールディングスのプレスリリース](https://images.openai.com/static-rsc-4/z8Ze-v0dT30WUbCuDGvWpvNvI0r_Tr2PO6l0G2kVZevpwInPlEWASJFIYqIk1GcE8NKfyS58nbLFgmhJuxLLNjpm_RZbLjpyZe5IE1mdtOA_uCeX1zFeJ_fvuOLWtkCBaVIKX2KRwP7h6xnV4ygQmPq4yZ2ddgbNnZtYM7l_vVk?purpose=inline)

[愛徒勇気（CV：松岡禎丞） (c)ヒロユキ／スクウェアエニックス・「マンアシ」製作委員会 \[画像ギャラリー 3/11\] - コミックナタリー](https://images.openai.com/static-rsc-4/Yztt_ypjqJuGNIyN95ztm5ymw5_cs4ueg1w67y3h8JjuEbznpx4oWgoMHndzF0-mz194I8S9Pjnp-JBMD915f9_RcmbW4FsA4fMpo8Go_xPoQq0Fgw3PPD7PFziKHHbflCwhOTVOSg6Jixw-16_H-caJXMYxiRHgvgFD27N1J3U?purpose=inline)

[ゼロからウィッシュリスト1万人集めた3ヵ月間をまとめてみた！｜MYU＠雑音系少年少女](https://images.openai.com/static-rsc-4/NkQfFQKw-UBDMZJtY-QkVJQ_l-k1_e2DY9j3CsU1dGvstPRGEut3R-mY5H0n6P2oexZjKeFycmG5DTJLkN7tRCSlLj13ed-4pmp0fNxSmhBZEpaKkoFs7kLPXfpRr09YpFa6ruethoAKr0adxTilB_9i4rwfZVrk5J8NyXUcpBI?purpose=inline)

12

Permanent identity: face, body, hair, eyes, defining features and approved reference views.

B. Outfit Canon

[Respect Kirito (Sword Art Online) : r/respectthreads](https://images.openai.com/static-rsc-4/JFk_6-JOOln4piqzzfD5wF9aJhgPUEXVi7ccZbQ10KnoXksP2KfICi0Hc7UcVTAZX8gqn5W6TwGxQMcT20B199xvO0ABssYT-aObcb6lNIsPknGHwcvgQefT3ykA31pSUVHEZG4zU9rUIqmPhwi23fUhIWIv00HqYcQ47Yjv1eA?purpose=inline)

[AYNE: Character Model Sheet by fabianrensch on DeviantArt](https://images.openai.com/static-rsc-4/rblUCh7eNUWtwnluB0xpQrrOaL7CzLOlF-yw-D1Jz47pgAu3ca5CuxX449NPAde_cjMTv6ps814aXG_VGQWPfabTlHH8iNjWXbawYhIkImueuqv5sthgWWuZWXRltxY_M4zBbe_ikKxG36aB8392bPI628txSba2jWkBnr89wcA?purpose=inline)

[波佐見 利佳 - CHARACTER | 「アリス・ギア・アイギス」公式サイト](https://images.openai.com/static-rsc-4/nSPSQNe1ZjG1eQ8UPt-nLbv6d4oozAnXKVOFk7vn9EJl2gtHxGeN7IQTlNWDgpCc70DbTck3mbtY2Lf3u8RanhR0M0pfBgE13q4fm7fS6I9d4Ff8Zl7OxVic1WdE-8APsjkdZ5w1SoG-oYzqVW4GKMiNhzuKX2N_xcGpzqA91Qw?purpose=inline)

17

Individual outfits with their own IDs, designs, materials, colours and valid story periods.

C. Accessories and Equipment

[シルバーソードレイド攻略チームの装備発表！](https://images.openai.com/static-rsc-4/FPzvGAN2Fk--ElH298QWix_kqdR_WVZ9-ycwqtRCU9ojjsyRA7b4F3Fz-zJS_JrDTMiSSf93BjZz14gQd_fH0qvMlUFL2wT9G4JuVSdcajH7s5ezPrFslNQDxt3c2orLpv3rXi7CLkXwqxZybgwZEGJ1WhpIj--j_Uq01GQJXdM?purpose=inline)

[ミラン・ケストレル🦋⏳にじさんじ (@Milan_Kestrel) on X](https://images.openai.com/static-rsc-4/6LXIt24OEEpBuZfNfYCz9-Ya7DyPUX4UkPER3HbqpjoKRDo2fNIx4j5DDsK6JLv9pLIOJ3wOE9NgfAXU5YrmFoJgo4EVxrMp1aGjulu7hhBPrkmtP5Nc64korYI-9-niB_LCAQbRSiVfEDQwf778NmB4OueYz7iq-u3C9em8P1w?purpose=inline)

[Pin by Sinbad Sev Cookies on Sinbad | Sinbad magi, Aladdin magi, Anime magi](https://images.openai.com/static-rsc-4/sgdtx1in-m8UqqiN_EnUhuS2Stj1sRFZE5u0jegFgGnyfzRYgAD71Ebrn-l9bY8F7n8SrD5R_zULtZWE94lr85s4PZbfE5W3AUefrtU7xRjNO8iRUxIRh1Zv2tGQI8jmbQdYiiuVAl858FeZYkJKGr_HngGjwumB6WPTxiZCgvo?purpose=inline)

17

Each important object gets its own visual identity and character relationships.

D. Environment Canon

[Anime scenery of a castle with a river and a bridge - SeaArt AI](https://images.openai.com/static-rsc-4/To0KaoYzwSRW1nDMCN10op3PGnIflD8_PHG64uHWL4Ommlonhj9hrjmO0ZT_evc_oIAdO5aDC4zZyEE3usuk-qqRGeH_OZKNB3966XST92AcEHuAw4vo41ShjBI7nsB8ZKHUzbgImgPkd9gx33lhRdjKrb1glmd5ItWkbWLSLDo?purpose=inline)

[Genshin Impact – Step into a Vast Magical World for Adventure](https://images.openai.com/static-rsc-4/HdlrWrk9IHx6G2D9TSNEsU5Kq7FqVHzcNVWrPSX-GfdCx4Rl7841P-lGSLqM57M-jU1IZA-xb_e7BX5LC9LNrG8lUY2Y2GHTVs1aIyhF6kyHAABrdTBbWJRBQtXSJrPDujgUhj8uDZWvZgdjgqra7M-bt4eiZ4LUxca4M2IO8v0?purpose=inline)

[Houses with a Story: Yoshida Seiji Art Works - Seiji Youshida | moom bookshop - art books and magazines](https://images.openai.com/static-rsc-4/fcWwE-pDB0jpxQm4YQTWOHDzegftxcTks6gx4coLrOvnyzpae_IFHkybSgr82Hu1sJWcSyqz_6Xn4pJoeNOXqfps_URVHQOizzXfhPMlHlMTtmWNgbmIVr-xOvKNdkaSHQyAMyEz8f1m6ukm6vQoVpdbFKfNoQg-La4E08j3clM?purpose=inline)

14

Architecture, geography, lighting characteristics, colour palette and multiple camera-angle references.

## The inheritance principle

Imagine we are generating a chapter containing Sunny.

The resolver should not simply retrieve Sunny's description.

It should retrieve:

```
CHARACTER
  CHAR-001
    ├── Canonical identity
    ├── Physical appearance
    ├── Arc appearance version
    └── Chapter-specific condition

OUTFIT
  OUTFIT-003
    ├── Design reference
    ├── Colour
    ├── Material
    └── Current condition

ACCESSORIES
  ITEM-008
  ITEM-011

LOCATION
  LOC-004
    ├── Environment identity
    ├── Architecture
    ├── Current state
    └── Camera references
```

The scene should inherit these assets automatically.

An injury should change the character's temporary appearance, not rewrite his canonical identity.

An outfit change should update the outfit relationship, not the character's permanent appearance.

This is essential for preventing visual drift across hundreds of chapters.

# 4. Second major implementation: A proper prompt architecture

This is probably one of the biggest improvements we need to make.

Your existing `prompt_compiler.py` uses Jinja2. That's perfectly fine. The problem is what we feed into it and how that information is organised.

I want to eliminate the practice of building one enormous prompt from several descriptions.

Instead, introduce an intermediate data structure called `ShotVisualContext`.

## Example of a resolved shot

Suppose the story says:

Sunny is standing in a dark room, holding a sword. His clothes are damaged and he is looking towards the entrance.

Before generating an image, the resolver should produce something like:

```
{
  "shot_id": "CH01_SC005_SH002",

  "style_profile": "STYLE-001",

  "subjects": [
    {
      "entity_id": "CHAR-001",
      "identity_version": "CANON-001",
      "appearance_version": "CH01-AP-001",
      "outfit_id": "OUTFIT-002",
      "accessories": [],
      "temporary_state": [
        "torn clothing",
        "visible injuries"
      ],
      "action": "standing and holding a sword"
    }
  ],

  "environment": {
    "location_id": "LOC-001",
    "location_version": "LOC-001-V02",
    "lighting": "dim interior"
  },

  "composition": {
    "camera": "medium shot",
    "angle": "low angle",
    "focus": "protagonist"
  },

  "required_constraints": [
    "preserve canonical face",
    "preserve hairstyle",
    "use specified outfit",
    "include sword",
    "preserve established location design"
  ]
}
```

This is an intermediate representation, not yet the final ComfyUI prompt.

The advantage is that the prompt compiler no longer needs to understand the entire novel. It only needs to convert a verified visual specification into generation instructions.

## The actual prompt structure

I propose the following fixed prompt architecture:

## Prompt Compiler

01\. Global Style Block

Always inherited from the project style profile.

02\. Character Identity Block

Canonical face, hair, eyes, body and distinguishing features.

03\. Appearance Block

Current outfit, condition, injuries and temporary appearance.

04\. Object and Accessory Block

Weapons, jewellery, equipment and other required items.

05\. Environment Block

Architecture, background, spatial relationships and environmental state.

06\. Action and Composition Block

Movement, pose, camera, framing and perspective.

07\. Lighting and Rendering Block

Lighting conditions and rendering requirements.

08\. Negative Constraints

Unwanted clothing, wrong accessories, identity changes, extra limbs, etc.

The ordering is deliberate.

A scene-specific instruction should not accidentally override the canonical character identity.

## A very important rule

I want to separate three different kinds of prompt information.

| Information                   | Who controls it?       |
| ----------------------------- | ---------------------- |
| Character identity            | Visual Canon Manager   |
| Character's current condition | Version Manager        |
| Scene action and composition  | Scene Director         |
| Art style                     | Global Style Profile   |
| Final prompt assembly         | Python Prompt Compiler |

The LLM can interpret the story and propose a visual scene. It must not silently rewrite approved visual identities or the project's rendering style.

We should also store the source and priority of every resolved attribute. When two instructions conflict, the system should report the conflict rather than concatenate both descriptions.

# 5. Third major implementation: Standardise ComfyUI workflows

This is particularly important given the problems you've encountered with model selection, undefined nodes, failed generation and workflow execution.

Your report describes 11 ComfyUI workflows and an existing client that submits jobs over HTTP/WebSocket.&#x20;

Pasted text(4).txt



I don't recommend increasing the number of workflows right now.

Instead, let's standardise them around five production responsibilities.

| Workflow       | Responsibility                         |
| -------------- | -------------------------------------- |
| WF-REF-001     | Character reference sheets             |
| WF-REF-002     | Environment reference sheets           |
| WF-SHOT-001    | Normal scene generation                |
| WF-EDIT-001    | Character, object and scene correction |
| WF-UPSCALE-001 | Final image upscaling                  |

These are proposed workflow contracts, not claims about the existing WF-001 to WF-008 files.

## How the workflow system should work

Currently, we should avoid allowing different agents to directly manipulate arbitrary ComfyUI nodes.

Instead, introduce a workflow registry.

For example:

```
{
  "workflow_id": "WF-SHOT-001",
  "workflow_version": "1.0.0",

  "model_requirements": {
    "checkpoint": "waiNSFWIllustrious_v110.safetensors",
    "vae": "sdxl_vae_fp16_fix.safetensors"
  },

  "inputs": {
    "positive_prompt": "required",
    "negative_prompt": "required",
    "character_reference": "optional",
    "environment_reference": "optional",
    "pose_reference": "optional",
    "seed": "required",
    "width": "required",
    "height": "required"
  },

  "capabilities": [
    "character_reference",
    "environment_reference",
    "controlnet",
    "text_to_image"
  ]
}
```

The actual file names and graph inputs must be verified against your installed models and workflow JSON.

### Introduce a node mapping layer

Your Python application should know what each ComfyUI node is responsible for.

For example:

```
NODE_MAP = {    "checkpoint": "node_id_checkpoint",    "vae": "node_id_vae",    "positive": "node_id_positive",    "negative": "node_id_negative",    "ipadapter": "node_id_ipadapter",    "controlnet": "node_id_controlnet",    "sampler": "node_id_sampler",    "output": "node_id_output"}
```

The node IDs above are placeholders.

The workflow registry should be the only place that knows the actual node IDs. Your scene-generation agent should never need to know them.

Before submission, validate:

1. Does the workflow JSON exist?
2. Are all required node IDs present?
3. Are the expected node classes installed?
4. Are the required checkpoint and VAE available?
5. Are IP-Adapter and ControlNet models compatible with the checkpoint?
6. Are all mandatory inputs populated?
7. Is the ComfyUI server reachable?

Only then should the job enter the generation queue.

## One particularly important correction

Your project report identifies an ephemeral Cloudflare tunnel and an OpenCV mock-canvas fallback when the ComfyUI connection fails.&#x20;

Pasted text(4).txt



We should change this immediately.

Production safeguard

Never silently substitute a mock image for a failed AI generation.

When ComfyUI is unavailable:

- Mark the generation job as `BLOCKED`.
- Record the connection error.
- Notify the orchestrator.
- Do not approve the mock image.
- Resume from the failed generation stage when the service is available.

Otherwise, you can end up with a successfully rendered video that contains images which were never actually generated by the intended AI pipeline.

# 6. Fourth major implementation: Visual reference management

This is where IP-Adapter becomes useful.

But I want us to stop thinking of IP-Adapter as a single character consistency solution.

It should be part of a reference-selection strategy.

For every scene, the resolver should provide separate references:

| Reference role      | Purpose                                |
| ------------------- | -------------------------------------- |
| Character identity  | Preserve the character's appearance    |
| Character full-body | Maintain body proportions and clothing |
| Outfit              | Maintain costume design                |
| Accessory           | Maintain important object details      |
| Environment         | Maintain location identity             |
| Composition         | Guide pose and camera placement        |
| Style               | Preserve global rendering aesthetic    |

Not every generation requires every reference.

A character close-up may need a face reference and an outfit reference.

A wide establishing shot may need an environment reference and a pose/composition guide.

## How I would use your existing models

Your project analysis already specifies WAI Illustrious XL, SDXL VAE, IP-Adapter Plus, Face IP-Adapter, ControlNet Union and AnimeSharp.&#x20;

Pasted text(4).txt



I would use them approximately as follows:

| Technology               | Assigned responsibility                                  |
| ------------------------ | -------------------------------------------------------- |
| WAI Illustrious XL       | Base image generation                                    |
| IP-Adapter Plus          | Visual reference conditioning                            |
| Face IP-Adapter          | Experimental face preservation, subject to compatibility |
| ControlNet OpenPose      | Character pose                                           |
| ControlNet Depth         | Spatial and structural guidance                          |
| ControlNet Canny/Lineart | Composition and outlines                                 |
| Qwen Image Edit 2511     | Targeted image corrections                               |
| Florence-2               | Reference-image tagging                                  |
| AnimeSharp               | Final upscaling                                          |

One caution: FaceID methods can depend on face-recognition models that don't work reliably on all anime illustrations. We should test them against our actual character references rather than assume they will provide identity locking.

Also, keep style references and identity references separate. Otherwise, changing a character reference could unintentionally change the rendering style.

# 7. Fifth major implementation: A better visual quality-control engine

Your existing three-gate QC system is a good foundation.

But I would expand its responsibilities.

Gate 0 — Generation Integrity

New

- Correct ComfyUI workflow executed.
- Correct checkpoint loaded.
- All input references resolved.
- No mock images.
- Generation completed and output verified.

Gate 1 — Technical Quality

Retain your existing OpenCV checks:

- Blur.
- Exposure.
- Resolution.
- Corrupted images.
- Structural quality.

Gate 2 — Visual Identity and Continuity

Major improvement

Separate checks for:

| Test                          | Method                                       |
| ----------------------------- | -------------------------------------------- |
| Character identity            | Reference embeddings and visual comparison   |
| Hairstyle and facial features | Image-region comparison and vision model     |
| Outfit                        | Visual checklist                             |
| Accessories                   | Required-object verification                 |
| Environment                   | Comparison with multiple location references |
| Art style                     | Style-reference embedding comparison         |

Gate 3 — Narrative and Semantic Validation

- Correct number of characters.
- Correct character actions.
- Correct outfit for the chapter.
- Required objects present.
- No forbidden future-story appearance.
- Lighting and atmosphere match the scene.
- No contradictory visual details.

Human Review

Failed or uncertain assets are sent for manual approval, with the reason for failure recorded.

For the technical implementation, I would evaluate DINOv2 or CLIP-style image embeddings alongside a vision-language model.

One important point: don't compare an entire close-up to an entire wide-angle reference using SSIM and expect meaningful identity verification.

We need multiple approved references and separate comparisons for faces, clothing, objects and backgrounds. Embedding thresholds must also be calibrated against our actual generated images, not arbitrarily selected.

The validator should return structured results, such as:

```
{
  "shot_id": "CH01_SC005_SH002",
  "status": "REVIEW_REQUIRED",
  "checks": {
    "character_identity": "PASS",
    "hairstyle": "PASS",
    "outfit": "FAIL",
    "accessories": "PASS",
    "environment": "PASS",
    "style": "PASS"
  },
  "issues": [
    {
      "type": "OUTFIT_MISMATCH",
      "expected": "OUTFIT-002",
      "observed": "Different upper garment"
    }
  ]
}
```

Most importantly, a failed outfit check should trigger a targeted correction, not automatically regenerate the entire scene with a different random seed.

# 8. Sixth major implementation: Lock the global art style

This is a separate requirement from character consistency.

Remember the 50-hour video you showed me? The reason the illustrations can look like they belong to the same production is likely that the creator maintains a relatively stable rendering configuration.

We should build the same capability into Novel Visualiser.

Create a `VisualStyleProfile`.

For example:

```
{
  "profile_id": "STYLE-001",
  "name": "Cinematic Manhwa",

  "checkpoint": "waiNSFWIllustrious_v110.safetensors",
  "vae": "sdxl_vae_fp16_fix.safetensors",

  "rendering": {
    "sampler": "verified_sampler",
    "scheduler": "verified_scheduler",
    "steps": 30,
    "cfg": 6.5,
    "resolution": [1024, 1024]
  },

  "style_prompt": [
    "cinematic anime illustration",
    "semi-realistic manhwa rendering",
    "detailed hair strands",
    "intricate fabric textures",
    "cinematic directional lighting",
    "atmospheric environmental depth"
  ],

  "negative_prompt": [
    "flat rendering",
    "inconsistent character design",
    "poor anatomy",
    "low detail"
  ],

  "locked": true,
  "version": 1
}
```

The numeric settings are an initial example, not verified optimal settings.

The important thing is that every generation job references a particular style profile version.

If you decide to change the rendering style, create `STYLE-002`. Don't silently modify `STYLE-001` halfway through a chapter.

I also want a generation manifest for every image:

```
SHOT_ID
STYLE_PROFILE_VERSION
WORKFLOW_VERSION
CHECKPOINT_HASH
VAE_HASH
LORA_CONFIGURATION
SAMPLER_SETTINGS
SEED
PROMPT_VERSION
CHARACTER_REFERENCE_IDS
LOCATION_REFERENCE_IDS
OUTFIT_REFERENCE_IDS
CONTROLNET_SETTINGS
GENERATION_TIMESTAMP
```

This gives us reproducibility and makes debugging much easier.

# 9. How I would modify your existing codebase

We should preserve your current architecture and introduce the consistency engine incrementally.

Based on the project analysis, these are the modules I would work on.

| Existing module               | Modification                                           |
| ----------------------------- | ------------------------------------------------------ |
| `database/models.py`          | Extend visual canon, versions and reference metadata   |
| `orchestrator/resolver.py`    | Implement complete visual context resolution           |
| `services/prompt_compiler.py` | Structured prompt compiler                             |
| `services/comfyui_client.py`  | Workflow registry, validation and generation contracts |
| `services/qc_service.py`      | Visual identity and continuity checks                  |
| `agents/world_architect.py`   | Manage visual asset creation and version changes       |
| `agents/scene_director.py`    | Output structured shot specifications                  |
| `orchestrator/graph.py`       | Add validation and approval transitions                |
| `orchestrator/state.py`       | Track resolved assets, prompt bundles and QC results   |

I would not modify the audio engine or film editor for this project phase.

## Proposed LangGraph execution

\#chatgpt-mermaid-\_r_1d1\_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(237, 237, 237);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-\_r_1d1\_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_1d1\_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-\_r_1d1\_ .error-icon{fill:rgb(48, 48, 48);}#chatgpt-mermaid-\_r_1d1\_ .error-text{fill:rgb(237, 237, 237);stroke:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1d1\_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-\_r_1d1\_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-\_r_1d1\_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-\_r_1d1\_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-\_r_1d1\_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-\_r_1d1\_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-\_r_1d1\_ .marker{fill:rgb(175, 175, 175);stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_1d1\_ .marker.cross{stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_1d1\_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-\_r_1d1\_ p{margin:0;}#chatgpt-mermaid-\_r_1d1\_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1d1\_ .cluster-label text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1d1\_ .cluster-label span{color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1d1\_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-\_r_1d1\_ .label text,#chatgpt-mermaid-\_r_1d1\_ span{fill:rgb(237, 237, 237);color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1d1\_ .node rect,#chatgpt-mermaid-\_r_1d1\_ .node circle,#chatgpt-mermaid-\_r_1d1\_ .node ellipse,#chatgpt-mermaid-\_r_1d1\_ .node polygon,#chatgpt-mermaid-\_r_1d1\_ .node path{fill:rgb(9, 23, 44);stroke:rgb(31, 78, 148);stroke-width:1px;}#chatgpt-mermaid-\_r_1d1\_ .rough-node .label text,#chatgpt-mermaid-\_r_1d1\_ .node .label text,#chatgpt-mermaid-\_r_1d1\_ .image-shape .label,#chatgpt-mermaid-\_r_1d1\_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-\_r_1d1\_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-\_r_1d1\_ .rough-node .label,#chatgpt-mermaid-\_r_1d1\_ .node .label,#chatgpt-mermaid-\_r_1d1\_ .image-shape .label,#chatgpt-mermaid-\_r_1d1\_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-\_r_1d1\_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-\_r_1d1\_ .root .anchor path{fill:rgb(175, 175, 175)!important;stroke-width:0;stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_1d1\_ .arrowheadPath{fill:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_1d1\_ .edgePath .path{stroke:rgb(175, 175, 175);stroke-width:1px;}#chatgpt-mermaid-\_r_1d1\_ .flowchart-link{stroke:rgb(175, 175, 175);fill:none;}#chatgpt-mermaid-\_r_1d1\_ .edgeLabel{background-color:rgb(0, 0, 0);text-align:center;}#chatgpt-mermaid-\_r_1d1\_ .edgeLabel p{background-color:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_1d1\_ .edgeLabel rect{opacity:0.5;background-color:rgb(0, 0, 0);fill:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_1d1\_ .labelBkg{background-color:rgba(0, 0, 0, 0.5);}#chatgpt-mermaid-\_r_1d1\_ .cluster rect{fill:rgb(48, 48, 48);stroke:rgba(255, 255, 255, 0.15);stroke-width:1px;}#chatgpt-mermaid-\_r_1d1\_ .cluster text{fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1d1\_ .cluster span{color:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1d1\_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(48, 48, 48);border:1px solid rgba(255, 255, 255, 0.15);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-\_r_1d1\_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(237, 237, 237);}#chatgpt-mermaid-\_r_1d1\_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-\_r_1d1\_ .icon-shape,#chatgpt-mermaid-\_r_1d1\_ .image-shape{background-color:rgb(0, 0, 0);text-align:center;}#chatgpt-mermaid-\_r_1d1\_ .icon-shape p,#chatgpt-mermaid-\_r_1d1\_ .image-shape p{background-color:rgb(0, 0, 0);padding:2px;}#chatgpt-mermaid-\_r_1d1\_ .icon-shape .label rect,#chatgpt-mermaid-\_r_1d1\_ .image-shape .label rect{opacity:0.5;background-color:rgb(0, 0, 0);fill:rgb(0, 0, 0);}#chatgpt-mermaid-\_r_1d1\_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-\_r_1d1\_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-\_r_1d1\_ .node .neo-node{stroke:rgb(31, 78, 148);}#chatgpt-mermaid-\_r_1d1\_ [data-look="neo"].node rect,#chatgpt-mermaid-\_r_1d1\_ [data-look="neo"].cluster rect,#chatgpt-mermaid-\_r_1d1\_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-\_r_1d1\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_1d1\_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-\_r_1d1\_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-\_r_1d1\_-gradient);stroke-width:1px;}#chatgpt-mermaid-\_r_1d1\_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_1d1\_ [data-look="neo"].node .neo-line path{stroke:rgb(31, 78, 148);filter:none;}#chatgpt-mermaid-\_r_1d1\_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-\_r_1d1\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_1d1\_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-\_r_1d1\_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-\_r_1d1\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_1d1\_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-\_r_1d1\_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-\_r_1d1\_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(153, 206, 255);}#chatgpt-mermaid-\_r_1d1\_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(153, 206, 255);}#chatgpt-mermaid-\_r_1d1\_ .node tspan[font-weight="normal"],#chatgpt-mermaid-\_r_1d1\_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-\_r_1d1\_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-width:1px;}#chatgpt-mermaid-\_r_1d1\_ .node rect,#chatgpt-mermaid-\_r_1d1\_ .node circle,#chatgpt-mermaid-\_r_1d1\_ .node ellipse,#chatgpt-mermaid-\_r_1d1\_ .node polygon,#chatgpt-mermaid-\_r_1d1\_ .node path{fill:rgb(0, 40, 77);stroke:rgba(255, 255, 255, 0.1);stroke-width:1px;}#chatgpt-mermaid-\_r_1d1\_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-\_r_1d1\_ .mermaid-decision .label-container{fill:rgb(0, 14, 26);stroke:rgb(26, 62, 95);stroke-dasharray:2,2;}#chatgpt-mermaid-\_r_1d1\_ .edgePaths .flowchart-link{stroke:rgb(175, 175, 175);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-\_r_1d1\_ .marker{fill:rgb(175, 175, 175);stroke:rgb(175, 175, 175);}#chatgpt-mermaid-\_r_1d1\_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Scene DirectorExtract Visual RequirementsResolve Character CanonResolve Appearance and OutfitResolve Accessories andObjectsResolve EnvironmentBuild ShotVisualContextValidate Missing AssetsCompile Structured PromptSelect ComfyUI WorkflowValidate Models andReferencesGenerate ImageTechnical QCVisual Continuity QCApproved?Commit Visual AssetTargeted CorrectionFilm AssemblyYesNo

There should be a manual review state between failed automated validation and expensive repeated generation when confidence is low.

# 10. Implementation roadmap: What we should actually do first

I don't recommend implementing all six subsystems simultaneously.

That would create another complicated system which is difficult to debug.

Here is the implementation sequence I would use.

# Novel Visualiser – VCE Implementation

1. Phase 0 — ComfyUI reliability

   First priority
   - Audit all 11 existing workflow JSON files.
   - Identify the working checkpoint and VAE.
   - Verify IP-Adapter and ControlNet node compatibility.
   - Eliminate silent mock-image fallback.
   - Create one verified production workflow.
   Deliverable: Reliable image generation from Python.
2. Phase 1 — Global Visual Style
   - Implement `VisualStyleProfile`.
   - Lock model and rendering configuration.
   - Generate a 10-image style benchmark.
   - Save the generation manifests.
   Deliverable: Stable illustration aesthetic.
3. Phase 2 — Visual Canon Database
   - Character identity.
   - Appearance versions.
   - Outfits.
   - Accessories.
   - Environments.
   - Reference image management.
   Deliverable: Complete visual asset hierarchy.
4. Phase 3 — Visual Context Resolver
   - Resolve all visual dependencies.
   - Implement temporal inheritance.
   - Detect missing references.
   - Add conflict detection.
   Deliverable: Structured `ShotVisualContext`.
5. Phase 4 — Prompt Compiler and Workflow Registry
   - Implement structured prompt blocks.
   - Create workflow contracts.
   - Add model and input validation.
   - Connect references to the correct workflow inputs.
   Deliverable: Fully parameterised ComfyUI generation.
6. Phase 5 — Visual Continuity QC
   - Character identity verification.
   - Outfit and accessory checking.
   - Environment verification.
   - Style consistency checks.
   - Correction and approval loop.
   Deliverable: Automated visual quality control.
7. Phase 6 — Full Chapter Benchmark

   Generate a chapter containing different characters, locations, outfits, objects, lighting conditions and camera angles.

   Deliverable: An illustrated chapter with measurable visual continuity.

# 11. The benchmark I want us to pass

Before calling the consistency engine complete, I want it to pass this experiment.

## Visual Consistency Benchmark

One character, multiple scenes

[Pin by Stephanie Schwartz on Drawing references | Illustration character design, Anime poses reference, Character art](https://images.openai.com/static-rsc-4/_ixUBnOkyjToDfKWZ1mN6SxiFqwkEOkJbxIIWnnI7lnNWxq7whIUC8bwfb9i6eGWXeBZSWl1433InSEgmukAyoftgoIoLyqqB_JEBtKofS1yq6xsUbK8DHqkGDNAtyCW4g7K3fGwUhzCDONI7NU6p4NGOHVWDJkQD921zOf_TTU?purpose=inline)

1\. Portrait

[Pinterest](https://images.openai.com/static-rsc-4/QOpyGFL05Jbnti759cZRw2wVWXMdI2nkBZ3YTGw2GCu2LGJw_geDmkFfstKJNx2An_YwRJxrFxUC6UoXcpCkvziwll4nUZEExH9HU2dvV3xU_-EZkRXAVsUFdfB69fWAE9ZJUST4LbzPWeL9ve54yMMrLjvwamaLaWSzFL5blQk?purpose=inline)

2\. Full body

[Anime Fight Scene Art: AI Action Illustration Maker](https://images.openai.com/static-rsc-4/QCtw270eG1J8en-ymepsewalXfALKEKm6hVOJLySX_DRzSOXZcoMo6RtqtRuzuZmqN3McCQRYQ23IaN2zN8QokqXwdpexZn6nJXod0OBU3YjkQilgeQycStLSRcnOM3JQi8nDevuUy41OhHowwRgJYhAQn6sifKlDaUni_10QXo?purpose=inline)

3\. Action pose

[STORY | TVアニメ「俺だけレベルアップな件」公式サイト](https://images.openai.com/static-rsc-4/q7mXsp-NVtF_qU62lj4IeHicdI-ms85JR7leWMU5Tc9-6UH0kM1n9zSoC61KT8OY3sS8U5dxCtBYShk6DhJys3EZLwrr47m_uOXIaCXQzl7RhqCry9uC-QgW2dv6bVlwQ4G2hXMJJUB7XAc0HFaaPPgz_6FUj5Ngu2BQsTdmS2M?purpose=inline)

4\. Close-up

Test conditions

| Test                  | Expected behaviour          |
| --------------------- | --------------------------- |
| 5 different poses     | Same identity               |
| 3 camera angles       | Same facial characteristics |
| 3 different locations | Same rendering style        |
| 3 outfits             | Correct outfit inheritance  |
| 2 lighting conditions | Recognisable identity       |
| 5 accessory shots     | Correct object design       |

Passing criterion

All required assets must be correct, with no silent changes to the canonical identity or global rendering style. We should measure pass rates and manual reviewer agreement across the benchmark.

Only after this works should we consider training custom character or style LoRAs. We should first establish how much consistency the existing checkpoint and reference-conditioning workflow can achieve.

# 12. What I need to inspect before modifying the code

There is one limitation in the material you shared: it is a comprehensive project analysis, not the actual Python source files or ComfyUI API workflow JSONs. So I have enough information to design the architecture, but not enough to verify the existing node mappings, database columns or implementation bugs.

I would like to work against these actual files:

| Priority | File                                                     |
| -------- | -------------------------------------------------------- |
| 1        | `app/orchestrator/resolver.py`                           |
| 2        | `app/services/prompt_compiler.py`                        |
| 3        | `app/services/comfyui_client.py`                         |
| 4        | `app/services/qc_service.py`                             |
| 5        | `app/database/models.py`                                 |
| 6        | `app/orchestrator/graph.py` and `state.py`               |
| 7        | `templates/comfyui/Workflows/` — all workflow JSON files |
| 8        | Relevant tests for Phases 0–4                            |

My recommendation is to start with Phase 0, not the database modifications.

We need one completely verified ComfyUI workflow, with a stable checkpoint, reference inputs and successful generation through the Python client. Otherwise, every later improvement will be built on an unreliable generation layer.

The larger architectural principle I want to preserve throughout this project is:

The story determines what should be illustrated. The Visual Canon determines what everything looks like. The Prompt Compiler determines how to describe it. ComfyUI determines how to render it. The Continuity Engine determines whether the result is acceptable.

That separation is what will allow Novel Visualiser to generate hundreds or thousands of illustrations without gradually losing its characters, environments, costumes and overall art direction.