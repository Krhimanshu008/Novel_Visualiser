You’re trying to solve the hardest part of this project: making every shot look like it belongs to the same world—not just generating attractive individual images.

Based on the analysis you pasted, I would not call the project production-ready yet. It appears to have substantial orchestration and video assembly, but those do not prove visual consistency.

My main diagnosis: your pipeline describes continuity in the database and prompts, but it may not enforce that continuity strongly enough during image generation.

I haven’t inspected the actual repository or images, so the points below are hypotheses to verify—not confirmed code findings.

1. Why consistency is probably failing
A. Database consistency is not image consistency
Your resolver reportedly produces:

Project style
+ Canonical identity
+ Arc version
+ Chapter changes
+ Scene action
That is useful for deciding what should appear. But an image model does not treat a prompt as a strict specification.

For example:

“Sunny, thin young man, pale skin, black hair, worn dark clothes”

can generate many different faces, outfits, and body proportions.

You need two separate systems:

System	Responsibility
Canon/state resolver	Decide which character, clothes, location, injuries, and items are correct
Visual conditioning system	Make the generated pixels resemble the approved designs
Your analysis demonstrates the first more clearly than the second.

B. One reference image cannot lock everything
A character reference may help with appearance, but it does not automatically preserve:

Face across extreme angles.
Outfit construction and back details.
Height relative to other characters.
A weapon’s exact geometry.
Room layout.
Which hand holds an item.
Likewise, applying an IP-Adapter to a whole image can introduce unwanted composition, clothing, or background influence.

Different consistency problems require different controls.

C. Your reported QC metrics do not measure identity reliably
These checks have limited purposes:

Check	Useful for	Not reliable for
Blur/exposure	Technical defects	Character identity
HSV histogram	Broad color similarity	Same face, garment, or room
Whole-image SSIM	Near-identical framing	Different shots of the same subject
Edge density	Rough image structure	Correct artifact design
Two different characters can share a color histogram. Two correct views of one character can have very low SSIM.

Also, a “semantic checklist” only verifies an image if something actually examines the generated pixels. Checking the requested prompt or metadata does not establish that the requested feature appeared.

D. Silent mock-image fallback could invalidate the entire result
The report mentions an OpenCV fallback when ComfyUI is unavailable. Verify this first.

Production must never silently substitute a mock render.

if not comfyui_available:
    raise GenerationUnavailable(
        "ComfyUI unavailable. Production rendering stopped."
    )
Keep mock rendering behind an explicit development setting, and visibly watermark its outputs.

A completed MP4 proves the assembly pipeline ran—not that the image-generation pipeline worked correctly.

2. Replace “generate every shot” with “build assets, then compose shots”
Your production architecture should become:

Story text
    ↓
Resolve canon and current scene state
    ↓
Select approved visual assets
    ↓
Plan layout, camera, pose, and visibility
    ↓
Generate or reuse environment plate
    ↓
Render characters with isolated conditioning
    ↓
Insert or repair important props
    ↓
Validate visible continuity
    ↓
Approve shot
The important change is this:

Do not ask the model to reinvent the character, costume, location, and weapon in every shot.

3. Create separate visual asset types
Avoid treating every visual change as one combined character prompt.

Character identity
Lock:

Face shape and distinctive features.
Hair silhouette and color.
Eye color.
Body proportions.
Apparent age.
Relative height.
Create separately approved images:

character/sunny/base_v001/
    face_front.png
    face_three_quarter.png
    face_profile.png
    full_body_front.png
    full_body_back.png
    manifest.json
Generate and approve these views individually. A generated contact sheet can contain inconsistent faces internally; it is not automatically a reliable reference.

Outfit
Give clothing its own identity:

outfit/sunny_ch01_v001/
    front.png
    back.png
    details.png
    manifest.json
Specify actual construction:

Collar shape.
Sleeve length.
Fasteners.
Belt placement.
Fabric and color.
Tear or stain locations.
“Worn black clothes” is not a stable costume design.

Location
A location needs more than descriptive tags:

location/police_room_v001/
    layout.json
    floor_plan.png
    camera_A_plate.png
    camera_B_plate.png
    camera_C_plate.png
Record landmarks:

Door wall.
Window wall.
Desk position.
Main light source.
Character blocking.
For rooms used repeatedly, a simple Blender blockout can provide more dependable perspective and geometry than repeatedly prompting a room from scratch.

Items and artifacts
Give important objects independent assets:

prop/artifact_001_v001/
    front.png
    side.png
    detail.png
    manifest.json
Specify silhouette, dimensions, material, symbols, and how it is carried.

For exact recurring symbols or complex weapon geometry, consider compositing a designed asset or rendering a simple 3D model. Text-to-image alone is often unreliable for exact repetition.

4. Use the right generation strategy for each problem
With your reported SDXL/IP-Adapter/ControlNet stack, I would start here:

Problem	Initial approach
Face drift	Approved face reference, face-focused conditioning, targeted face repair
Body/outfit drift	Full-body reference plus pose control and masked outfit repair
Pose drift	OpenPose or another compatible pose-control workflow
Room geometry drift	Reused background plate or blockout-derived depth/edges
Important prop changes shape	Isolated prop insertion or masked inpainting
Two characters exchange features	Separate masked passes or validated regional conditioning
Minor variation from an approved frame	Image-to-image or localized editing rather than fresh text-to-image
Important: validate compatibility with your exact checkpoint and installed ComfyUI nodes. Possessing model files does not prove that the workflow loads them or applies them correctly.

For two-character scenes, start with:

Generate the environment.
Insert character A in its region.
Insert character B in its region.
Repair contact areas, hands, shadows, and overlapping objects.
This costs more passes, but is usually easier to debug than a single prompt containing multiple identities and accessories.

LoRA training may help recurring designs later. Do not train it on inconsistent generated references, and do not make it your first fix before verifying the existing conditioning workflow.

5. Make every shot carry an explicit continuity contract
Before compiling a prompt, resolve a structured shot specification:

{
  "shot_id": "CH01_S004",
  "location_version": "police_room_v001",
  "camera_setup": "camera_B",
  "characters": [
    {
      "entity_id": "sunny",
      "identity_version": "sunny_base_v001",
      "outfit_version": "sunny_ch01_v001",
      "position": "chair_left",
      "pose": "seated",
      "state": {
        "wet": false,
        "injuries": []
      }
    }
  ],
  "props": [
    {
      "asset_version": "cup_v001",
      "holder": "sunny",
      "hand": "right"
    }
  ],
  "required_visible": [
    "sunny_face",
    "outfit_collar",
    "cup"
  ]
}
Then compile both:

The text prompt.
The actual conditioning inputs: reference files, masks, pose maps, depth maps, adapter settings, and workflow selection.
A resolved reference that never reaches a ComfyUI conditioning node provides no visual enforcement.

6. Fix validation and retries
Use separate gates:

Technical: valid image, expected size, no obvious corruption.
Identity: compare relevant character crops against approved references.
Outfit/prop: inspect required visible details.
Environment: verify landmarks and layout.
Continuity: check state against the previous shot.
Embedding similarity and vision-language checks can assist, but must be calibrated against human-reviewed examples—especially for stylized anime faces.

Use three outcomes:

PASS
FAIL
UNVERIFIABLE
A hidden eye color or off-screen weapon cannot be verified.

Most importantly, retry locally:

Wrong face       → repair face region
Wrong sleeve     → repair clothing region
Missing artifact → insert artifact
Wrong room       → replace/rebuild environment
Do not regenerate an otherwise correct image just to fix one small defect.

7. The best next milestone: a small consistency benchmark
Pause full-chapter generation. Build this test first:

One character
Generate six shots:

Front portrait.
Three-quarter portrait.
Profile.
Full body.
Seated.
Different lighting.
One location
Generate three camera views with the same fixed layout.

One prop
Show it:

Alone.
Held by the character.
Resting in the location.
One interaction
Generate two shots containing two named characters without identity or outfit swapping.

Review these manually and record exactly what changes. Use those examples to evaluate workflow changes rather than judging isolated attractive outputs.

Also save a render manifest containing:

Checkpoint and model hashes
Workflow version
Reference hashes
Resolved asset versions
Seed
Sampler / scheduler / steps / CFG
Resolution
Adapter and ControlNet settings
ComfyUI/custom-node versions
QC results
A fixed seed alone does not lock identity, and exact reproducibility may still vary across software or hardware changes.

What I would prioritize in your project
Disable silent production fallbacks.
Prove that reference images actually reach the intended conditioning nodes.
Approve one identity, one outfit, one room, and one prop.
Pass the small benchmark above.
Add masked repair and continuity-aware QC.
Only then resume chapter-scale production.
For a concrete code-level diagnosis, share these next:

One actual exported ComfyUI API workflow JSON used for production.
prompt_compiler.py and the workflow-submission portion of comfyui_client.py.
One approved reference plus 3–5 generated shots showing the drift.
The resolved shot payload and generation settings for those images.
Those will let us determine whether your primary failure is missing conditioning, unsuitable references, workflow wiring, competing controls, or unrealistic expectations from single-pass generation.