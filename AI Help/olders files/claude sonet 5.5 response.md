Why consistency failed, and how to fix it
Your analysis describes a strong orchestration layer (DB, resolver, spoiler gate, QC, assembly), but none of that makes images consistent. Consistency comes from the generation layer and the assets feeding it, and that is the weakest part. The pipeline tracks what each character should look like in a database, but nothing forces the model to draw it that way.

Root causes
Problem	Why it happens
Character drift	You rely on prompt text plus IP-Adapter alone. Prompt tags never lock a face, and one IP-Adapter reference is weak, especially in a different pose or angle.
Clothing drift	Outfits are free-text tags that get merged, reordered, and diluted by the Resolver. SDXL reads token order and weight, so a different order gives a different outfit.
Location drift	Each shot is generated from scratch from text. Nothing anchors the room, so every shot invents a new one.
Item drift	Small objects are not reliably reproduced by text, and IP-Adapter treats them as style noise.
QC can't catch it	HSV histogram correlation and edge density check colour and blur, not identity. A different girl with the same palette passes.
Silent failures	If the Cloudflare tunnel dies, you fall back to the OpenCV mock canvas, so you may be "testing" fake images.
Checkpoint choice	waiNSFWIllustrious is NSFW-leaning and has strong style and anatomy biases that fight your prompts. A neutral Illustrious/NoobAI base is safer.
Fix 1: Build assets before generating shots
Create a one-time asset-creation stage per entity:

Character: generate a character sheet (front, side, back, 3/4, face close-up, 3–4 expressions) on a plain background. Pick the best 15–30 images.
Outfit: keep one canonical full-body image per outfit version.
Location: generate a master plate (wide establishing shot, no characters) plus 2–3 other angles.
Item: generate an item sheet (isolated on white, 2–3 angles).
Store these as visual_refs, and make them mandatory. The pipeline should refuse to generate a shot if a required entity has no approved reference.

Fix 2: Train a LoRA for each main character
This is the single biggest improvement.

Train a character LoRA on the sheet images with kohya_ss or OneTrainer. It runs on a Kaggle T4 in roughly 1–2 hours at 1024px, rank 16–32.
Include the outfit in the training tags so it binds to the character, and tag outfit variants differently.
At generation time, use the LoRA at 0.7–0.9 with a fixed trigger word, then add IP-Adapter at low weight (0.3–0.5) for reinforcement only.
For minor characters, use IP-Adapter Plus Face with 2–3 reference images and no LoRA.

Fix 3: Lock the prompt, don't merge it
Replace free-form merging with fixed, ordered blocks:

[quality] [style] [shot/camera] [CHAR_A_LOCK] [CHAR_A_OUTFIT_v1] [CHAR_A_STATE_DELTA]
[CHAR_B_LOCK] [ACTION] [LOCATION_LOCK] [LIGHTING]
CHAR_A_LOCK is a fixed string stored once and inserted verbatim, never rewritten by an LLM.
Use Danbooru-style tags (Illustrious was trained on them), not natural-language sentences.
Deltas such as torn sleeve, mud, or injury are appended after the locked block.
Use a deterministic seed per shot (hash(project, chapter, shot_id)) so reruns are reproducible.
Fix 4: Anchor environments
Pick one of these approaches:

Plate → depth ControlNet: run the location master plate through depth/lineart (your Union ControlNet supports this) and use it as structure for every shot in that location, at strength 0.5–0.7.
Background-first compositing: generate the background, then inpaint characters into it with a regional mask and the character LoRA. This is the most reliable method but more work.
IP-Adapter style on the plate: the weakest option, but it carries mood and palette.
Also limit variety: each scene gets one location, one lighting setup, and one time of day.

Fix 5: Items and artifacts
Keep them out of hero close-ups at first.
For key items, train a small LoRA or use inpainting with an IP-Adapter on the item sheet.
Add the item's exact tags to the locked block of whoever holds it.
Fix 6: Make QC verify identity
Replace the histogram gate with real checks:

WD14 tagger on every output, then compare against required tags (hair colour, eye colour, outfit pieces). Auto-reject and reseed on mismatch.
Face embedding similarity against the canonical references, using CLIP or DINOv2 image embeddings. An anime-specific face model such as ccip from the imgutils library works well for anime identity checking.
Face detailer pass: run Impact Pack's FaceDetailer with the character LoRA to fix small faces in wide shots.
Fail loudly: if ComfyUI is unreachable, raise an error. Never fall back to mock images in a real run.
Suggested workflow node order
Checkpoint → LoRA (char) → LoRA (style, optional)
→ CLIP encode (locked prompt)
→ IP-Adapter (low weight) + ControlNet (depth/pose from plate)
→ KSampler (fixed seed) → FaceDetailer → upscale → WD14 QC
Priority plan
Remove the silent mock fallback and add a ComfyUI health check.
Run a consistency test on one character: 12 images in different scenes with fixed settings, changing only the prompt action. This is your benchmark.
Make the character sheet and train the LoRA for Sunny, then repeat for Nephis.
Switch to locked prompt blocks with fixed seeds.
Add location master plates with depth ControlNet.
Add WD14 and embedding QC with auto-retry.
Handle items and minor characters last.
Don't regenerate the whole chapter until the benchmark in step 2 passes. Fix one variable at a time.

To help you more specifically, I need:
3–4 failed images, or a description of how they fail (face changes, clothes change, background changes?)
One compiled prompt from prompt_compiler.py
The ComfyUI workflow JSON you actually use for a shot (WF-00x)
Whether you can train LoRAs on Kaggle (session time and GPU quota)
With those I can rewrite the Resolver and Prompt Compiler for locked blocks, design the LoRA training notebook, or build the WD14 QC gate.