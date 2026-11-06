"""
Novel Visualiser — Kaggle ComfyUI Production Runner (Phase 3 & 4 / VCE).

Production-hardened ComfyUI Orchestrator implementing the 6 + 2 Specialized Workflows:
  Stage A (Asset Foundry - Pre-Creation):
    1. WF-FOUNDRY-CHAR: Character Turnaround & 4-Expression Matrix
    2. WF-FOUNDRY-LOC: Location Master Plate & Depth/Lineart Extraction
    3. WF-FOUNDRY-PROP: Isolated Weapon & Item Sheet

  Stage B (Shot Production):
    4. WF-SHOT-SOLO: Single-Character Workhorse (LoRA + IP-Adapter Face + ControlNet Depth + FaceDetailer)
    5. WF-SHOT-MULTI: Multi-Subject Regional Isolation (Zero Attribute Bleeding via SolidMask)
    6. WF-SHOT-ENV: Pure Scenery / Establishing Vista

  Stage C (Quality Remediation):
    7. WF-EDIT-INPAINT: Targeted Regional Inpaint & Prop Fix (DifferentialDiffusion)

  Stage D (Final Mastering):
    8. WF-MASTER-UPSCALE: 4K AnimeSharp Tiled VAE Refine Pass

Key Kaggle & Colab Production Hardening:
1. Subprocess Pipe Deadlock Fixed: Redirects main.py stdout/stderr directly to a disk log file.
2. Hierarchical Path Mismatch Fixed: Auto-symlinks both relative nested paths and flat filenames.
3. Native Regional Conditioning: SolidMask + MaskComposite + ConditioningSetMask.
4. Offline Structural Depth Prior: Computes depth plates locally with OpenCV.
5. VRAM Fragmentation Prevention: Tiled VAE encode/decode and automatic cache flushing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("kaggle_production_runner")

WORKSPACE = "/kaggle/working" if os.path.exists("/kaggle/working") else ("/content" if os.path.exists("/content") else str(Path.cwd()))
COMFY_DIR = os.path.join(WORKSPACE, "ComfyUI")
COMFY_URL = "http://127.0.0.1:8188"
LOG_FILE = os.path.join(WORKSPACE, "comfyui_server.log")

CUSTOM_NODES = {
    "ComfyUI_IPAdapter_plus": "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git",
    "ComfyUI-Impact-Pack": "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git",
    "ComfyUI-Impact-Subpack": "https://github.com/ltdrdata/ComfyUI-Impact-Subpack.git",
}

CATEGORY_MAP = {
    "checkpoints": "checkpoints",
    "clip_vision": "clip_vision",
    "controlnet": "controlnet",
    "ipadapter": "ipadapter",
    "ultralytics": "ultralytics",
    "upscale_models": "upscale_models",
    "vae": "vae",
    "loras": "loras",
}


def run_cmd(cmd: str, cwd: Optional[str] = None) -> None:
    """Execute shell command with stream echo and non-interactive check."""
    logger.info("▶ Running: %s", cmd)
    subprocess.run(cmd, shell=True, cwd=cwd, check=True)


def safe_symlink(src: Path, dst: Path) -> None:
    """Create symlink safely replacing stale links or existing target."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            dst.unlink()
        except OSError:
            pass
    try:
        os.symlink(src, dst)
    except OSError:
        # Fallback to copy if symlinks unprivileged on Windows
        shutil.copyfile(src, dst)


def install_comfyui_and_nodes() -> None:
    """Cell 1: Install ComfyUI & Required Custom Nodes (Fast, Non-Interactive)."""
    comfy_path = Path(COMFY_DIR)
    if not comfy_path.exists():
        run_cmd(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git {COMFY_DIR}")

    nodes_dir = comfy_path / "custom_nodes"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    for name, url in CUSTOM_NODES.items():
        target = nodes_dir / name
        if not target.exists():
            run_cmd(f"git clone --depth 1 {url} {target}")

    # Install requirements without interactive prompts
    run_cmd(f"{sys.executable} -m pip install -q --no-input -r requirements.txt", cwd=COMFY_DIR)
    run_cmd(
        f"{sys.executable} -m pip install -q --no-input "
        "ultralytics opencv-python-headless GitPython piexif segment-anything scikit-image dill matplotlib websocket-client httpx"
    )
    logger.info("✅ Cell 1 Complete: ComfyUI and Custom Nodes installed.")


def link_model_library() -> Path:
    """Cell 2: Auto-Discover and Link nv-comfyui-model-library Dataset."""
    search_roots = [Path("/kaggle/input"), Path("/content"), Path(WORKSPACE), Path.cwd()]
    dataset_root = None

    for root in search_roots:
        if not root.exists():
            continue
        candidates = [root / "nv-comfyui-model-library"] + list(root.glob("**/nv-comfyui-model-library"))
        for c in candidates:
            if c.exists() and (c / "checkpoints").exists():
                dataset_root = c
                break
        if dataset_root:
            break

    if dataset_root is None:
        logger.warning("Could not auto-locate 'nv-comfyui-model-library'. Creating local models fallback directory.")
        dataset_root = Path(COMFY_DIR) / "models"
        dataset_root.mkdir(parents=True, exist_ok=True)
        return dataset_root

    logger.info("📂 Found Dataset at: %s", dataset_root)
    comfy_models_root = Path(COMFY_DIR) / "models"

    linked_files = []
    for ds_cat, comfy_cat in CATEGORY_MAP.items():
        src_dir = dataset_root / ds_cat
        if not src_dir.exists():
            continue
        dst_dir = comfy_models_root / comfy_cat
        dst_dir.mkdir(parents=True, exist_ok=True)

        for file_path in src_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(src_dir)
                # 1. Preserving relative hierarchy
                safe_symlink(file_path, dst_dir / rel_path)
                # 2. Flattened filename at root
                safe_symlink(file_path, dst_dir / file_path.name)
                # 3. For ultralytics bbox
                if ds_cat == "ultralytics" and "bbox" in str(rel_path):
                    safe_symlink(file_path, dst_dir / "bbox" / file_path.name)
                linked_files.append((comfy_cat, file_path.name))

    logger.info("✅ Linked %d models from dataset into %s", len(linked_files), comfy_models_root)
    return dataset_root


def is_server_alive(server_url: str = COMFY_URL) -> bool:
    """Check if ComfyUI /system_stats responds with HTTP 200."""
    try:
        with urllib.request.urlopen(f"{server_url}/system_stats", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_comfyui_server(server_url: str = COMFY_URL) -> None:
    """Cell 3: Non-blocking background ComfyUI launcher preventing OS pipe buffer freeze."""
    if is_server_alive(server_url):
        logger.info("🚀 ComfyUI Server is already running at %s", server_url)
        return

    log_handle = open(LOG_FILE, "w", encoding="utf-8")
    subprocess.Popen(
        [
            sys.executable,
            "main.py",
            "--listen", "0.0.0.0",
            "--port", "8188",
            "--fp16-vae",
            "--disable-xformers",
        ],
        cwd=COMFY_DIR,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        close_fds=(sys.platform != "win32"),
    )

    logger.info("⏳ Starting ComfyUI server in background (logging to %s)...", LOG_FILE)
    for i in range(90):
        if is_server_alive(server_url):
            logger.info("🚀 ComfyUI Server is ONLINE at %s (took %ds)", server_url, i * 2)
            return
        time.sleep(2)

    raise RuntimeError(f"ComfyUI server failed to start within 180s. Check {LOG_FILE}")


class NovelVisualiserEngine:
    """Production-hardened ComfyUI Orchestrator implementing the 6 + 2 Specialized Workflows.
    
    Bound directly to exact nv-comfyui-model-library filenames and parameters.
    """

    CKPT_ILLUSTRIOUS = "waiNSFWIllustrious_v110.safetensors"
    CKPT_TURBO = "DreamShaperXL_Turbo_v2_1.safetensors"
    VAE_SDXL = "sdxl_vae_fp16_fix.safetensors"
    CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    CN_UNION = "xinsir-controlnet-union-sdxl-1.0-promax.safetensors"
    IPA_FACE = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
    IPA_PLUS = "ip-adapter-plus_sdxl_vit-h.safetensors"
    YOLO_FACE = "bbox/face_yolov8n.pt"
    UPSCALER_ANIME = "4x-AnimeSharp.pth"

    DEFAULT_NEG = (
        "lowres, bad anatomy, bad hands, text, error, missing fingers, "
        "extra digit, fewer digits, cropped, worst quality, low quality, "
        "normal quality, jpeg artifacts, signature, watermark, username, blurry, deformed eyes"
    )

    def __init__(self, server_url: str = COMFY_URL, workspace: str = WORKSPACE):
        self.server_url = server_url.rstrip("/")
        self.client_id = str(uuid.uuid4())
        self.workspace = Path(workspace)
        self.comfy_input = Path(COMFY_DIR) / "input"
        self.comfy_output = Path(COMFY_DIR) / "output"
        self.comfy_input.mkdir(parents=True, exist_ok=True)
        self.comfy_output.mkdir(parents=True, exist_ok=True)

    def free_vram(self, unload_models: bool = False) -> None:
        """Flushes VRAM cache between heavy multi-node workflows."""
        try:
            payload = json.dumps({"unload_models": unload_models, "free_memory": True}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.server_url}/free",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            logger.debug("ComfyUI /free note: %s", exc)

    def stage_input_image(self, image_path: str | Path) -> str:
        """Copies an external image into ComfyUI/input and returns its filename."""
        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(f"Input image not found: {src}")
        dst_name = f"{uuid.uuid4().hex[:8]}_{src.name}"
        shutil.copyfile(src, self.comfy_input / dst_name)
        return dst_name

    def execute_workflow(self, wf_id: str, prompt_graph: dict[str, Any], timeout_sec: int = 600) -> list[Path]:
        """Submits workflow graph to ComfyUI and polls smoothly until completion."""
        payload = json.dumps({"prompt": prompt_graph, "client_id": self.client_id}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.server_url}/prompt",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                prompt_id = json.loads(resp.read().decode("utf-8"))["prompt_id"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8")
            raise RuntimeError(f"[{wf_id}] ComfyUI rejected workflow graph:\n{err_body}")

        logger.info("⚙️ [%s] Queued (prompt_id=%s) — generating...", wf_id, prompt_id[:8])
        start_t = time.time()
        while time.time() - start_t < timeout_sec:
            with urllib.request.urlopen(f"{self.server_url}/history/{prompt_id}", timeout=10) as resp:
                history = json.loads(resp.read().decode("utf-8"))
            if prompt_id in history:
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"[{wf_id}] Execution error: {status}")
                outputs = history[prompt_id].get("outputs", {})
                saved_files = []
                for node_id, node_out in outputs.items():
                    for img_info in node_out.get("images", []):
                        sub = img_info.get("subfolder", "")
                        fname = img_info["filename"]
                        full_path = self.comfy_output / sub / fname if sub else self.comfy_output / fname
                        saved_files.append(full_path)
                elapsed = time.time() - start_t
                logger.info("✅ [%s] Completed in %.1fs -> %s", wf_id, elapsed, [p.name for p in saved_files])
                return saved_files
            time.sleep(1.0)

        raise TimeoutError(f"[{wf_id}] Timed out after {timeout_sec}s.")

    # =========================================================================
    # STAGE A: ASSET FOUNDRY WORKFLOWS (PRE-CREATION)
    # =========================================================================

    def wf_foundry_char(
        self,
        char_id: str,
        identity_tags: str,
        mode: str = "turnaround",
        seed: int = 42,
    ) -> Path:
        """01. WF-FOUNDRY-CHAR: Character Turnaround & Expression Sheet."""
        if mode == "turnaround":
            framing = (
                "masterpiece, best quality, character reference sheet, orthographic 3-angle turnaround, "
                "front view, three-quarter view, profile side view, full body, standing straight, "
                "simple background, solid neutral grey studio background #2b2b2b, flat even studio lighting, no shadows"
            )
        else:
            framing = (
                "masterpiece, best quality, expression sheet, 4-panel matrix, close-up headshots, "
                "neutral expression, intense combat expression, shocked wary expression, cunning smirk, "
                "simple background, solid neutral grey studio background #2b2b2b, flat studio lighting"
            )

        pos_prompt = f"{framing}, {identity_tags}"
        neg_prompt = f"{self.DEFAULT_NEG}, complex background, scenery, dramatic lighting, lens flare, multiple outfits"

        graph = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.CKPT_ILLUSTRIOUS}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": self.VAE_SDXL}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": pos_prompt, "clip": ["1", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_prompt, "clip": ["1", 1]}},
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed, "steps": 28, "cfg": 6.5,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
                    "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["3", 0],
                },
            },
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
            "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"WF01_CHAR_{char_id}_{mode}"}},
        }
        return self.execute_workflow("WF-FOUNDRY-CHAR", graph)[0]

    def wf_foundry_loc(
        self,
        project_id: str,
        arc_id: str,
        loc_id: str,
        landscape_tags: str,
        seed: int = 100,
    ) -> dict[str, Path]:
        """02. WF-FOUNDRY-LOC: Location Master Plate & Depth/Lineart Guides."""
        pos_prompt = f"masterpiece, best quality, anime background art, establishing wide shot, no humans, {landscape_tags}"
        neg_prompt = f"1boy, 1girl, character, person, crowd, silhouette, human, figure, {self.DEFAULT_NEG}"

        graph = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.CKPT_ILLUSTRIOUS}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": self.VAE_SDXL}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {"width": 1216, "height": 832, "batch_size": 1}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": pos_prompt, "clip": ["1", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_prompt, "clip": ["1", 1]}},
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed, "steps": 28, "cfg": 6.5,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
                    "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["3", 0],
                },
            },
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
            "8": {"class_type": "Canny", "inputs": {"image": ["7", 0], "low_threshold": 0.15, "high_threshold": 0.35}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"WF02_LOC_{loc_id}_rgb"}},
            "10": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": f"WF02_LOC_{loc_id}_lineart"}},
        }
        outputs = self.execute_workflow("WF-FOUNDRY-LOC", graph)
        rgb_src, lineart_src = outputs[0], outputs[1]

        plate_dir = self.workspace / f"data/projects/{project_id}/arcs/{arc_id}/location_plates"
        plate_dir.mkdir(parents=True, exist_ok=True)

        master_path = plate_dir / f"{loc_id}_master.png"
        lineart_path = plate_dir / f"{loc_id}_lineart.png"
        depth_path = plate_dir / f"{loc_id}_depth.png"

        shutil.copyfile(rgb_src, master_path)
        shutil.copyfile(lineart_src, lineart_path)

        # Offline structural depth pass with OpenCV (Zero download hang)
        img_bgr = cv2.imread(str(master_path))
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur_near = cv2.GaussianBlur(gray, (9, 9), 0).astype(np.float32) / 255.0
        h, w = gray.shape
        v_grad = np.linspace(0.2, 1.0, h, dtype=np.float32)[:, None] * np.ones((1, w), dtype=np.float32)
        depth_map = np.clip((0.55 * v_grad + 0.45 * (1.0 - blur_near)) * 255.0, 0, 255).astype(np.uint8)
        cv2.imwrite(str(depth_path), cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR))

        logger.info("🏛️ [WF-FOUNDRY-LOC] Saved Master, Depth & Lineart plates to: %s", plate_dir)
        return {"master": master_path, "depth": depth_path, "lineart": lineart_path}

    def wf_foundry_prop(
        self,
        prop_id: str,
        item_tags: str,
        scale_cm: float = 45.0,
        seed: int = 200,
    ) -> dict[str, Path]:
        """03. WF-FOUNDRY-PROP: Isolated Item & Weapon Sheet."""
        pos_prompt = (
            f"masterpiece, best quality, isolated weapon prop sheet, {item_tags}, "
            "no humans, centered object, orthographic view, simple solid dark grey studio background #2b2b2b, clean sharp lineart"
        )
        neg_prompt = f"1boy, 1girl, hand, fingers, holding, character, cluttered background, {self.DEFAULT_NEG}"

        graph = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.CKPT_ILLUSTRIOUS}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": self.VAE_SDXL}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": pos_prompt, "clip": ["1", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_prompt, "clip": ["1", 1]}},
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed, "steps": 26, "cfg": 6.5,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
                    "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["3", 0],
                },
            },
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
            "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"WF03_PROP_{prop_id}"}},
        }
        img_path = self.execute_workflow("WF-FOUNDRY-PROP", graph)[0]
        meta_path = img_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"prop_id": prop_id, "scale_cm": scale_cm, "tags": item_tags}, f, indent=2)
        return {"image": img_path, "metadata": meta_path}

    # =========================================================================
    # STAGE B: CHAPTER SHOT PRODUCTION WORKFLOWS
    # =========================================================================

    def wf_shot_solo(
        self,
        shot_id: str,
        prompt_8block: str,
        face_ref_path: Optional[str] = None,
        loc_depth_path: Optional[str] = None,
        lora_name: Optional[str] = None,
        lora_weight: float = 0.75,
        cn_depth_weight: float = 0.50,
        ipa_face_weight: float = 0.35,
        seed: int = 300,
    ) -> Path:
        """04. WF-SHOT-SOLO: The Core Production Workhorse (~70% of shots)."""
        graph: dict[str, Any] = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.CKPT_ILLUSTRIOUS}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": self.VAE_SDXL}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {"width": 1216, "height": 832, "batch_size": 1}},
        }

        model_ref = ["1", 0]
        clip_ref = ["1", 1]
        if lora_name:
            graph["1b"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": model_ref, "clip": clip_ref,
                    "lora_name": lora_name, "strength_model": lora_weight, "strength_clip": lora_weight,
                },
            }
            model_ref = ["1b", 0]
            clip_ref = ["1b", 1]

        graph["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": f"masterpiece, best quality, {prompt_8block}", "clip": clip_ref}}
        graph["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": self.DEFAULT_NEG, "clip": clip_ref}}

        # IP-Adapter Face
        if face_ref_path:
            face_inp = self.stage_input_image(face_ref_path)
            graph.update({
                "6": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": self.CLIP_VISION}},
                "7": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": self.IPA_FACE}},
                "8": {"class_type": "LoadImage", "inputs": {"image": face_inp}},
                "9": {
                    "class_type": "IPAdapterAdvanced",
                    "inputs": {
                        "model": model_ref, "ipadapter": ["7", 0], "image": ["8", 0],
                        "weight": ipa_face_weight, "weight_type": "linear",
                        "combine_embeds": "concat", "start_at": 0.0, "end_at": 0.85,
                        "embeds_scaling": "V only", "clip_vision": ["6", 0],
                    },
                },
            })
            model_ref = ["9", 0]

        pos_ref = ["4", 0]
        neg_ref = ["5", 0]

        # ControlNet Depth
        if loc_depth_path:
            depth_inp = self.stage_input_image(loc_depth_path)
            graph.update({
                "10": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": self.CN_UNION}},
                "11": {"class_type": "LoadImage", "inputs": {"image": depth_inp}},
                "12": {
                    "class_type": "ControlNetApplyAdvanced",
                    "inputs": {
                        "positive": pos_ref, "negative": neg_ref,
                        "control_net": ["10", 0], "image": ["11", 0],
                        "strength": cn_depth_weight, "start_percent": 0.0, "end_percent": 0.75, "vae": ["2", 0],
                    },
                },
            })
            pos_ref = ["12", 0]
            neg_ref = ["12", 1]

        graph.update({
            "13": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed, "steps": 28, "cfg": 6.5,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
                    "model": model_ref, "positive": pos_ref, "negative": neg_ref, "latent_image": ["3", 0],
                },
            },
            "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["2", 0]}},
            "15": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": self.YOLO_FACE}},
            "16": {
                "class_type": "FaceDetailer",
                "inputs": {
                    "image": ["14", 0], "model": model_ref, "clip": clip_ref, "vae": ["2", 0],
                    "guide_size": 512, "guide_size_for": True, "max_size": 1024,
                    "seed": seed + 1, "steps": 16, "cfg": 6.0,
                    "sampler_name": "euler_ancestral", "scheduler": "normal",
                    "positive": ["4", 0], "negative": ["5", 0],
                    "denoise": 0.40, "feather": 5, "noise_mask": True, "force_inpaint": True,
                    "bbox_threshold": 0.5, "bbox_dilation": 10, "bbox_crop_factor": 3.0,
                    "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93,
                    "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7,
                    "sam_mask_hint_use_negative": "False", "drop_size": 10,
                    "bbox_detector": ["15", 0], "wildcard": "", "cycle": 1,
                },
            },
            "17": {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": f"WF04_SOLO_{shot_id}"}},
        })
        return self.execute_workflow("WF-SHOT-SOLO", graph)[0]

    def wf_shot_multi(
        self,
        shot_id: str,
        bg_prompt: str,
        subj1_prompt: str,
        subj2_prompt: str,
        subj1_face_path: str,
        subj2_face_path: str,
        loc_depth_path: Optional[str] = None,
        seed: int = 400,
    ) -> Path:
        """05. WF-SHOT-MULTI: Multi-Subject Regional Isolation (Zero Attribute Bleeding)."""
        f1_inp = self.stage_input_image(subj1_face_path)
        f2_inp = self.stage_input_image(subj2_face_path)

        graph: dict[str, Any] = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.CKPT_ILLUSTRIOUS}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": self.VAE_SDXL}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {"width": 1216, "height": 832, "batch_size": 1}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": f"masterpiece, best quality, wide two-shot, {bg_prompt}", "clip": ["1", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": f"masterpiece, best quality, {subj1_prompt}", "clip": ["1", 1]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": f"masterpiece, best quality, {subj2_prompt}", "clip": ["1", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": self.DEFAULT_NEG, "clip": ["1", 1]}},

            "8":  {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": 1216, "height": 832}},
            "9":  {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": 547, "height": 666}},
            "10": {"class_type": "MaskComposite", "inputs": {"destination": ["8", 0], "source": ["9", 0], "x": 0, "y": 83, "operation": "add"}},
            "11": {"class_type": "MaskComposite", "inputs": {"destination": ["8", 0], "source": ["9", 0], "x": 669, "y": 83, "operation": "add"}},

            "12": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["5", 0], "mask": ["10", 0], "strength": 1.0, "set_cond_area": "default"}},
            "13": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["6", 0], "mask": ["11", 0], "strength": 1.0, "set_cond_area": "default"}},
            "14": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["4", 0], "conditioning_2": ["12", 0]}},
            "15": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["14", 0], "conditioning_2": ["13", 0]}},

            "16": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": self.CLIP_VISION}},
            "17": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": self.IPA_FACE}},
            "18": {"class_type": "LoadImage", "inputs": {"image": f1_inp}},
            "19": {"class_type": "LoadImage", "inputs": {"image": f2_inp}},
            "20": {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": ["1", 0], "ipadapter": ["17", 0], "image": ["18", 0],
                    "weight": 0.35, "weight_type": "linear", "combine_embeds": "concat",
                    "start_at": 0.0, "end_at": 0.85, "embeds_scaling": "V only",
                    "clip_vision": ["16", 0], "attn_mask": ["10", 0],
                },
            },
            "21": {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": ["20", 0], "ipadapter": ["17", 0], "image": ["19", 0],
                    "weight": 0.35, "weight_type": "linear", "combine_embeds": "concat",
                    "start_at": 0.0, "end_at": 0.85, "embeds_scaling": "V only",
                    "clip_vision": ["16", 0], "attn_mask": ["11", 0],
                },
            },
        }

        pos_cond = ["15", 0]
        neg_cond = ["7", 0]

        if loc_depth_path:
            depth_inp = self.stage_input_image(loc_depth_path)
            graph.update({
                "22": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": self.CN_UNION}},
                "23": {"class_type": "LoadImage", "inputs": {"image": depth_inp}},
                "24": {
                    "class_type": "ControlNetApplyAdvanced",
                    "inputs": {
                        "positive": pos_cond, "negative": neg_cond,
                        "control_net": ["22", 0], "image": ["23", 0],
                        "strength": 0.45, "start_percent": 0.0, "end_percent": 0.70, "vae": ["2", 0],
                    },
                },
            })
            pos_cond = ["24", 0]
            neg_cond = ["24", 1]

        graph.update({
            "25": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed, "steps": 28, "cfg": 6.5,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
                    "model": ["21", 0], "positive": pos_cond, "negative": neg_cond, "latent_image": ["3", 0],
                },
            },
            "26": {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["2", 0]}},
            "27": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": self.YOLO_FACE}},
            "28": {
                "class_type": "FaceDetailer",
                "inputs": {
                    "image": ["26", 0], "model": ["21", 0], "clip": ["1", 1], "vae": ["2", 0],
                    "guide_size": 512, "guide_size_for": True, "max_size": 1024,
                    "seed": seed + 1, "steps": 15, "cfg": 6.0,
                    "sampler_name": "euler_ancestral", "scheduler": "normal",
                    "positive": ["15", 0], "negative": ["7", 0],
                    "denoise": 0.38, "feather": 5, "noise_mask": True, "force_inpaint": True,
                    "bbox_threshold": 0.5, "bbox_dilation": 10, "bbox_crop_factor": 3.0,
                    "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93,
                    "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7,
                    "sam_mask_hint_use_negative": "False", "drop_size": 10,
                    "bbox_detector": ["27", 0], "wildcard": "", "cycle": 1,
                },
            },
            "29": {"class_type": "SaveImage", "inputs": {"images": ["28", 0], "filename_prefix": f"WF05_MULTI_{shot_id}"}},
        })
        return self.execute_workflow("WF-SHOT-MULTI", graph)[0]

    def wf_shot_env(
        self,
        shot_id: str,
        env_prompt: str,
        loc_depth_path: Optional[str] = None,
        resolution: tuple[int, int] = (1344, 768),
        seed: int = 500,
    ) -> Path:
        """06. WF-SHOT-ENV: Pure Scenery / Establishing / Vista (~10% of shots)."""
        w, h = resolution
        pos_prompt = f"masterpiece, best quality, establishing cinematic vista, no humans, {env_prompt}"
        neg_prompt = f"1boy, 1girl, character, person, crowd, {self.DEFAULT_NEG}"

        graph: dict[str, Any] = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.CKPT_ILLUSTRIOUS}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": self.VAE_SDXL}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": pos_prompt, "clip": ["1", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_prompt, "clip": ["1", 1]}},
        }

        pos_ref, neg_ref = ["4", 0], ["5", 0]
        if loc_depth_path:
            depth_inp = self.stage_input_image(loc_depth_path)
            graph.update({
                "6": {"class_type": "ControlNetLoader", "inputs": {"control_net_name": self.CN_UNION}},
                "7": {"class_type": "LoadImage", "inputs": {"image": depth_inp}},
                "8": {
                    "class_type": "ControlNetApplyAdvanced",
                    "inputs": {
                        "positive": ["4", 0], "negative": ["5", 0],
                        "control_net": ["6", 0], "image": ["7", 0],
                        "strength": 0.45, "start_percent": 0.0, "end_percent": 0.70, "vae": ["2", 0],
                    },
                },
            })
            pos_ref, neg_ref = ["8", 0], ["8", 1]

        graph.update({
            "9": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed, "steps": 25, "cfg": 6.5,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
                    "model": ["1", 0], "positive": pos_ref, "negative": neg_ref, "latent_image": ["3", 0],
                },
            },
            "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["2", 0]}},
            "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": f"WF06_ENV_{shot_id}"}},
        })
        return self.execute_workflow("WF-SHOT-ENV", graph)[0]

    # =========================================================================
    # STAGE C: QUALITY REMEDIATION WORKFLOW
    # =========================================================================

    def wf_edit_inpaint(
        self,
        shot_id: str,
        failed_image_path: str,
        repair_prompt: str,
        defect_bbox: tuple[int, int, int, int],
        denoise: float = 0.62,
        seed: int = 600,
    ) -> Path:
        """07. WF-EDIT-INPAINT: Targeted Regional Inpaint & Prop Fix (DifferentialDiffusion)."""
        img_inp = self.stage_input_image(failed_image_path)
        with Image.open(failed_image_path) as im:
            W, H = im.size
        bx, by, bw, bh = defect_bbox

        graph = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.CKPT_ILLUSTRIOUS}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": self.VAE_SDXL}},
            "3": {"class_type": "LoadImage", "inputs": {"image": img_inp}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": f"masterpiece, best quality, {repair_prompt}", "clip": ["1", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": self.DEFAULT_NEG, "clip": ["1", 1]}},
            "6": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": W, "height": H}},
            "7": {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": bw, "height": bh}},
            "8": {"class_type": "MaskComposite", "inputs": {"destination": ["6", 0], "source": ["7", 0], "x": bx, "y": by, "operation": "add"}},
            "9": {"class_type": "DifferentialDiffusion", "inputs": {"model": ["1", 0]}},
            "10": {
                "class_type": "InpaintModelConditioning",
                "inputs": {
                    "positive": ["4", 0], "negative": ["5", 0],
                    "vae": ["2", 0], "pixels": ["3", 0], "mask": ["8", 0],
                    "noise_mask": True,
                },
            },
            "11": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed, "steps": 24, "cfg": 6.5,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": denoise,
                    "model": ["9", 0], "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["10", 2],
                },
            },
            "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["2", 0]}},
            "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"WF07_INPAINT_{shot_id}"}},
        }
        return self.execute_workflow("WF-EDIT-INPAINT", graph)[0]

    # =========================================================================
    # STAGE D: FINAL MASTERING WORKFLOW
    # =========================================================================

    def wf_master_upscale(
        self,
        shot_id: str,
        approved_image_path: str,
        prompt_tags: str,
        scale_factor: float = 2.0,
        denoise: float = 0.24,
        seed: int = 700,
    ) -> Path:
        """08. WF-MASTER-UPSCALE: 4K Lineart & Texture Polish (Tiled VAE Refine)."""
        self.free_vram()
        img_inp = self.stage_input_image(approved_image_path)
        downscale_ratio = max(0.25, min(1.0, scale_factor / 4.0))

        graph = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self.CKPT_ILLUSTRIOUS}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": self.VAE_SDXL}},
            "3": {"class_type": "LoadImage", "inputs": {"image": img_inp}},
            "4": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": self.UPSCALER_ANIME}},
            "5": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["4", 0], "image": ["3", 0]}},
            "6": {"class_type": "ImageScaleBy", "inputs": {"image": ["5", 0], "upscale_method": "lanczos", "scale_by": downscale_ratio}},
            "7": {
                "class_type": "VAEEncodeTiled",
                "inputs": {
                    "pixels": ["6", 0], "vae": ["2", 0],
                    "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8,
                },
            },
            "8": {"class_type": "CLIPTextEncode", "inputs": {"text": f"masterpiece, best quality, ultra detailed, crisp lineart, {prompt_tags}", "clip": ["1", 1]}},
            "9": {"class_type": "CLIPTextEncode", "inputs": {"text": self.DEFAULT_NEG, "clip": ["1", 1]}},
            "10": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed, "steps": 18, "cfg": 6.0,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": denoise,
                    "model": ["1", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["7", 0],
                },
            },
            "11": {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["10", 0], "vae": ["2", 0],
                    "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8,
                },
            },
            "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": f"WF08_MASTER_{shot_id}"}},
        }
        return self.execute_workflow("WF-MASTER-UPSCALE", graph)[0]


def run_full_production_suite() -> None:
    """Executes the full test chain for Sunny, Nephis, and Crimson Spire across all 8 workflows."""
    logger.info("🎬 Starting Full End-to-End Production Chain Test...")
    engine = NovelVisualiserEngine()

    # Stage A
    sunny_ref = engine.wf_foundry_char(
        char_id="sunny",
        identity_tags="1boy, pale skin, messy black hair, sharp dark onyx eyes, slender build, black leather tunic",
        mode="turnaround",
        seed=101,
    )
    nephis_ref = engine.wf_foundry_char(
        char_id="nephis",
        identity_tags="1girl, silver hair, calm grey eyes, tall athletic build, silver knight armor, stoic expression",
        mode="expressions",
        seed=102,
    )
    crimson_spire_loc = engine.wf_foundry_loc(
        project_id="shadow_slave",
        arc_id="forgotten_shore",
        loc_id="crimson_spire_hall",
        landscape_tags="ancient ruined gothic cathedral interior, obsidian pillars, crimson twilight rays through broken stained glass",
        seed=202,
    )
    dagger_prop = engine.wf_foundry_prop(
        prop_id="midnight_shard",
        item_tags="curved silver dagger, obsidian hilt, glowing runic engravings, dark fantasy artifact",
        scale_cm=38.0,
        seed=303,
    )

    # Stage B
    shot_solo = engine.wf_shot_solo(
        shot_id="ch01_s01",
        prompt_8block="medium shot, 1boy, pale skin, messy black hair, dark leather armor, inside ruined gothic cathedral",
        face_ref_path=str(sunny_ref),
        loc_depth_path=str(crimson_spire_loc["depth"]),
        seed=404,
    )
    shot_multi = engine.wf_shot_multi(
        shot_id="ch01_s02",
        bg_prompt="ancient ruined gothic cathedral interior, obsidian pillars, crimson twilight",
        subj1_prompt="1boy, pale skin, messy black hair, sharp dark onyx eyes, dark leather tunic, combat stance",
        subj2_prompt="1girl, silver hair, calm grey eyes, silver knight armor, composed stance",
        subj1_face_path=str(sunny_ref),
        subj2_face_path=str(nephis_ref),
        loc_depth_path=str(crimson_spire_loc["depth"]),
        seed=505,
    )
    shot_env = engine.wf_shot_env(
        shot_id="ch01_s03",
        env_prompt="storm brewing over dark jagged mountains, giant crimson spire on horizon, ominous lightning",
        loc_depth_path=str(crimson_spire_loc["depth"]),
        seed=606,
    )

    # Stage C
    shot_repaired = engine.wf_edit_inpaint(
        shot_id="ch01_s01_fix",
        failed_image_path=str(shot_solo),
        repair_prompt="holding curved silver dagger with obsidian hilt, glowing runic engravings, gauntleted hand",
        defect_bbox=(680, 480, 320, 280),
        denoise=0.62,
        seed=707,
    )

    # Stage D
    shot_mastered = engine.wf_master_upscale(
        shot_id="ch01_s01_master",
        approved_image_path=str(shot_repaired),
        prompt_tags="1boy, messy black hair, dark leather armor, holding silver dagger, gothic cathedral",
        scale_factor=2.0,
        denoise=0.24,
        seed=808,
    )

    logger.info("🎉 All 8 Workflows executed successfully! Mastered output: %s", shot_mastered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Novel Visualiser ComfyUI Production Runner")
    parser.add_argument("--setup", action="store_true", help="Install ComfyUI and custom nodes")
    parser.add_argument("--link", action="store_true", help="Link nv-comfyui-model-library dataset")
    parser.add_argument("--start", action="store_true", help="Start background ComfyUI server")
    parser.add_argument("--test", action="store_true", help="Run full 8-workflow test suite")
    parser.add_argument("--all", action="store_true", help="Run full setup, link, server start and test")
    args = parser.parse_args()

    if args.all or args.setup:
        install_comfyui_and_nodes()
    if args.all or args.link:
        link_model_library()
    if args.all or args.start:
        start_comfyui_server()
    if args.all or args.test:
        run_full_production_suite()
    if not (args.setup or args.link or args.start or args.test or args.all):
        parser.print_help()

