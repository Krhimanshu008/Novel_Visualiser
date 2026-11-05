"""
Novel Visualiser — ComfyUI Client Service (Phase 3)

Provides bidirectional communication with ComfyUI over HTTP and WebSocket:
- Supports local ComfyUI or remote Kaggle/Google Colab T4 GPU via Cloudflare/ngrok tunnels
- Handles image reference uploads to ComfyUI input folder
- Converts and parameterizes user-tested workflows (01_WAI.. to 08_WAI..)
- Submits generation jobs and streams execution progress
- Automatically downloads rendered outputs to output/generated_images/
- Manages canonical Visual Reference locking (WF-001 to WF-004)
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import (
    Entity,
    GenerationJob,
    JobStatus,
    RefType,
    Shot,
    ShotStatus,
    VisualRef,
)

logger = logging.getLogger("novel_visualiser.comfyui")


class ComfyUIWorkflowCompiler:
    """
    Modular Workflow Compiler for Novel Visualiser.
    Implements the 8 verified workflows (WF 00 to WF 07) mapped to the
    14.08 GB Kaggle Dual Tesla T4 model library.
    """

    DEFAULT_CKPT = "sdxl/illustrious/waiNSFWIllustrious_v110.safetensors"
    DEFAULT_VAE = "sdxl_vae_fp16_fix.safetensors"
    DEFAULT_IPADAPTER = "sdxl/plus/ip-adapter-plus_sdxl_vit-h.safetensors"
    DEFAULT_IPADAPTER_FACE = "sdxl/plus/ip-adapter-plus-face_sdxl_vit-h.safetensors"
    DEFAULT_CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    DEFAULT_CONTROLNET = "sdxl/xinsir-controlnet-union-sdxl-1.0-promax.safetensors"
    DEFAULT_UPSCALER = "anime/4x-AnimeSharp.pth"

    WORKFLOW_DIR = Path(__file__).parent.parent.parent / "templates" / "comfyui" / "Workflows"

    @classmethod
    def load_workflow_template(cls, filename: str) -> dict[str, Any]:
        """Load a workflow JSON file from the templates directory."""
        file_path = cls.WORKFLOW_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Workflow file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -------------------------------------------------------------------------
    # WF 00: Diagnostic Base Generator (Fast Draft, No Adapters)
    # -------------------------------------------------------------------------
    @classmethod
    def compile_diagnostic_base(
        cls,
        positive_prompt: str,
        negative_prompt: str = "low quality, worst quality, blurry, deformed, bad eyes",
        seed: Optional[int] = None,
        width: int = 832,
        height: int = 1216,
        steps: int = 24,
        cfg: float = 6.0,
        sampler_name: str = "euler_ancestral",
        scheduler: str = "normal",
        denoise: float = 1.0,
        checkpoint: Optional[str] = None,
        vae: Optional[str] = None,
        filename_prefix: str = "NovelVisualizer/Base_Draft",
    ) -> dict[str, Any]:
        import random
        run_seed = seed if seed is not None else random.randint(1000000, 2147483647)
        ckpt = checkpoint or cls.DEFAULT_CKPT
        vae_model = vae or cls.DEFAULT_VAE

        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": vae_model}},
            "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": positive_prompt}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": negative_prompt}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                    "seed": run_seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": denoise,
                },
            },
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
            "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix}},
        }

    # -------------------------------------------------------------------------
    # WF 01: Character Identity Locking (IP-Adapter Face / Plus)
    # -------------------------------------------------------------------------
    @classmethod
    def compile_character_identity(
        cls,
        reference_image_filename: str,
        positive_prompt: str,
        negative_prompt: str = "worst quality, low quality, bad anatomy, bad hands, blurry, disfigured",
        ipadapter_weight: float = 0.70,
        use_face_model: bool = False,
        seed: Optional[int] = None,
        width: int = 832,
        height: int = 1216,
        steps: int = 24,
        cfg: float = 6.0,
        checkpoint: Optional[str] = None,
        vae: Optional[str] = None,
        filename_prefix: str = "NovelVisualizer/Character_Identity",
    ) -> dict[str, Any]:
        import random
        run_seed = seed if seed is not None else random.randint(1000000, 2147483647)
        ckpt = checkpoint or cls.DEFAULT_CKPT
        vae_model = vae or cls.DEFAULT_VAE
        ipa_model = cls.DEFAULT_IPADAPTER_FACE if use_face_model else cls.DEFAULT_IPADAPTER

        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": vae_model}},
            "9": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": ipa_model}},
            "10": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": cls.DEFAULT_CLIP_VISION}},
            "11": {"class_type": "LoadImage", "inputs": {"image": reference_image_filename}},
            "12": {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": ["1", 0],
                    "ipadapter": ["9", 0],
                    "image": ["11", 0],
                    "clip_vision": ["10", 0],
                    "weight": ipadapter_weight,
                    "weight_type": "linear",
                    "combine_embeds": "concat",
                    "embeds_scaling": "V only",
                    "start_at": 0.0,
                    "end_at": 1.0,
                },
            },
            "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": positive_prompt}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": negative_prompt}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["12", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                    "seed": run_seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
            "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix}},
        }

    # -------------------------------------------------------------------------
    # WF 02: Location & World Generation (Landscape / Architecture)
    # -------------------------------------------------------------------------
    @classmethod
    def compile_location_plate(
        cls,
        positive_prompt: str,
        negative_prompt: str = "worst quality, low quality, bad architecture, blurry, modern cars",
        seed: Optional[int] = None,
        width: int = 1216,
        height: int = 832,
        steps: int = 28,
        cfg: float = 6.0,
        checkpoint: Optional[str] = None,
        vae: Optional[str] = None,
        filename_prefix: str = "NovelVisualizer/Locations",
    ) -> dict[str, Any]:
        return cls.compile_diagnostic_base(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            checkpoint=checkpoint,
            vae=vae,
            filename_prefix=filename_prefix,
        )

    # -------------------------------------------------------------------------
    # WF 03: Scene Composition (Character + OpenPose / Depth ControlNet)
    # -------------------------------------------------------------------------
    @classmethod
    def compile_scene_composition(
        cls,
        reference_image_filename: str,
        positive_prompt: str,
        negative_prompt: str = "worst quality, low quality, bad anatomy, bad hands, blurry",
        ipadapter_weight: float = 0.70,
        control_image_filename: Optional[str] = None,
        control_type: str = "openpose",
        controlnet_strength: float = 0.85,
        seed: Optional[int] = None,
        width: int = 832,
        height: int = 1216,
        steps: int = 24,
        cfg: float = 6.0,
        filename_prefix: str = "NovelVisualizer/Scene_Composition",
    ) -> dict[str, Any]:
        import random
        run_seed = seed if seed is not None else random.randint(1000000, 2147483647)

        nodes: dict[str, Any] = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": cls.DEFAULT_CKPT}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": cls.DEFAULT_VAE}},
            "9": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": cls.DEFAULT_IPADAPTER}},
            "10": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": cls.DEFAULT_CLIP_VISION}},
            "11": {"class_type": "LoadImage", "inputs": {"image": reference_image_filename}},
            "12": {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": ["1", 0],
                    "ipadapter": ["9", 0],
                    "image": ["11", 0],
                    "clip_vision": ["10", 0],
                    "weight": ipadapter_weight,
                    "weight_type": "linear",
                    "combine_embeds": "concat",
                    "embeds_scaling": "V only",
                    "start_at": 0.0,
                    "end_at": 1.0,
                },
            },
            "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": positive_prompt}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": negative_prompt}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        }

        pos_conditioning = ["3", 0]
        neg_conditioning = ["4", 0]

        if control_image_filename:
            nodes["13"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cls.DEFAULT_CONTROLNET}}
            nodes["14"] = {"class_type": "SetUnionControlNetType", "inputs": {"control_net": ["13", 0], "type": control_type}}
            nodes["15"] = {"class_type": "LoadImage", "inputs": {"image": control_image_filename}}
            nodes["16"] = {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "control_net": ["14", 0],
                    "image": ["15", 0],
                    "strength": controlnet_strength,
                    "start_percent": 0.0,
                    "end_percent": 1.0,
                },
            }
            pos_conditioning = ["16", 0]
            neg_conditioning = ["16", 1]

        nodes["6"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": ["12", 0],
                "positive": pos_conditioning,
                "negative": neg_conditioning,
                "latent_image": ["5", 0],
                "seed": run_seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
            },
        }
        nodes["7"] = {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}}
        nodes["8"] = {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix}}
        return nodes

    # -------------------------------------------------------------------------
    # WF 04: Items & Artifacts (Weapons, Relics)
    # -------------------------------------------------------------------------
    @classmethod
    def compile_item_plate(
        cls,
        positive_prompt: str,
        negative_prompt: str = "worst quality, low quality, hands, human, blurry",
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        filename_prefix: str = "NovelVisualizer/Items_Artifacts",
    ) -> dict[str, Any]:
        return cls.compile_diagnostic_base(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            filename_prefix=filename_prefix,
        )

    # -------------------------------------------------------------------------
    # WF 05: Cinematic Battles (High-action Multi-figure)
    # -------------------------------------------------------------------------
    @classmethod
    def compile_cinematic_battle(
        cls,
        reference_image_filename: str,
        positive_prompt: str,
        negative_prompt: str = "worst quality, low quality, bad anatomy, deformed limbs, blurry",
        ipadapter_weight: float = 0.65,
        seed: Optional[int] = None,
        width: int = 1216,
        height: int = 832,
        steps: int = 28,
        cfg: float = 6.0,
        filename_prefix: str = "NovelVisualizer/Cinematic_Battles",
    ) -> dict[str, Any]:
        return cls.compile_character_identity(
            reference_image_filename=reference_image_filename,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            ipadapter_weight=ipadapter_weight,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            filename_prefix=filename_prefix,
        )

    # -------------------------------------------------------------------------
    # WF 07: Master Upscale Pass (4x AnimeSharp + ScaleBy)
    # -------------------------------------------------------------------------
    @classmethod
    def compile_upscale_pass(
        cls,
        input_image_filename: str,
        scale_by: float = 0.5,
        upscale_method: str = "lanczos",
        upscale_model: Optional[str] = None,
        filename_prefix: str = "NovelVisualizer/Upscaled_Master",
    ) -> dict[str, Any]:
        model_name = upscale_model or cls.DEFAULT_UPSCALER
        return {
            "1": {"class_type": "LoadImage", "inputs": {"image": input_image_filename}},
            "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}},
            "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
            "4": {"class_type": "ImageScaleBy", "inputs": {"image": ["3", 0], "upscale_method": upscale_method, "scale_by": scale_by}},
            "5": {"class_type": "SaveImage", "inputs": {"images": ["4", 0], "filename_prefix": filename_prefix}},
        }


class ComfyUIClient:
    """
    HTTP & WebSocket client communicating with ComfyUI.
    Works seamlessly whether ComfyUI is running locally (127.0.0.1:8188)
    or on a remote Kaggle / Google Colab T4 GPU via a Cloudflare / ngrok tunnel!
    """

    def __init__(self, base_url: Optional[str] = None):
        self.settings = get_settings()
        self.base_url = (base_url or self.settings.comfyui.base_url).rstrip("/")
        self.client_id = self.settings.comfyui.client_id
        self.timeout = float(self.settings.comfyui.timeout_seconds)
        self.http_client = httpx.Client(timeout=self.timeout, follow_redirects=True)
        self.compiler = ComfyUIWorkflowCompiler

    def close(self):
        self.http_client.close()

    def check_health(self) -> dict[str, Any]:
        """
        Verify connection to ComfyUI.
        Returns system stats (GPU device, VRAM free, OS).
        """
        url = f"{self.base_url}/system_stats"
        try:
            resp = self.http_client.get(url, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                devices = data.get("devices", [])
                gpu_name = devices[0].get("name", "Unknown GPU") if devices else "CPU / Unknown"
                vram_free = devices[0].get("vram_free", 0) / (1024**3) if devices else 0
                return {
                    "status": "online",
                    "base_url": self.base_url,
                    "gpu_name": gpu_name,
                    "vram_free_gb": round(vram_free, 2),
                    "devices": devices,
                }
            return {"status": "error", "code": resp.status_code, "detail": resp.text}
        except Exception as exc:
            return {"status": "offline", "base_url": self.base_url, "error": str(exc)}

    def upload_image(self, image_path: Path | str, overwrite: bool = True) -> str:
        """
        Upload a local image (e.g. character reference art) to ComfyUI's input directory.
        Returns the filename stored on the ComfyUI server.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {path}")

        url = f"{self.base_url}/upload/image"
        with open(path, "rb") as f:
            files = {"image": (path.name, f, "image/png")}
            data = {"overwrite": "true" if overwrite else "false"}
            resp = self.http_client.post(url, files=files, data=data)
            resp.raise_for_status()
            res_data = resp.json()
            server_filename = res_data.get("name", path.name)
            logger.info("Uploaded reference image to ComfyUI: %s", server_filename)
            return server_filename

    def submit_prompt(self, prompt_dict: dict[str, Any]) -> str:
        """
        Submit a prompt dictionary to ComfyUI's /prompt queue.
        Automatically checks and uploads any local reference images referenced
        by LoadImage nodes before submission.
        Returns the prompt_id (job UUID).
        """
        # Auto-upload local reference images referenced by LoadImage nodes
        for node_id, node_data in prompt_dict.items():
            if isinstance(node_data, dict) and node_data.get("class_type") == "LoadImage":
                img_val = node_data.get("inputs", {}).get("image")
                if img_val and isinstance(img_val, str):
                    candidate_paths = [
                        Path(img_val),
                        Path("output/generated_images") / img_val,
                        Path("data/references") / img_val,
                    ]
                    for cp in candidate_paths:
                        if cp.exists() and cp.is_file():
                            try:
                                server_name = self.upload_image(cp)
                                node_data["inputs"]["image"] = server_name
                            except Exception as e:
                                logger.warning("Could not auto-upload reference image %s: %s", cp, e)
                            break

        url = f"{self.base_url}/prompt"
        payload = {
            "prompt": prompt_dict,
            "client_id": self.client_id,
        }
        resp = self.http_client.post(url, json=payload)
        if resp.status_code != 200:
            logger.error("ComfyUI /prompt rejected with status %d: %s", resp.status_code, resp.text)
            raise RuntimeError(f"ComfyUI /prompt rejected with status {resp.status_code}: {resp.text}")
        resp.raise_for_status()
        data = resp.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI rejected prompt: {data}")
        logger.info("Queued ComfyUI job (prompt_id: %s)", prompt_id)
        return prompt_id

    def wait_for_execution(
        self,
        prompt_id: str,
        poll_interval: float = 2.0,
        max_wait: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Poll /history/{prompt_id} until the generation job completes.
        Returns the output node dictionary containing image filenames.
        """
        timeout = max_wait if max_wait is not None else self.timeout
        start_time = time.time()
        url = f"{self.base_url}/history/{prompt_id}"

        while time.time() - start_time < timeout:
            resp = self.http_client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if prompt_id in data:
                    job_data = data[prompt_id]
                    outputs = job_data.get("outputs", {})
                    status_info = job_data.get("status", {})
                    if status_info.get("completed", False) or outputs:
                        logger.info("ComfyUI job %s completed in %.1fs", prompt_id, time.time() - start_time)
                        return outputs

            time.sleep(poll_interval)

        raise TimeoutError(f"ComfyUI job {prompt_id} timed out after {timeout}s")

    def download_output_image(
        self,
        filename: str,
        subfolder: str = "",
        image_type: str = "output",
        target_dir: Optional[Path] = None,
    ) -> Path:
        """
        Download a generated image from ComfyUI /view endpoint to local output directory.
        """
        settings = get_settings()
        dest_dir = target_dir or Path(settings.project.media_root) / "generated_images"
        dest_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "filename": filename,
            "subfolder": subfolder,
            "type": image_type,
        }
        url = f"{self.base_url}/view"
        resp = self.http_client.get(url, params=params)
        resp.raise_for_status()

        dest_file = dest_dir / filename
        dest_file.write_bytes(resp.content)
        logger.info("Saved generated image to %s (%d KB)", dest_file, len(resp.content) // 1024)
        return dest_file

    def generate_image_sync(
        self,
        prompt_dict: dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> list[Path]:
        """
        End-to-end sync execution:
        1. Submits prompt
        2. Waits for completion
        3. Downloads and returns local paths to all generated images
        """
        prompt_id = self.submit_prompt(prompt_dict)
        outputs = self.wait_for_execution(prompt_id)

        downloaded_paths: list[Path] = []
        for node_id, node_out in outputs.items():
            if "images" in node_out:
                for img_info in node_out["images"]:
                    fname = img_info.get("filename")
                    subf = img_info.get("subfolder", "")
                    itype = img_info.get("type", "output")
                    if fname:
                        saved_path = self.download_output_image(
                            filename=fname,
                            subfolder=subf,
                            image_type=itype,
                            target_dir=output_dir,
                        )
                        downloaded_paths.append(saved_path)

        return downloaded_paths

    def lock_canonical_visual_reference(
        self,
        db: Session,
        entity_id: str,
        image_path: str,
        ref_type: RefType = RefType.TURNAROUND_SHEET,
        seed: Optional[int] = None,
        lora_config: Optional[dict] = None,
        ip_adapter_config: Optional[dict] = None,
    ) -> VisualRef:
        """
        Phase 3 Lock Gate:
        Stores approved reference image, seed, and IPAdapter config in SQLite visual_refs table.
        This locks the canonical biometric & visual appearance for future chapters.
        """
        ref = VisualRef(
            entity_id=entity_id,
            ref_type=ref_type,
            image_path=image_path,
            seed=seed,
            lora_config=lora_config,
            ip_adapter_config=ip_adapter_config,
            approved=True,
            approved_at=datetime.utcnow(),
        )
        db.add(ref)
        db.commit()
        logger.info("Visual reference locked for entity %s: %s (seed: %s)", entity_id, image_path, seed)
        return ref

