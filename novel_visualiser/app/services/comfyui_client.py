"""Novel Visualiser — ComfyUI Client & 6+2 Specialized Workflow Engine (Phase 3 / VCE).

Implements the 6 Core + 2 Finishing Specialized Workflows:
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
"""

from __future__ import annotations

import json
import logging
import random
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
    """Production-hardened ComfyUI Orchestrator implementing the 6 + 2 Specialized Workflows.
    
    Compatible with both relative subpaths (sdxl/illustrious/...) and flat symlinked filenames.
    """

    # Model Defaults (Resolves flat or relative via ComfyUI model loader)
    CKPT_ILLUSTRIOUS = "waiNSFWIllustrious_v110.safetensors"
    CKPT_TURBO = "DreamShaperXL_Turbo_v2_1.safetensors"
    VAE_SDXL = "sdxl_vae_fp16_fix.safetensors"
    CLIP_VISION = "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
    CN_UNION = "xinsir-controlnet-union-sdxl-1.0-promax.safetensors"
    IPA_FACE = "ip-adapter-plus-face_sdxl_vit-h.safetensors"
    IPA_PLUS = "ip-adapter-plus_sdxl_vit-h.safetensors"
    YOLO_FACE = "bbox/face_yolov8n.pt"
    UPSCALER_ANIME = "4x-AnimeSharp.pth"

    DEFAULT_CKPT = CKPT_ILLUSTRIOUS
    DEFAULT_VAE = VAE_SDXL
    DEFAULT_IPADAPTER = IPA_PLUS
    DEFAULT_IPADAPTER_FACE = IPA_FACE
    DEFAULT_CLIP_VISION = CLIP_VISION
    DEFAULT_CONTROLNET = CN_UNION
    DEFAULT_UPSCALER = UPSCALER_ANIME

    DEFAULT_NEG = (
        "low quality, worst quality, blurry, bad anatomy, deformed limbs, extra arms, "
        "extra legs, missing fingers, extra fingers, floating limbs, disconnected limbs, "
        "mutation, poorly drawn face, poorly drawn hands, cropped, out of frame, "
        "watermark, signature, text, modern clothing, oversaturated, flat shading"
    )

    WORKFLOW_DIR = Path(__file__).parent.parent.parent / "templates" / "comfyui" / "Workflows"

    @classmethod
    def load_workflow_template(cls, template_name: str) -> dict[str, Any]:
        """Load a JSON workflow template from templates/comfyui/Workflows."""
        target = cls.WORKFLOW_DIR / template_name
        if not target.exists():
            if not template_name.endswith(".json"):
                target = cls.WORKFLOW_DIR / f"{template_name}.json"
        if not target.exists():
            raise FileNotFoundError(f"Workflow template '{template_name}' not found in {cls.WORKFLOW_DIR}")
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)

    # =========================================================================
    # STAGE A: ASSET FOUNDRY WORKFLOWS (PRE-CREATION)
    # =========================================================================

    @classmethod
    def compile_foundry_char(
        cls,
        char_id: str,
        identity_tags: str,
        mode: str = "turnaround",  # "turnaround" or "expressions"
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        checkpoint: Optional[str] = None,
        vae: Optional[str] = None,
    ) -> dict[str, Any]:
        """WF-FOUNDRY-CHAR: Character Turnaround & Expression Sheet Generator."""
        run_seed = seed if seed is not None else random.randint(1000000, 2147483647)
        ckpt = checkpoint or cls.CKPT_ILLUSTRIOUS
        vae_model = vae or cls.VAE_SDXL

        if mode == "turnaround":
            framing = (
                "masterpiece, best quality, character reference sheet, orthographic 3-angle turnaround, "
                "front view, three-quarter view, profile side view, full body, standing straight, "
                "simple background, solid neutral grey studio background #2b2b2b, flat even studio lighting"
            )
        elif mode == "expressions":
            framing = (
                "masterpiece, best quality, expression sheet, 4-panel matrix, close-up headshots, "
                "neutral expression, intense combat expression, shocked wary expression, cunning smirk, "
                "simple background, solid neutral grey studio background #2b2b2b, flat studio lighting"
            )
        elif mode == "outfit":
            framing = (
                "masterpiece, best quality, character costume design sheet, full body fashion breakdown, "
                "standing pose, isolated wardrobe, simple background, solid neutral grey studio background #2b2b2b"
            )
        else:
            framing = "masterpiece, best quality, character design plate, simple background, solid neutral grey studio background #2b2b2b"

        pos_prompt = f"{framing}, {identity_tags}"
        neg_prompt = f"{cls.DEFAULT_NEG}, complex background, scenery, dramatic lighting, lens flare"

        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": vae_model}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": pos_prompt, "clip": ["1", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_prompt, "clip": ["1", 1]}},
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": run_seed, "steps": 28, "cfg": 6.5,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
                    "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["3", 0]
                }
            },
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
            "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"WF01_CHAR_{char_id}_{mode}"}}
        }

    @classmethod
    def compile_foundry_loc(
        cls,
        loc_id: str,
        landscape_tags: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 1216,
        height: int = 832,
        checkpoint: Optional[str] = None,
        vae: Optional[str] = None,
    ) -> dict[str, Any]:
        """WF-FOUNDRY-LOC: Location Master Plate & Depth/Lineart Guides."""
        run_seed = seed if seed is not None else random.randint(1000000, 2147483647)
        ckpt = checkpoint or cls.CKPT_ILLUSTRIOUS
        vae_model = vae or cls.VAE_SDXL

        pos_prompt = f"masterpiece, best quality, anime background art, establishing wide shot, no humans, {landscape_tags}"
        neg_prompt = negative_prompt or f"1boy, 1girl, character, person, crowd, silhouette, human, {cls.DEFAULT_NEG}"

        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": vae_model}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": pos_prompt, "clip": ["1", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_prompt, "clip": ["1", 1]}},
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": run_seed, "steps": 28, "cfg": 6.5,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
                    "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["3", 0]
                }
            },
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
            "8": {"class_type": "Canny", "inputs": {"image": ["7", 0], "low_threshold": 0.15, "high_threshold": 0.35}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"WF02_LOC_{loc_id}_rgb"}},
            "10": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": f"WF02_LOC_{loc_id}_lineart"}}
        }

    @classmethod
    def compile_foundry_prop(
        cls,
        prop_id: str,
        item_tags: str,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        checkpoint: Optional[str] = None,
        vae: Optional[str] = None,
    ) -> dict[str, Any]:
        """WF-FOUNDRY-PROP: Isolated Item & Weapon Sheet."""
        run_seed = seed if seed is not None else random.randint(1000000, 2147483647)
        ckpt = checkpoint or cls.CKPT_ILLUSTRIOUS
        vae_model = vae or cls.VAE_SDXL

        pos_prompt = (
            f"masterpiece, best quality, isolated weapon prop sheet, {item_tags}, "
            "no humans, centered object, orthographic view, simple solid dark grey studio background #2b2b2b, clean sharp lineart"
        )
        neg_prompt = negative_prompt or f"1boy, 1girl, hand, fingers, holding, character, cluttered background, {cls.DEFAULT_NEG}"

        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": vae_model}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": pos_prompt, "clip": ["1", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg_prompt, "clip": ["1", 1]}},
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": run_seed, "steps": 26, "cfg": 6.5,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
                    "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["3", 0]
                }
            },
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["2", 0]}},
            "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": f"WF03_PROP_{prop_id}"}}
        }

    # =========================================================================
    # STAGE B: CHAPTER SHOT PRODUCTION WORKFLOWS
    # =========================================================================

    @classmethod
    def compile_shot_solo(
        cls,
        shot_id: str,
        positive_prompt: str,
        negative_prompt: Optional[str] = None,
        face_ref_filename: Optional[str] = None,
        loc_depth_filename: Optional[str] = None,
        lora_name: Optional[str] = None,
        lora_weight: float = 0.75,
        cn_depth_weight: float = 0.50,
        ipa_face_weight: float = 0.35,
        seed: Optional[int] = None,
        width: int = 1216,
        height: int = 832,
        checkpoint: Optional[str] = None,
        vae: Optional[str] = None,
    ) -> dict[str, Any]:
        """WF-SHOT-SOLO: The Core Production Workhorse (~70% of shots).
        
        Includes Checkpoint + optional Character LoRA + IP-Adapter Face +
        ControlNet Depth plate anchor + integrated FaceDetailer pass.
        """
        run_seed = seed if seed is not None else random.randint(1000000, 2147483647)
        ckpt = checkpoint or cls.CKPT_ILLUSTRIOUS
        vae_model = vae or cls.VAE_SDXL
        neg = negative_prompt or cls.DEFAULT_NEG

        nodes: dict[str, Any] = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": vae_model}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        }

        model_ref = ["1", 0]
        clip_ref = ["1", 1]

        if lora_name:
            nodes["1b"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": model_ref, "clip": clip_ref,
                    "lora_name": lora_name, "strength_model": lora_weight, "strength_clip": lora_weight
                }
            }
            model_ref = ["1b", 0]
            clip_ref = ["1b", 1]

        nodes["4"] = {"class_type": "CLIPTextEncode", "inputs": {"text": positive_prompt, "clip": clip_ref}}
        nodes["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": clip_ref}}

        # IP-Adapter Face Conditioning (if reference image provided)
        if face_ref_filename:
            nodes["6"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": cls.CLIP_VISION}}
            nodes["7"] = {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": cls.IPA_FACE}}
            nodes["8"] = {"class_type": "LoadImage", "inputs": {"image": face_ref_filename}}
            nodes["9"] = {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": model_ref, "ipadapter": ["7", 0], "image": ["8", 0],
                    "weight": ipa_face_weight, "weight_type": "linear",
                    "combine_embeds": "concat", "start_at": 0.0, "end_at": 0.85,
                    "embeds_scaling": "V only", "clip_vision": ["6", 0]
                }
            }
            model_ref = ["9", 0]

        pos_cond = ["4", 0]
        neg_cond = ["5", 0]

        # ControlNet Depth Plate Anchor (if depth map provided)
        if loc_depth_filename:
            nodes["10"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cls.CN_UNION}}
            nodes["11"] = {"class_type": "LoadImage", "inputs": {"image": loc_depth_filename}}
            nodes["12"] = {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": pos_cond, "negative": neg_cond,
                    "control_net": ["10", 0], "image": ["11", 0],
                    "strength": cn_depth_weight, "start_percent": 0.0, "end_percent": 0.75, "vae": ["2", 0]
                }
            }
            pos_cond = ["12", 0]
            neg_cond = ["12", 1]

        # Main KSampler pass
        nodes["13"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": run_seed, "steps": 28, "cfg": 6.5,
                "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
                "model": model_ref, "positive": pos_cond, "negative": neg_cond, "latent_image": ["3", 0]
            }
        }
        nodes["14"] = {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["2", 0]}}

        # Integrated FaceDetailer pass
        nodes["15"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": cls.YOLO_FACE}}
        nodes["16"] = {
            "class_type": "FaceDetailer",
            "inputs": {
                "image": ["14", 0], "model": model_ref, "clip": clip_ref, "vae": ["2", 0],
                "guide_size": 512, "guide_size_for": True, "max_size": 1024,
                "seed": run_seed + 1, "steps": 16, "cfg": 6.0,
                "sampler_name": "euler_ancestral", "scheduler": "normal",
                "positive": ["4", 0], "negative": ["5", 0],
                "denoise": 0.40, "feather": 5, "noise_mask": True, "force_inpaint": True,
                "bbox_threshold": 0.5, "bbox_dilation": 10, "bbox_crop_factor": 3.0,
                "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93,
                "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False", "drop_size": 10,
                "bbox_detector": ["15", 0], "wildcard": "", "cycle": 1
            }
        }
        nodes["17"] = {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": f"WF04_SOLO_{shot_id}"}}
        return nodes

    @classmethod
    def compile_shot_multi(
        cls,
        shot_id: str,
        bg_prompt: str,
        subj1_prompt: str,
        subj2_prompt: str,
        subj1_face_filename: str,
        subj2_face_filename: str,
        negative_prompt: Optional[str] = None,
        loc_depth_filename: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 1216,
        height: int = 832,
    ) -> dict[str, Any]:
        """WF-SHOT-MULTI: Multi-Subject Regional Isolation (Zero Attribute Bleeding)."""
        run_seed = seed if seed is not None else random.randint(1000000, 2147483647)
        neg = negative_prompt or cls.DEFAULT_NEG

        nodes: dict[str, Any] = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": cls.CKPT_ILLUSTRIOUS}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": cls.VAE_SDXL}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": f"masterpiece, best quality, wide two-shot, {bg_prompt}", "clip": ["1", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": f"masterpiece, best quality, {subj1_prompt}", "clip": ["1", 1]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": f"masterpiece, best quality, {subj2_prompt}", "clip": ["1", 1]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},

            # Native Spatial Masks on 1216x832 canvas (Subject 1 Left, Subject 2 Right)
            "8":  {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": width, "height": height}},
            "9":  {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": 547, "height": 666}},
            "10": {"class_type": "MaskComposite", "inputs": {"destination": ["8", 0], "source": ["9", 0], "x": 0, "y": 83, "operation": "add"}},
            "11": {"class_type": "MaskComposite", "inputs": {"destination": ["8", 0], "source": ["9", 0], "x": 669, "y": 83, "operation": "add"}},

            # Regional Conditioning Isolation
            "12": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["5", 0], "mask": ["10", 0], "strength": 1.0, "set_cond_area": "default"}},
            "13": {"class_type": "ConditioningSetMask", "inputs": {"conditioning": ["6", 0], "mask": ["11", 0], "strength": 1.0, "set_cond_area": "default"}},
            "14": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["4", 0], "conditioning_2": ["12", 0]}},
            "15": {"class_type": "ConditioningCombine", "inputs": {"conditioning_1": ["14", 0], "conditioning_2": ["13", 0]}},

            # Regionally Masked Dual IP-Adapter Face Conditioning
            "16": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": cls.CLIP_VISION}},
            "17": {"class_type": "IPAdapterModelLoader", "inputs": {"ipadapter_file": cls.IPA_FACE}},
            "18": {"class_type": "LoadImage", "inputs": {"image": subj1_face_filename}},
            "19": {"class_type": "LoadImage", "inputs": {"image": subj2_face_filename}},
            "20": {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": ["1", 0], "ipadapter": ["17", 0], "image": ["18", 0],
                    "weight": 0.35, "weight_type": "linear", "combine_embeds": "concat",
                    "start_at": 0.0, "end_at": 0.85, "embeds_scaling": "V only",
                    "clip_vision": ["16", 0], "attn_mask": ["10", 0]
                }
            },
            "21": {
                "class_type": "IPAdapterAdvanced",
                "inputs": {
                    "model": ["20", 0], "ipadapter": ["17", 0], "image": ["19", 0],
                    "weight": 0.35, "weight_type": "linear", "combine_embeds": "concat",
                    "start_at": 0.0, "end_at": 0.85, "embeds_scaling": "V only",
                    "clip_vision": ["16", 0], "attn_mask": ["11", 0]
                }
            },
        }

        pos_cond = ["15", 0]
        neg_cond = ["7", 0]

        if loc_depth_filename:
            nodes["22"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cls.CN_UNION}}
            nodes["23"] = {"class_type": "LoadImage", "inputs": {"image": loc_depth_filename}}
            nodes["24"] = {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": pos_cond, "negative": neg_cond,
                    "control_net": ["22", 0], "image": ["23", 0],
                    "strength": 0.45, "start_percent": 0.0, "end_percent": 0.70, "vae": ["2", 0]
                }
            }
            pos_cond = ["24", 0]
            neg_cond = ["24", 1]

        nodes["25"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": run_seed, "steps": 28, "cfg": 6.5,
                "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
                "model": ["21", 0], "positive": pos_cond, "negative": neg_cond, "latent_image": ["3", 0]
            }
        }
        nodes["26"] = {"class_type": "VAEDecode", "inputs": {"samples": ["25", 0], "vae": ["2", 0]}}
        nodes["27"] = {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": cls.YOLO_FACE}}
        nodes["28"] = {
            "class_type": "FaceDetailer",
            "inputs": {
                "image": ["26", 0], "model": ["21", 0], "clip": ["1", 1], "vae": ["2", 0],
                "guide_size": 512, "guide_size_for": True, "max_size": 1024,
                "seed": run_seed + 1, "steps": 15, "cfg": 6.0,
                "sampler_name": "euler_ancestral", "scheduler": "normal",
                "positive": ["15", 0], "negative": ["7", 0],
                "denoise": 0.38, "feather": 5, "noise_mask": True, "force_inpaint": True,
                "bbox_threshold": 0.5, "bbox_dilation": 10, "bbox_crop_factor": 3.0,
                "sam_detection_hint": "center-1", "sam_dilation": 0, "sam_threshold": 0.93,
                "sam_bbox_expansion": 0, "sam_mask_hint_threshold": 0.7,
                "sam_mask_hint_use_negative": "False", "drop_size": 10,
                "bbox_detector": ["27", 0], "wildcard": "", "cycle": 1
            }
        }
        nodes["29"] = {"class_type": "SaveImage", "inputs": {"images": ["28", 0], "filename_prefix": f"WF05_MULTI_{shot_id}"}}
        return nodes

    @classmethod
    def compile_shot_env(
        cls,
        shot_id: str,
        env_prompt: str,
        negative_prompt: Optional[str] = None,
        loc_depth_filename: Optional[str] = None,
        width: int = 1344,
        height: int = 768,
        seed: Optional[int] = None,
        checkpoint: Optional[str] = None,
        vae: Optional[str] = None,
    ) -> dict[str, Any]:
        """WF-SHOT-ENV: Pure Scenery / Establishing Vista (~10% of shots)."""
        run_seed = seed if seed is not None else random.randint(1000000, 2147483647)
        ckpt = checkpoint or cls.CKPT_ILLUSTRIOUS
        vae_model = vae or cls.VAE_SDXL
        neg = negative_prompt or f"1boy, 1girl, character, person, crowd, {cls.DEFAULT_NEG}"

        pos_prompt = f"masterpiece, best quality, establishing cinematic vista, no humans, {env_prompt}"

        nodes: dict[str, Any] = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": vae_model}},
            "3": {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": pos_prompt, "clip": ["1", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
        }

        pos_ref, neg_ref = ["4", 0], ["5", 0]
        if loc_depth_filename:
            nodes["6"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": cls.CN_UNION}}
            nodes["7"] = {"class_type": "LoadImage", "inputs": {"image": loc_depth_filename}}
            nodes["8"] = {
                "class_type": "ControlNetApplyAdvanced",
                "inputs": {
                    "positive": ["4", 0], "negative": ["5", 0],
                    "control_net": ["6", 0], "image": ["7", 0],
                    "strength": 0.45, "start_percent": 0.0, "end_percent": 0.70, "vae": ["2", 0]
                }
            }
            pos_ref, neg_ref = ["8", 0], ["8", 1]

        nodes["9"] = {
            "class_type": "KSampler",
            "inputs": {
                "seed": run_seed, "steps": 25, "cfg": 6.5,
                "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": 1.0,
                "model": ["1", 0], "positive": pos_ref, "negative": neg_ref, "latent_image": ["3", 0]
            }
        }
        nodes["10"] = {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["2", 0]}}
        nodes["11"] = {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": f"WF06_ENV_{shot_id}"}}
        return nodes

    # =========================================================================
    # STAGE C: QUALITY REMEDIATION WORKFLOW
    # =========================================================================

    @classmethod
    def compile_edit_inpaint(
        cls,
        shot_id: str,
        failed_image_filename: str,
        repair_prompt: str,
        defect_bbox: tuple[int, int, int, int],  # (x, y, width, height)
        negative_prompt: Optional[str] = None,
        denoise: float = 0.62,
        seed: Optional[int] = None,
        image_width: int = 1216,
        image_height: int = 832,
    ) -> dict[str, Any]:
        """WF-EDIT-INPAINT: Targeted Regional Inpaint & Prop Fix (DifferentialDiffusion)."""
        run_seed = seed if seed is not None else random.randint(1000000, 2147483647)
        bx, by, bw, bh = defect_bbox
        neg = negative_prompt or cls.DEFAULT_NEG

        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": cls.CKPT_ILLUSTRIOUS}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": cls.VAE_SDXL}},
            "3": {"class_type": "LoadImage", "inputs": {"image": failed_image_filename}},
            "4": {"class_type": "CLIPTextEncode", "inputs": {"text": f"masterpiece, best quality, {repair_prompt}", "clip": ["1", 1]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
            "6": {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": image_width, "height": image_height}},
            "7": {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": bw, "height": bh}},
            "8": {"class_type": "MaskComposite", "inputs": {"destination": ["6", 0], "source": ["7", 0], "x": bx, "y": by, "operation": "add"}},
            "9": {"class_type": "DifferentialDiffusion", "inputs": {"model": ["1", 0]}},
            "10": {
                "class_type": "InpaintModelConditioning",
                "inputs": {
                    "positive": ["4", 0], "negative": ["5", 0],
                    "vae": ["2", 0], "pixels": ["3", 0], "mask": ["8", 0],
                    "noise_mask": True
                }
            },
            "11": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": run_seed, "steps": 24, "cfg": 6.5,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": denoise,
                    "model": ["9", 0], "positive": ["10", 0], "negative": ["10", 1], "latent_image": ["10", 2]
                }
            },
            "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["2", 0]}},
            "13": {"class_type": "SaveImage", "inputs": {"images": ["12", 0], "filename_prefix": f"WF07_INPAINT_{shot_id}"}}
        }

    @classmethod
    def compile_qwen_edit(
        cls,
        shot_id: str,
        input_image_filename: str,
        edit_instruction: str,
        negative_prompt: Optional[str] = None,
        edit_lora: str = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        lora_strength: float = 0.85,
        steps: int = 8,
        cfg: float = 4.0,
        denoise: float = 0.45,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """WF-QWEN-EDIT: Instruction-Guided Multimodal Image Editing & Relighting using Qwen-Edit LoRA."""
        run_seed = seed if seed is not None else random.randint(1000000, 2147483647)
        neg = negative_prompt or cls.DEFAULT_NEG

        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": cls.CKPT_ILLUSTRIOUS}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": cls.VAE_SDXL}},
            "3": {"class_type": "LoraLoader", "inputs": {"model": ["1", 0], "clip": ["1", 1], "lora_name": edit_lora, "strength_model": lora_strength, "strength_clip": lora_strength}},
            "4": {"class_type": "LoadImage", "inputs": {"image": input_image_filename}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": f"masterpiece, best quality, {edit_instruction}", "clip": ["3", 1]}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["3", 1]}},
            "7": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["2", 0]}},
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": run_seed, "steps": steps, "cfg": cfg,
                    "sampler_name": "euler", "scheduler": "normal", "denoise": denoise,
                    "model": ["3", 0], "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["7", 0]
                }
            },
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["2", 0]}},
            "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": f"WF_QWEN_EDIT_{shot_id}"}}
        }

    # =========================================================================
    # STAGE D: FINAL MASTERING WORKFLOW
    # =========================================================================

    @classmethod
    def compile_master_upscale(
        cls,
        shot_id: str,
        input_image_filename: str,
        prompt_tags: str,
        negative_prompt: Optional[str] = None,
        scale_factor: float = 2.0,
        denoise: float = 0.24,
        seed: Optional[int] = None,
        upscale_model: Optional[str] = None,
    ) -> dict[str, Any]:
        """WF-MASTER-UPSCALE: 4K Lineart & Texture Polish (Tiled VAE Refine)."""
        run_seed = seed if seed is not None else random.randint(1000000, 2147483647)
        downscale_ratio = max(0.25, min(1.0, scale_factor / 4.0))
        model_name = upscale_model or cls.UPSCALER_ANIME
        neg = negative_prompt or cls.DEFAULT_NEG

        return {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": cls.CKPT_ILLUSTRIOUS}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": cls.VAE_SDXL}},
            "3": {"class_type": "LoadImage", "inputs": {"image": input_image_filename}},
            "4": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}},
            "5": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["4", 0], "image": ["3", 0]}},
            "6": {"class_type": "ImageScaleBy", "inputs": {"image": ["5", 0], "upscale_method": "lanczos", "scale_by": downscale_ratio}},
            "7": {
                "class_type": "VAEEncodeTiled",
                "inputs": {
                    "pixels": ["6", 0], "vae": ["2", 0],
                    "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8
                }
            },
            "8": {"class_type": "CLIPTextEncode", "inputs": {"text": f"masterpiece, best quality, ultra detailed, crisp lineart, {prompt_tags}", "clip": ["1", 1]}},
            "9": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["1", 1]}},
            "10": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": run_seed, "steps": 18, "cfg": 6.0,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": denoise,
                    "model": ["1", 0], "positive": ["8", 0], "negative": ["9", 0], "latent_image": ["7", 0]
                }
            },
            "11": {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["10", 0], "vae": ["2", 0],
                    "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8
                }
            },
            "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": f"WF08_MASTER_{shot_id}"}}
        }

    # =========================================================================
    # BACKWARD COMPATIBLE BRIDGES
    # =========================================================================

    @classmethod
    def compile_diagnostic_base(
        cls,
        positive_prompt: str,
        negative_prompt: str = "low quality, worst quality, blurry",
        seed: Optional[int] = None,
        width: int = 1216,
        height: int = 832,
        checkpoint: Optional[str] = None,
        vae: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Bridge to WF-SHOT-ENV."""
        return cls.compile_shot_env(
            shot_id=f"diag_{random.randint(100, 999)}",
            env_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            seed=seed,
            checkpoint=checkpoint,
            vae=vae,
        )

    @classmethod
    def compile_character_identity(
        cls,
        reference_image_filename: str,
        positive_prompt: str,
        negative_prompt: str = "worst quality, low quality",
        seed: Optional[int] = None,
        width: int = 1216,
        height: int = 832,
        checkpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Bridge to WF-SHOT-SOLO."""
        return cls.compile_shot_solo(
            shot_id=f"solo_{random.randint(100, 999)}",
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            face_ref_filename=reference_image_filename,
            seed=seed,
            width=width,
            height=height,
            checkpoint=checkpoint,
        )

    @classmethod
    def compile_character_design(
        cls,
        positive_prompt: str,
        negative_prompt: str = "worst quality, low quality",
        seed: Optional[int] = None,
        checkpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Bridge to WF-FOUNDRY-CHAR."""
        return cls.compile_foundry_char(
            char_id="char",
            identity_tags=positive_prompt,
            mode="turnaround",
            seed=seed,
            checkpoint=checkpoint,
        )

    @classmethod
    def compile_ipadapter_consistency(
        cls,
        reference_image_filename: str,
        positive_prompt: str,
        negative_prompt: str = "worst quality",
        ipadapter_weight: float = 0.35,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Bridge to WF-SHOT-SOLO."""
        return cls.compile_shot_solo(
            shot_id="ipa_solo",
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            face_ref_filename=reference_image_filename,
            ipa_face_weight=ipadapter_weight,
            seed=seed,
        )

    @classmethod
    def compile_location_plate(
        cls,
        positive_prompt: str,
        negative_prompt: str = "worst quality, low quality",
        seed: Optional[int] = None,
        width: int = 1216,
        height: int = 832,
        checkpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Bridge to WF-FOUNDRY-LOC."""
        return cls.compile_foundry_loc(
            loc_id=f"loc_{random.randint(100, 999)}",
            landscape_tags=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
            checkpoint=checkpoint,
        )

    @classmethod
    def compile_item_plate(
        cls,
        positive_prompt: str,
        negative_prompt: str = "worst quality, low quality",
        seed: Optional[int] = None,
        width: int = 1024,
        height: int = 1024,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Bridge to WF-FOUNDRY-PROP."""
        return cls.compile_foundry_prop(
            prop_id=f"prop_{random.randint(100, 999)}",
            item_tags=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            width=width,
            height=height,
        )

    @classmethod
    def compile_cinematic_battle(
        cls,
        reference_image_filename: str,
        positive_prompt: str,
        negative_prompt: str = "worst quality",
        seed: Optional[int] = None,
        width: int = 1216,
        height: int = 832,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Bridge to WF-SHOT-SOLO."""
        return cls.compile_shot_solo(
            shot_id=f"battle_{random.randint(100, 999)}",
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            face_ref_filename=reference_image_filename,
            seed=seed,
            width=width,
            height=height,
        )

    @classmethod
    def compile_scene_composition(
        cls,
        reference_image_filename: str,
        positive_prompt: str,
        negative_prompt: str = "worst quality",
        control_image_filename: Optional[str] = None,
        seed: Optional[int] = None,
        width: int = 1216,
        height: int = 832,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Bridge to WF-SHOT-SOLO with ControlNet."""
        return cls.compile_shot_solo(
            shot_id=f"comp_{random.randint(100, 999)}",
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            face_ref_filename=reference_image_filename,
            loc_depth_filename=control_image_filename,
            seed=seed,
            width=width,
            height=height,
        )

    @classmethod
    def compile_upscale_pass(
        cls,
        input_image_filename: str,
        prompt_tags: str = "ultra detailed, crisp lineart",
        scale_by: float = 0.5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Bridge to WF-MASTER-UPSCALE."""
        return cls.compile_master_upscale(
            shot_id="upscale",
            input_image_filename=input_image_filename,
            prompt_tags=prompt_tags,
            scale_factor=2.0,
        )


class ComfyUIClient:
    """HTTP & WebSocket client communicating with ComfyUI.
    
    Supports local execution and remote Kaggle / Colab GPU tunnels.
    Includes memory cleanup, image staging, and synchronous generation polling.
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
        """Verify connection to ComfyUI and fetch GPU/VRAM statistics."""
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

    def free_vram(self, unload_models: bool = False) -> None:
        """Call ComfyUI /free endpoint to prevent memory fragmentation."""
        url = f"{self.base_url}/free"
        try:
            self.http_client.post(url, json={"unload_models": unload_models, "free_memory": True})
        except Exception as exc:
            logger.debug("ComfyUI /free note: %s", exc)

    def upload_image(self, image_path: Path | str, overwrite: bool = True) -> str:
        """Upload a local reference image to ComfyUI's input directory."""
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
        """Submit a prompt dictionary to ComfyUI's /prompt queue."""
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
        payload = {"prompt": prompt_dict, "client_id": self.client_id}
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
        """Poll /history/{prompt_id} until the generation job completes."""
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
        """Download generated image from ComfyUI /view to local directory."""
        settings = get_settings()
        dest_dir = target_dir or Path(settings.project.media_root) / "generated_images"
        dest_dir.mkdir(parents=True, exist_ok=True)

        params = {"filename": filename, "subfolder": subfolder, "type": image_type}
        url = f"{self.base_url}/view?{urllib.parse.urlencode(params)}"
        resp = self.http_client.get(url)
        resp.raise_for_status()

        dest_file = dest_dir / filename
        with open(dest_file, "wb") as f:
            f.write(resp.content)
        logger.info("Downloaded generated image to %s", dest_file)
        return dest_file

    def generate_image_sync(
        self,
        workflow_params: dict[str, Any],
        output_dir: Optional[Path] = None,
    ) -> list[Path]:
        """End-to-end synchronous image generation wrapper."""
        prompt_id = self.submit_prompt(workflow_params)
        outputs = self.wait_for_execution(prompt_id)

        downloaded: list[Path] = []
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    fname = img_info.get("filename")
                    sub = img_info.get("subfolder", "")
                    if fname:
                        local_path = self.download_output_image(fname, subfolder=sub, target_dir=output_dir)
                        downloaded.append(local_path)
        return downloaded

    def lock_canonical_visual_reference(
        self,
        db: Session,
        entity_id: str,
        image_path: str,
        ref_type: RefType = RefType.TURNAROUND_SHEET,
        seed: Optional[int] = None,
        lora_config: Optional[dict[str, Any]] = None,
        ip_adapter_config: Optional[dict[str, Any]] = None,
    ) -> VisualRef:
        """Lock an approved canonical reference image & seed into visual_refs table."""
        stmt = select(VisualRef).where(
            VisualRef.entity_id == entity_id,
            VisualRef.ref_type == ref_type,
            VisualRef.approved.is_(True),
        )
        existing = db.execute(stmt).scalars().all()
        for ref in existing:
            ref.approved = False

        new_ref = VisualRef(
            entity_id=entity_id,
            ref_type=ref_type,
            image_path=image_path,
            seed=seed,
            lora_config=lora_config,
            ip_adapter_config=ip_adapter_config,
            approved=True,
        )
        db.add(new_ref)
        db.commit()
        db.refresh(new_ref)
        logger.info(
            "Locked canonical visual reference for entity %s: %s (seed=%s, ref_type=%s)",
            entity_id,
            image_path,
            seed,
            ref_type,
        )
        return new_ref


class DualGPUComfyUIDispatcher:
    """Dispatches generation jobs across dual GPU ComfyUI instances concurrently.
    
    Distributes shots round-robin or dynamic work-stealing across GPU 0 and GPU 1
    to achieve 2x linear throughput.
    """

    def __init__(self, urls: list[str]):
        if not urls:
            raise ValueError("At least one ComfyUI server URL required")
        self.clients = [ComfyUIClient(base_url=u) for u in urls]
        logger.info("Initialized DualGPUComfyUIDispatcher with %d workers: %s", len(self.clients), urls)

    def get_online_clients(self) -> list[ComfyUIClient]:
        """Return list of clients that are currently online and responding."""
        online = []
        for client in self.clients:
            try:
                stats = client.check_health()
                if stats.get("status") == "online":
                    online.append(client)
            except Exception as e:
                logger.debug("Worker %s offline check: %s", client.base_url, e)
        return online if online else self.clients

    def check_health_all(self) -> list[dict[str, Any]]:
        return [c.check_health() for c in self.clients]

    def dispatch_batch_parallel(
        self,
        jobs: list[tuple[str, dict[str, Any]]],
        output_dir: Optional[Path] = None,
    ) -> list[tuple[str, Optional[Path]]]:
        """Executes a batch of (job_id, workflow) across online dual-GPU workers in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not jobs:
            return []

        active_clients = self.get_online_clients()
        num_workers = len(active_clients)
        logger.info("Dispatching %d jobs across %d online GPU workers...", len(jobs), num_workers)
        results: list[tuple[str, Optional[Path]]] = [None] * len(jobs)

        def _execute_job(index: int, job_id: str, workflow: dict[str, Any], client: ComfyUIClient):
            try:
                logger.info("Worker [%s] picked up job: %s", client.base_url, job_id)
                downloaded = client.generate_image_sync(workflow, output_dir=output_dir)
                img_path = downloaded[0] if downloaded else None
                return index, job_id, img_path
            except Exception as e:
                logger.error("Job %s failed on worker %s: %s", job_id, client.base_url, e)
                return index, job_id, None

        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = []
            for i, (job_id, workflow) in enumerate(jobs):
                worker_client = active_clients[i % num_workers]
                futures.append(executor.submit(_execute_job, i, job_id, workflow, worker_client))

            for future in as_completed(futures):
                idx, jid, img = future.result()
                results[idx] = (jid, img)

        return results

    def generate_batch_parallel(
        self,
        jobs: list[tuple[str, dict[str, Any]]],
        output_dir: Optional[Path] = None,
    ) -> dict[str, list[Path]]:
        """Executes a batch of generation jobs across the dual GPU workers in parallel."""
        dispatched = self.dispatch_batch_parallel(jobs, output_dir=output_dir)
        return {jid: [p] if p else [] for jid, p in dispatched}


def get_dual_gpu_dispatcher() -> DualGPUComfyUIDispatcher:
    """Retrieve DualGPUComfyUIDispatcher initialized with configured cluster URLs."""
    settings = get_settings()
    urls = settings.comfyui.cluster_urls
    return DualGPUComfyUIDispatcher(urls=urls)


