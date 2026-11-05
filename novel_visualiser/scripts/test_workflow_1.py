"""
Test generating an image using Workflow 1: 1_NV_Scene_Generator_and_Upscaler.json
on the live remote Kaggle T4 GPU ComfyUI instance.
"""

import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.comfyui_client import ComfyUIClient


def main():
    print("=======================================================")
    print(" NOVEL VISUALISER — LIVE TEST: WORKFLOW 1 (SCENE + UPSCALE)")
    print("=======================================================\n")

    client = ComfyUIClient()
    health = client.check_health()
    print(f"ComfyUI Health: {health.get('status')} at {client.base_url}")
    print(f"Active GPU: {health.get('gpu_name')} ({health.get('vram_free_gb')} GB Free)\n")

    if health.get("status") != "online":
        print("ERROR: ComfyUI is not online!")
        return

    # Build the API Prompt Payload based on 1_NV_Scene_Generator_and_Upscaler.json
    print("[1/2] Building prompt payload for Scene Generator + 4x AnimeSharp Upscaler...")
    prompt_payload = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "sdxl/illustrious/waiNSFWIllustrious_v110.safetensors"
            }
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "sdxl_vae_fp16_fix.safetensors"
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": "masterpiece, best quality, absurdres, anime illustration, light novel illustration, 1boy, Sunny, slender 16-year-old youth with messy jet-black hair, pale skin, dark eyes, wearing dark tattered coat, sitting on a rusted metal bench outside an austere police station, holding paper cup of coffee, cold bleak dawn light, cinematic framing"
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": "worst quality, low quality, bad anatomy, bad hands, extra digits, blurry, text, watermark, deformed, bad eyes, disfigured, realistic, photo"
            }
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 832,
                "height": 1216,
                "batch_size": 1
            }
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": 987654,
                "steps": 24,
                "cfg": 6.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0
            }
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["6", 0],
                "vae": ["2", 0]
            }
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["7", 0],
                "filename_prefix": "NovelVisualizer/NV_Base_Scene"
            }
        },
        "9": {
            "class_type": "UpscaleModelLoader",
            "inputs": {
                "model_name": "anime/4x-AnimeSharp.pth"
            }
        },
        "10": {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["9", 0],
                "image": ["7", 0]
            }
        },
        "11": {
            "class_type": "ImageScaleBy",
            "inputs": {
                "image": ["10", 0],
                "upscale_method": "lanczos",
                "scale_by": 0.5
            }
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["11", 0],
                "filename_prefix": "NovelVisualizer/NV_Upscaled_2x"
            }
        }
    }

    # Submit and execute on Kaggle GPU
    print("[2/2] Submitting prompt to remote ComfyUI (Kaggle dual Tesla T4 GPU)...")
    out_dir = Path("novel_visualiser/output/live_tests")
    out_dir.mkdir(parents=True, exist_ok=True)

    start_t = time.time()
    downloaded = client.generate_image_sync(prompt_payload, output_dir=out_dir)
    elapsed = time.time() - start_t

    if downloaded:
        print(f"\nSUCCESS! Generated {len(downloaded)} image(s) in {elapsed:.2f} seconds!")
        for img_path in downloaded:
            print(f"  - Downloaded to: {img_path} ({img_path.stat().st_size} bytes)")
    else:
        print(f"\nNo images downloaded. Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()

