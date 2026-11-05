"""
Test generating an image using the user's exact IPAdapter workflow and Sunny.jfif reference
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
    print(" NOVEL VISUALISER — LIVE KAGGLE GPU WORKFLOW TEST")
    print("=======================================================\n")

    client = ComfyUIClient()
    health = client.check_health()
    print(f"ComfyUI Health: {health.get('status')} at {client.base_url}")
    print(f"Active GPU: {health.get('gpu_name')} ({health.get('vram_free_gb')} GB Free)\n")

    if health.get("status") != "online":
        print("ERROR: ComfyUI is not online! Check the Cloudflare tunnel URL.")
        return

    # 1. Upload Sunny reference image
    ref_image_path = Path("novel_visualiser/assets/canonical_refs/Sunny_portrait.png")
    if not ref_image_path.exists():
        ref_image_path = Path("AI Studio Help/Sunny.jfif")

    print(f"[1/3] Uploading reference portrait '{ref_image_path.name}' to ComfyUI server...")
    server_filename = client.upload_image(ref_image_path)
    print(f"      Uploaded as '{server_filename}' on ComfyUI server.\n")

    # 2. Build the API Prompt Payload based on 2_NV_Character_Consistency_IPAdapter.json
    print("[2/3] Building prompt payload with IPAdapter character consistency...")
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
        "11": {
            "class_type": "LoadImage",
            "inputs": {
                "image": server_filename
            }
        },
        "9": {
            "class_type": "IPAdapterModelLoader",
            "inputs": {
                "ipadapter_file": "sdxl/plus/ip-adapter-plus_sdxl_vit-h.safetensors"
            }
        },
        "10": {
            "class_type": "CLIPVisionLoader",
            "inputs": {
                "clip_name": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"
            }
        },
        "12": {
            "class_type": "IPAdapterAdvanced",
            "inputs": {
                "model": ["1", 0],
                "ipadapter": ["9", 0],
                "image": ["11", 0],
                "clip_vision": ["10", 0],
                "weight": 0.70,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "embeds_scaling": "V only",
                "start_at": 0.0,
                "end_at": 1.0
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": "masterpiece, best quality, ultra-detailed, anime illustration, light novel illustration, 1boy, Sunny, slender 16-year-old youth with messy jet-black hair, pale skin, dark eyes, wearing dark simple coat, sitting on a rusted bench outside a suburban police station, holding paper cup of coffee, cold bleak dawn light, cinematic framing"
            }
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": "worst quality, low quality, bad anatomy, bad hands, extra digits, blurry, text, watermark, deformed, bad eyes, disfigured"
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
                "model": ["12", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": 421098,
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
                "filename_prefix": "NovelVisualizer/Live_Sunny_IPAdapter"
            }
        }
    }

    # 3. Submit and execute on Kaggle GPU
    print("[3/3] Submitting prompt to remote ComfyUI (Kaggle dual Tesla T4 GPU)...")
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
