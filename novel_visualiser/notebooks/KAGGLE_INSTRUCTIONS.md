# 🚀 Kaggle / Google Colab T4 Setup Guide for ComfyUI

This guide explains how to run **ComfyUI** with all required models on Kaggle or Colab for the **Novel Visualiser** project.

---

## ⚡ 1-Click Method: Upload Notebook to Kaggle
1. Open [Kaggle](https://www.kaggle.com/) -> Click **+ Create** -> **New Notebook**.
2. Click **File** -> **Import Notebook** -> Select `notebooks/ComfyUI_Colab_Kaggle_T4.ipynb`.
3. In the right panel under **Notebook settings**:
   - **Accelerator**: Select **GPU T4 x2** (or **GPU P100**).
   - **Internet**: Toggle **Internet ON** (Required to download models).
4. Run all cells in order.
5. In Cell 4, copy the generated URL (e.g. `https://random-words.trycloudflare.com`).
6. Paste it into your local `novel_visualiser/.env`:
   ```env
   COMFYUI_URL=https://random-words.trycloudflare.com
   ```

---

## 📋 Manual Shell Commands (If running cell-by-cell in Kaggle)

### Cell 1: Install ComfyUI & Fast Downloader
```bash
!apt-get update -qq && apt-get install -y -qq aria2
!git clone https://github.com/comfyanonymous/ComfyUI.git /kaggle/working/ComfyUI
%cd /kaggle/working/ComfyUI
!pip install -q torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
!pip install -q -r requirements.txt
!pip install -q httpx pyngrok timm einops transformers
```

### Cell 2: Install Custom Nodes (IPAdapter Plus & Florence-2)
```bash
%cd /kaggle/working/ComfyUI/custom_nodes
!git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
!git clone https://github.com/kijai/ComfyUI-Florence2.git
%cd /kaggle/working/ComfyUI/custom_nodes/ComfyUI-Florence2
!pip install -q -r requirements.txt
%cd /kaggle/working/ComfyUI
```

### Cell 3: Download Models (Parallel multi-connection aria2c ~1 min)
```bash
# Checkpoint (6.9 GB) -> Wai_Illustrious_v16.safetensors
!aria2c -c -x 16 -s 16 -k 1M "https://huggingface.co/seraphimzz/checkpoints/resolve/main/waiIllustriousSDXL_v160.safetensors" -d "/kaggle/working/ComfyUI/models/checkpoints" -o "Wai_Illustrious_v16.safetensors"

# IP-Adapter Plus SDXL (808 MB)
!aria2c -c -x 8 -s 8 -k 1M "https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors" -d "/kaggle/working/ComfyUI/models/ipadapter" -o "ip-adapter-plus_sdxl_vit-h.safetensors"

# CLIP-Vision ViT-H (2.4 GB)
!aria2c -c -x 16 -s 16 -k 1M "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" -d "/kaggle/working/ComfyUI/models/clip_vision" -o "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"

# VAE SDXL Fix (319 MB)
!aria2c -c -x 4 -s 4 -k 1M "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl.vae.safetensors" -d "/kaggle/working/ComfyUI/models/vae" -o "sdxl.vae.safetensors"
```

### Cell 4: Launch ComfyUI & Cloudflare Tunnel
```python
import subprocess, time, re

!wget -nc -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /tmp/cloudflared
!chmod +x /tmp/cloudflared

comfy_proc = subprocess.Popen([
    "python", "main.py",
    "--listen", "127.0.0.1",
    "--port", "8188",
    "--preview-method", "auto",
    "--lowvram"
])
time.sleep(5)

cf_proc = subprocess.Popen([
    "/tmp/cloudflared", "tunnel",
    "--url", "http://127.0.0.1:8188",
    "--metrics", "127.0.0.1:4444"
], stderr=subprocess.PIPE, text=True)

print("Starting Cloudflare tunnel...")
for line in cf_proc.stderr:
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        print(f"\n🎉 COMFYUI PUBLIC URL: {match.group(0)}")
        print(f"👉 Set this in your local novel_visualiser/.env:")
        print(f"COMFYUI_URL={match.group(0)}\n")
        break

try:
    comfy_proc.wait()
except KeyboardInterrupt:
    comfy_proc.kill()
    cf_proc.kill()
```

