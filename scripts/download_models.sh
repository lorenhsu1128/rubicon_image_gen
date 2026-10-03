#!/usr/bin/env bash
# Download all model files into ComfyUI/models. Safe to re-run: finished files are skipped,
# partial files are resumed. File names / sizes were verified against the Hugging Face API (M0).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
MODELS=ComfyUI/models
HF=https://huggingface.co

# repo | path in repo | destination subdir | local file name (empty = same as path basename)
FILES=(
  "unsloth/Qwen-Image-Edit-2511-GGUF|qwen-image-edit-2511-Q4_K_M.gguf|unet|"
  "unsloth/Qwen-Image-Edit-2511-GGUF|qwen-image-edit-2511-Q3_K_M.gguf|unet|"
  "Comfy-Org/Qwen-Image_ComfyUI|split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors|text_encoders|"
  "Comfy-Org/Qwen-Image_ComfyUI|split_files/vae/qwen_image_vae.safetensors|vae|"
  "lightx2v/Qwen-Image-Edit-2511-Lightning|Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors|loras|"
  "tori29umai/QwenImageEdit2511_LoRA|QIE2511_ObjectExtraction_V10_dim1_1e-3-000120.safetensors|loras|"
  "DiffSynth-Studio/Qwen-Image-Edit-2511-ICEdit-LoRA|model.safetensors|loras|Qwen-Image-Edit-2511-ICEdit-LoRA.safetensors"
  "lrzjason/Anything2Real|anything2real_2601_A_final_patched.safetensors|loras|"
  "prithivMLmods/QIE-2511-Studio-DeLight|QIE-2511-Studio-DeLight-5000.safetensors|loras|"
  "berkerdooo/qwen-image-edit-2511-camera-angle-lora|qwen_edit_2511_camera_angle_lora.safetensors|loras|"
  "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA|qwen-image-edit-2511-multiple-angles-lora.safetensors|loras|"
  "prithivMLmods/QIE-2511-Object-Remover-v2|Qwen-Image-Edit-2511-Object-Remover-v2-9200.safetensors|loras|"
)

for entry in "${FILES[@]}"; do
  IFS='|' read -r repo path sub name <<<"$entry"
  name=${name:-$(basename "$path")}
  dest="$MODELS/$sub/$name"
  mkdir -p "$MODELS/$sub"
  url="$HF/$repo/resolve/main/$path"
  expected=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="x-linked-size:"{print $2} tolower($1)=="content-length:"{cl=$2} END{}' | tail -1)
  if [ -f "$dest" ] && [ -n "$expected" ] && [ "$(stat -c %s "$dest")" = "$expected" ]; then
    echo "[skip] $dest"; continue
  fi
  echo "[get ] $repo/$path -> $dest"
  curl -L --fail --retry 5 --retry-delay 5 -C - -o "$dest" "$url"
done
echo "All model files present."
