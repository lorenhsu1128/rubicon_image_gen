#!/usr/bin/env bash
# Install ComfyUI + custom nodes (pinned) into ./ComfyUI and a uv venv in the WSL home dir.
# Run inside WSL Ubuntu-24.04. Safe to re-run. Models: scripts/download_models.sh
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
VENV="${COMFY_VENV:-$HOME/.venvs/rubicon-comfy}"

COMFY_TAG=v0.38.0
# name | url | commit
NODES=(
  "ComfyUI-GGUF|https://github.com/city96/ComfyUI-GGUF.git|6ea2651"
  "ComfyUI-RMBG|https://github.com/1038lab/ComfyUI-RMBG.git|229529e"
)

[ -d ComfyUI/.git ] || git clone --branch "$COMFY_TAG" https://github.com/Comfy-Org/ComfyUI.git ComfyUI
git -C ComfyUI fetch --depth 1 origin tag "$COMFY_TAG" && git -C ComfyUI checkout -q "$COMFY_TAG"
for entry in "${NODES[@]}"; do
  IFS='|' read -r name url commit <<<"$entry"
  dir="ComfyUI/custom_nodes/$name"
  [ -d "$dir/.git" ] || git clone "$url" "$dir"
  git -C "$dir" fetch -q origin && git -C "$dir" checkout -q "$commit"
done

# Our own node pack + step-by-step templates (shown first under 擴充功能 in the template library)
ln -sfn ../../comfy_nodes/0_MechPipeline ComfyUI/custom_nodes/0_MechPipeline

[ -x "$VENV/bin/python" ] || uv venv --python 3.13 "$VENV"
source "$VENV/bin/activate"
# cu130 wheels include sm_120 (RTX 50 series / Blackwell)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
uv pip install -r ComfyUI/requirements.txt -r ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt
# onnxruntime-gpu is skipped: BiRefNet runs on torch, and its download repeatedly stalled here (M1)
grep -vE '^\s*(#|$)|onnxruntime-gpu' ComfyUI/custom_nodes/ComfyUI-RMBG/requirements.txt > /tmp/rmbg_requirements.txt
uv pip install -r /tmp/rmbg_requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
