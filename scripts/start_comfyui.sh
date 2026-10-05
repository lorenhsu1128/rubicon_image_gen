#!/usr/bin/env bash
# Launch ComfyUI inside WSL (venv in the WSL home dir, code and models in this repo).
set -e
cd "$(dirname "$(readlink -f "$0")")/.."
VENV="${COMFY_VENV:-$HOME/.venvs/rubicon-comfy}"
source "$VENV/bin/activate"
export COMFY_PORT="${COMFY_PORT:-8188}"
# 0.0.0.0 = reachable from other machines on the LAN; set COMFY_LISTEN=127.0.0.1 for local only
export COMFY_LISTEN="${COMFY_LISTEN:-0.0.0.0}"
export PYTHONUNBUFFERED=1
# Do not set PYTORCH_CUDA_ALLOC_CONF here: ComfyUI picks the allocator backend itself
# (cudaMallocAsync by default) and an override makes torch fail with an INTERNAL ASSERT.
# Fixed mmap threshold so freed tensor buffers return to the OS instead of fragmenting the heap
export MALLOC_MMAP_THRESHOLD_=1048576 MALLOC_TRIM_THRESHOLD_=67108864

# The GPU is shared with TRELLIS.2; warn if something else already holds VRAM
USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ "${USED_MB:-0}" -gt 1500 ]; then
  echo "警告：GPU 已被占用 ${USED_MB} MiB（TRELLIS.2 是否還開著？），出圖可能變慢或顯存不足。"
fi

echo "============================================================"
echo " ComfyUI 啟動中"
echo " 本機：      http://127.0.0.1:${COMFY_PORT}"
echo " 監聽位址：  ${COMFY_LISTEN}（區網其他電腦用本機 IP 連線）"
echo " 關閉服務：  在此視窗按 Ctrl+C"
echo "============================================================"
# Server output goes to a log file and this window only tails it: selecting text in a Windows
# console (QuickEdit) pauses its output, which would otherwise block the server mid-request.
LOG_DIR="$HOME/.cache/rubicon"
LOG="$LOG_DIR/comfyui.log"
mkdir -p "$LOG_DIR"
[ -f "$LOG" ] && mv -f "$LOG" "$LOG.prev"
echo " 伺服器紀錄： $LOG"
python ComfyUI/main.py --listen "$COMFY_LISTEN" --port "$COMFY_PORT" "$@" > "$LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' INT TERM EXIT
tail -n +1 -F --pid="$SERVER_PID" "$LOG" 2>/dev/null
rc=0
wait "$SERVER_PID" || rc=$?
echo "伺服器已停止（結束代碼 $rc），詳細紀錄見 $LOG"
