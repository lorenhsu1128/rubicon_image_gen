# CLAUDE.md

本專案是「機甲部件出圖流水線」：以 Qwen-Image-Edit-2511（GGUF）＋ComfyUI 產生機甲全身圖、部件設計稿與給 TRELLIS.2 用的 RGBA 圖。完整規格見 `mech-pipeline-spec.md`，本檔是精簡版工作守則。

## 工作守則

- 對話一律用**繁體中文**；程式碼、識別字、註解用英文。
- **一次只做一個里程碑**（規格第 8 節），完成後停下來回報（做了什麼、產出路徑、問題、下一步），等使用者驗收；每個里程碑一個 git commit。
- **不准猜**：模型檔名、LoRA 觸發提示詞與強度、ComfyUI 節點類別與欄位名，一律從 model card、ComfyUI `/object_info`、外掛 README／範例 workflow 查證；查不到或矛盾就問使用者。
- 範圍外：訓練 LoRA、3D 生成、TRELLIS.2 的修改、網頁 UI、SAM3 自動分割。
- 每張產出圖旁要有同名 `.json` metadata，可用 `mechpipe rerun <json>` 重跑。

## 環境（WSL2 Ubuntu-24.04，比照 TRELLIS.2 專案）

- 程式碼與 `ComfyUI/`（含外掛、模型）都在本 repo 目錄；`ComfyUI/`、`runs/` 已在 `.gitignore`。
- ComfyUI 版本固定在 tag `v0.38.0`（`Comfy-Org/ComfyUI`）；外掛：`ComfyUI-GGUF`、`ComfyUI-RMBG`（只用 BiRefNet，**禁用 RMBG-2.0 權重**）。
- Python 環境放在 WSL 家目錄，**不要**放到 `/mnt/c`：
  - ComfyUI：`~/.venvs/rubicon-comfy`（uv，Python 3.13，torch cu130）
  - 不得修改其他專案的環境（例如 conda `trellis2`）。
- 從 Claude Code 的 Git Bash 呼叫 WSL 時要加 `MSYS_NO_PATHCONV=1`，否則 `/mnt/c/...` 會被改寫：
  `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu-24.04 --cd "/mnt/c/Users/loren/Documents/_projects/rubicon_image_gen" -- bash -lc '...'`
- 安裝／重建 ComfyUI 與外掛（版本固定）：`bash scripts/setup_comfyui.sh`
- 下載模型：`bash scripts/download_models.sh`（可重複執行，會續傳）
- 不要設定 `PYTORCH_CUDA_ALLOC_CONF`：會和 ComfyUI 自選的 `cudaMallocAsync` 衝突，導致無法啟動。
- 啟動 ComfyUI：Windows 雙擊 `start_comfyui.bat`，或在 WSL 執行 `bash scripts/start_comfyui.sh`；紀錄檔在 `~/.cache/rubicon/comfyui.log`。

## 硬體限制

- GPU：RTX 5070 Ti **Laptop，12GB 顯存**；WSL 記憶體 48GB。
- 與 TRELLIS.2 共用 GPU，出圖前要先關掉 TRELLIS.2。
- 一個 workflow 只放 Qwen-Image-Edit 主流程；去背、分割另開 workflow。
- 每次只掛 Lightning LoRA ＋ 最多一顆任務 LoRA。
- 輸出尺寸 1024×1024；部件裁切圖先補白邊成正方形再縮放。

## 取樣參數（依官方範本 `image_qwen_image_edit_2511.json`）

- 草稿：Lightning 4 步 LoRA，4 步，CFG 1.0
- 定稿：不掛 Lightning，40 步，CFG 4.0（範本中由 Switch 節點決定；KSampler 面板上的 CFG 3 會被覆蓋）
- 共通：euler／simple、`ModelSamplingAuraFlow` shift 3.1、`CFGNorm` 1.0、`FluxKontextMultiReferenceLatentMethod` = `index_timestep_zero`
- UNet：預設 `qwen-image-edit-2511-Q4_K_M.gguf`，A/B 比較 `Q3_K_M`

## Workflow 替換規則

- 依「節點標題（`_meta.title`）＋欄位名稱」替換，不要用節點 ID。
- 每個 workflow 都保存 `*.api.json` 與 `*.ui.json`，兩份的標題要一致。
