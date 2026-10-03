# M1 報告：ComfyUI 安裝與官方 2511 範本（GGUF）

日期：2026-10-04

## 環境

| 項目 | 版本 |
|---|---|
| 平台 | WSL2 Ubuntu-24.04.4（kernel 6.6.114.1），WSL 記憶體 48GB |
| GPU | RTX 5070 Ti Laptop 12GB，驅動 617.14 |
| ComfyUI | `v0.38.0`（6b747c0），前端 1.53.6 |
| Python / PyTorch | 3.13.15 / 2.14.1+cu130（含 sm_120），venv：`~/.venvs/rubicon-comfy` |
| ComfyUI-GGUF | 6ea2651 |
| ComfyUI-RMBG | 229529e（`BiRefNetRMBG` 節點可正常載入） |

重建方式：`bash scripts/setup_comfyui.sh` → `bash scripts/download_models.sh`。

## 測試 workflow

`workflows/m1_template_2511.api.json`：由官方範本 `image_qwen_image_edit_2511.json` 改寫，`UNETLoader` 換成 `UnetLoaderGGUF`，其餘節點與參數相同；所有節點類別與欄位都已用 `/object_info` 查證。輸入圖片為官方範本附的 `leather_sofa.png` 與 `texture_fur.png`，提示詞也沿用範本。

## 實測結果（1024×1024）

| 測試 | UNet | 模式 | 總時間 | 取樣速度 | 顯存峰值 |
|---|---|---|---|---|---|
| q4_draft_cold | Q4_K_M | Lightning 4 步，CFG 1 | 195 秒 | 約 11 秒/步 | 10.4 GB |
| q4_draft_warm | Q4_K_M | 同上 | **50 秒** | 約 11 秒/步 | 10.5 GB |
| q3_draft_cold | Q3_K_M | 同上 | 150 秒 | 約 11 秒/步 | 10.5 GB |
| q3_draft_warm | Q3_K_M | 同上 | 51 秒 | 約 11 秒/步 | 10.6 GB |
| q4_final | Q4_K_M | 40 步，CFG 4 | **16.5 分鐘** | 約 23 秒/步 | 10.4 GB |

- 「cold」是先 `/free` 卸載所有模型後再跑；「warm」是模型已在記憶體中。
- 顯存峰值是 `nvidia-smi` 的整張卡用量（ComfyUI 本身會留 1–2GB 的餘裕）。
- 產出圖：`ComfyUI/output/m1/`（`q4_draft_cold_00001_.png` 等）。

## 觀察

1. **Q4_K_M 與 Q3_K_M 暖機速度幾乎一樣**（50 秒 vs 51 秒）。Q4_K_M 有約 4GB 卸載到 RAM，Q3_K_M 也還有 0.7GB 卸載，瓶頸不在卸載量。Q3_K_M 只省冷啟動的讀檔時間。這張測試圖兩者的畫質差異很小（毛皮斑點的形狀略有不同），**建議維持 Q4_K_M 為預設**，等 M2 用機甲線稿再比一次細節（刻線、編號標記）。
2. **冷啟動多出的約 150 秒，主要是從 `/mnt/c` 讀取約 22GB 的模型**（約 150MB/s）。批次出圖時模型會常駐記憶體，只有第一張要付這個成本；換任務 LoRA 不需重讀 UNet。目前先不搬到 ext4，若 M2 之後切換頻繁再處理。
3. **定稿模式（40 步、CFG 4）一張約 16.5 分鐘**，因為 CFG > 1 時每一步要算兩次。這在 12GB 筆電上偏慢，建議定稿模式只用於最後挑選出的少數圖片，日常一律用草稿模式。
4. 草稿模式的毛皮呈現清晰的點狀斑紋；定稿模式比較柔和，更接近參考的雪豹毛皮。

## 遇到的問題與處理

- **`PYTORCH_CUDA_ALLOC_CONF=backend:native` 會讓 ComfyUI 無法啟動**（torch INTERNAL ASSERT：allocator backend 不一致）。TRELLIS.2 啟動腳本的這段設定不適用，已從 `scripts/start_comfyui.sh` 移除，交給 ComfyUI 自己選（`cudaMallocAsync`）。
- **`onnxruntime-gpu`（235MB）從 PyPI 下載兩次都卡住**。BiRefNet 走 PyTorch，不需要它，因此安裝 RMBG 相依套件時排除了它（已寫在 `setup_comfyui.sh`）。若之後要用 RMBG 外掛中依賴 ONNX GPU 的其他模型，再補裝。
- 從 Git Bash 呼叫 `wsl.exe` 時要設 `MSYS_NO_PATHCONV=1`，否則 `/mnt/c/...` 路徑會被改寫（已記在 `CLAUDE.md`）。

## 尚未完成（移到 M2）

- `*.ui.json`：M1 的 workflow 只是冒煙測試，M2 正式建立 `s1_master` 時再一起產生 UI 格式並核對標題。
