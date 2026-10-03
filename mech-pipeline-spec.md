# 機甲部件出圖流水線 — 實作規格書（給 Claude Code）

> 本文件是給 Claude Code 的工作規格。請先完整讀完，再從「里程碑 M0」開始。

---

## 0. 工作守則（請嚴格遵守）

1. **溝通語言**：所有對話一律使用繁體中文；程式碼、識別字、註解用英文。
2. **分階段執行**：一次只做一個里程碑（見第 8 節）。每完成一個里程碑就**停下來**，回報結果、附上產出圖的路徑，等我驗收後再繼續。
3. **不准猜**：模型檔名、LoRA 觸發提示詞、ComfyUI 節點類別名稱、節點輸入欄位名稱，一律要從以下來源**查證**：
   - Hugging Face / Civitai 的 model card
   - ComfyUI 的 `/object_info` API（查節點與欄位）
   - 外掛的 README 或範例 workflow
   查不到或互相矛盾時，**停下來問我**，不要自己編。
4. **環境隔離**：不得修改本機其他專案的 Python / conda 環境（例如 TRELLIS.2 的 `trellis2`）。本專案全部跑在 WSL2 Ubuntu-24.04 內；ComfyUI 與 Python 客戶端各用一個 `uv` venv，放在 WSL 家目錄（`~/.venvs/`），不放在 `/mnt/c`。
5. **範圍限制**：本專案**不訓練任何 LoRA**、**不做 3D 轉換**。3D 由另一台機器上的 TRELLIS.2 fork 處理，我們只負責交出合格的圖片。
6. **可重現**：每張產出圖都要有對應的 metadata（種子、模型、LoRA 與強度、提示詞、輸入圖路徑），能一鍵重跑。

---

## 1. 目標

建立一條以 **Qwen-Image-Edit-2511（GGUF）** 為核心的「圖＋文生圖」流水線：

```
新機甲設計（草稿/參考圖＋文字）
  → S1 全身基準圖（標準化：正面 A-pose ＋ 3/4 視角）
  → S2 部件抽取（每個部件一張，線稿設計稿風格）
  → S3 3D 輸入版（渲染風格、平光、3/4 視角、無投影）
  → S4 去背輸出 RGBA PNG
  → 交付給 TRELLIS.2（另一台機器）
```

產出兩套圖：
- **設計稿**：賽璐璐線稿風格，給人看、定案用。
- **3D 輸入版**：渲染風格 RGBA，給 TRELLIS.2 吃。

---

## 2. 硬體與平台

| 項目 | 規格 |
|---|---|
| 工作機 GPU | NVIDIA RTX 5070 Ti **Laptop**（**12GB** VRAM，Blackwell sm_120）— M0 實測 |
| 作業系統 | Windows 11 ＋ WSL2 Ubuntu-24.04；**ComfyUI 跑在 WSL 內**（比照 TRELLIS.2 專案的做法） |
| 系統記憶體 | 主機 64GB；`.wslconfig` 設 `memory=48GB` 給 WSL（文字編碼器與部分 UNet 會卸載到 CPU） |
| 後端 | ComfyUI（git clone，`ComfyUI/` 於本 repo 內但不納入版控），預設 `http://127.0.0.1:8188`；`networkingMode=mirrored`，Windows 可直接連 |

**顯存規則**：
- 同一個 workflow 只放「Qwen-Image-Edit 主流程」，**分割（SAM 類）與去背另開獨立 workflow**，不要串在同一張圖裡，12GB 會爆。
- 與 TRELLIS.2 共用同一張 GPU，**出圖時 TRELLIS.2 必須關閉**。
- 每次只掛 **Lightning LoRA ＋ 最多一顆任務 LoRA**。
- 輸出尺寸固定為 1024×1024，或與輸入同比例並對齊 16 的倍數，避免畫面位移。

---

## 3. 模型與外掛清單

### 3.1 ComfyUI 外掛（custom nodes）

| 外掛 | 用途 | 備註 |
|---|---|---|
| `city96/ComfyUI-GGUF` | 載入 GGUF 主模型 | 必要 |
| `lrzjason/Comfyui-QwenEditUtils` | 多參考圖、每張圖分別設定 | 官方 2511 範本只用原生 `TextEncodeQwenImageEditPlus`，先不裝；不夠再裝 |
| `1038lab/ComfyUI-RMBG` | BiRefNet 去背、（選用）SAM3 分割 | **只用 BiRefNet 系列**；RMBG-2.0 權重禁止商用，不要用 |
| `jtydhr88/ComfyUI-qwenmultiangle` | 產生 Multiple-Angles LoRA 的提示詞 | 選用，M5 才需要 |

### 3.2 基礎模型

| 檔案 | 放置位置 | 備註 |
|---|---|---|
| `qwen-image-edit-2511-Q4_K_M.gguf`（unsloth/Qwen-Image-Edit-2511-GGUF，13.24GB） | `models/unet/` | 預設（超過 12GB 顯存，會部分卸載到 RAM）。與 `Q3_K_M`（9.92GB）做 A/B 比較 |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors`（Comfy-Org/Qwen-Image_ComfyUI） | `models/text_encoders/` | 依 ComfyUI 官方 2511 教學 |
| `qwen_image_vae.safetensors`（Comfy-Org/Qwen-Image_ComfyUI） | `models/vae/` | 同上 |
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors`（lightx2v） | `models/loras/` | 4 步，CFG 1.0 |
| （選用）Qwen-Image-2512 GGUF | `models/unet/` | 只在「純文字生圖、沒有任何參考圖」時使用 |

**取樣參數**：
- 草稿模式：Lightning，4 步，CFG 1.0
- 定稿模式：不掛 Lightning，40 步，CFG 4.0（依官方範本 `image_qwen_image_edit_2511.json` 的 Switch (Steps)/Switch (CFG) 節點；KSampler 面板上顯示的 CFG 3 會被覆蓋。其他：euler／simple、`ModelSamplingAuraFlow` shift 3.1、`CFGNorm` 1.0、`FluxKontextMultiReferenceLatentMethod` = `index_timestep_zero`）

### 3.3 任務 LoRA（全部為現成、不訓練）

> 每顆 LoRA 的**觸發提示詞、建議強度、相容性**都必須從 model card 抄下來，寫進 `config/loras.yaml`，並在 `docs/lora-notes.md` 記錄來源網址。不可憑印象填寫。

| 代號 | Hugging Face repo | 用在 | 待查證事項 |
|---|---|---|---|
| `extract` | `tori29umai/QwenImageEdit2511_LoRA` 內的 `QIE2511_ObjectExtraction_V10_dim1_1e-3-000120.safetensors` | S2 | 已查證（M0）：同一顆 LoRA 同時負責「單色遮罩區重建＋抽取＋白底」，即原規格的 `amodal`；遮罩為**不透明單色**（建議原色）；提示詞為單張圖版本（"From picture1, extract only the main object…"） |
| `icedit` | `DiffSynth-Studio/Qwen-Image-Edit-2511-ICEdit-LoRA` | S2（B 路線） | **能否在 ComfyUI＋GGUF 下載入並正常作用**（原範例是 DiffSynth 框架） |
| `anything2real` | `lrzjason/Anything2Real`（Qwen Edit 2511 的 2601 A 版） | S3 | 檔名、提示詞格式、建議強度 |
| `delight` | `prithivMLmods/QIE-2511-Studio-DeLight` | S3 | 提示詞 |
| `cam_object` | `berkerdooo/qwen-image-edit-2511-camera-angle-lora` | S3（單一部件轉角度） | 以「度數」描述鏡頭移動的提示詞格式；訓練解析度 512 的影響 |
| `cam_full` | `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` | S1（全身轉角度） | `<sks>` 開頭的提示詞格式 |
| `remover` | `prithivMLmods/QIE-2511-Object-Remover-v2` | 修補 | 提示詞（範例用紅色標記要移除的物件） |
| `upscale` | `prithivMLmods/Qwen-Image-Edit-2511-Unblur-Upscale` | 修補 | 提示詞 |

---

## 4. 專案結構

```
rubicon_image_gen/           # 本 repo 根目錄即專案根目錄
├─ ComfyUI/                  # ComfyUI 本體＋外掛＋模型（.gitignore 排除）
├─ scripts/                  # download_models.sh、start_comfyui.sh 等
├─ start_comfyui.bat         # Windows 端啟動（呼叫 WSL）
├─ CLAUDE.md                 # 由你依本規格產生的精簡版工作守則
├─ README.md
├─ pyproject.toml
├─ config/
│  ├─ settings.yaml          # ComfyUI 位址、輸出路徑、預設取樣參數
│  ├─ loras.yaml             # 每顆 LoRA：檔名、觸發提示詞、建議強度、來源網址
│  └─ parts.yaml             # 部件清單（見第 6 節）
├─ workflows/                # ComfyUI API 格式 JSON（*.api.json）
│  ├─ s1_master.api.json
│  ├─ s2_extract.api.json
│  ├─ s2_icedit.api.json
│  ├─ s3_render.api.json
│  ├─ s3_angle.api.json
│  └─ s4_rmbg.api.json
├─ prompts/                  # 提示詞範本（Jinja2 或 str.format）
├─ mechpipe/                 # Python 套件
│  ├─ comfy_client.py        # 上傳圖片、送出 prompt、輪詢、下載結果
│  ├─ workflow_patch.py      # 依「節點標題」替換輸入
│  ├─ stages.py              # S1–S4 各階段
│  ├─ crop.py                # 依 bbox 裁切、塗粉紅遮罩
│  ├─ contact_sheet.py       # 產生 HTML 對照表供驗收
│  └─ cli.py                 # 指令列入口
├─ assets/
│  ├─ style_refs/            # 系列風格參考圖（2–3 張定稿）
│  └─ examples/              # ICEdit 用的示範配對（全身 → 部件）
└─ runs/
   └─ <mech_id>/<stage>/<part>/<timestamp>_<seed>.png  ＋ 同名 .json
```

### 4.1 Workflow 替換規則（重要）

- 不要用節點 ID 去替換輸入（ID 會因為重新匯出而改變）。
- 在 ComfyUI 中為需要替換的節點**設定固定標題**，例如：`IN_IMAGE_1`、`IN_IMAGE_2`、`IN_IMAGE_3`、`PROMPT_POS`、`SEED`、`LORA_TASK`、`LORA_TASK_STRENGTH`、`STEPS`、`CFG`、`OUT_SIZE`。
- `workflow_patch.py` 依標題找節點並替換值。
- 每個 workflow 同時保存一份 UI 格式（`*.ui.json`），方便我在 ComfyUI 介面裡打開檢查。

### 4.2 Metadata

每張輸出圖旁邊放同名 `.json`，內容至少包括：
`mech_id`、`stage`、`part`、`seed`、`unet`、`loras`（名稱＋強度）、`steps`、`cfg`、`prompt`、`input_images`（路徑）、`workflow`（檔名）、`timestamp`、`comfy_prompt_id`。

並提供 `mechpipe rerun <json>` 指令可依 metadata 重跑。

---

## 5. 各階段規格

> 提示詞範本以英文撰寫。`{...}` 為變數。使用任務 LoRA 時，**以 model card 的固定提示詞為主**，下面的範本只在不掛任務 LoRA 時使用，或作為補充描述。

### S1 全身基準圖

- **輸入**：草稿或既有設計圖（`IN_IMAGE_1`），選用風格參考圖（`IN_IMAGE_2`）。
- **輸出**：`master_front.png`（正面 A-pose）、`master_34.png`（3/4 視角）。
- **提示詞範本（正面）**：
  ```
  Redraw this mech as a full-body front view in a neutral A-pose,
  arms held slightly away from the torso so the thighs are fully visible,
  orthographic, plain light grey background, no ground shadow,
  keep exactly the same armor shapes, {color_scheme},
  {markings}, and line-art style.
  ```
- **3/4 視角**：先試基底模型（2511 已內建視角生成能力），效果不足再掛 `cam_full`。
- **流程**：每次產生 4 個種子，輸出對照表給我挑選；我選定後該張複製為 `runs/<mech_id>/master/` 下的正式版本。

### S2 部件抽取（設計稿風格）

依 `config/parts.yaml` 逐一處理。實作兩條路線，M3 會做 A/B 比較。

**A 路線：裁切 → 單色遮罩 → 物件抽取（`extract`）**
1. 依 `parts.yaml` 中的 bbox 從 `master_front.png` 裁切該部位（四周保留 10–15% 邊距），補白邊成正方形後縮放到 1024×1024。
2. 若 `parts.yaml` 標記了遮擋區（`occluders`），在裁切圖上把遮擋物塗成**不透明單色**（依 model card，建議原色，且避開機甲本身的配色）。
3. 處理後的裁切圖放 `IN_IMAGE_1`，以 model card 的固定提示詞跑 `extract`（一顆 LoRA 同時完成補完與抽取）。
4. （M3 額外變體）再把全身基準圖放 `IN_IMAGE_2` 當參考，比較是否更能保留配色與標記。

**B 路線：ICEdit 示範**
- `IN_IMAGE_1`：示範全身圖（`assets/examples/<pair>/full.png`）
- `IN_IMAGE_2`：示範部件圖（`assets/examples/<pair>/<part>.png`）
- `IN_IMAGE_3`：目標機甲的全身基準圖
- 提示詞依 model card。

**部件圖通用要求（寫進提示詞或作為驗收標準）**：
- 只有該部件，獨立擺放。
- 與全身圖相同的比例、裝甲形狀、配色、刻線與標記。
- 斷面是**乾淨的機械關節座**，不要垂掛的斷線。
- 淺灰或白色純色背景，無地面陰影。

### S3 3D 輸入版

依序（每步一個 workflow、一顆任務 LoRA）：
1. **線稿轉渲染**：`anything2real`（2601 A 版）。若出現擅自加入的環境或景深，先降低強度，不要加大。
2. **統一打光**：`delight`。
3. **轉 3/4 視角**：`cam_object`。輸入必須是步驟 1、2 處理後的渲染圖，不要直接用線稿。

補充提示詞範本（不掛 LoRA 時的對照組）：
```
Convert this line-art mech part into a clean 3D render,
same shape and colors, no black outlines, flat even studio lighting,
no cast shadows, no ground shadow, 3/4 view, plain light grey background.
```

### S4 去背與交付

- 用 BiRefNet（`ComfyUI-RMBG`）輸出 RGBA PNG，獨立 workflow。
- 檢查 alpha：灰色裝甲與細小結構不能被吃掉；若被吃掉，改試 BiRefNet 的其他變體或調整參數，並回報給我。
- 交付資料夾：`runs/<mech_id>/deliver/`，檔名 `<mech_id>_<PART>.png`，另附 `manifest.json` 列出每個部件的來源 metadata。

### 修補（按需）

- `remover`：清除斷線、地面陰影等不要的東西。
- `upscale`：Lightning 讓編號標記變糊時使用。

---

## 6. 部件清單（`config/parts.yaml` 初版）

```yaml
parts:
  - id: HEAD
  - id: TORSO
  - id: BACKPACK_WEAPON
  - id: SHOULDER_L
  - id: SHOULDER_R
  - id: ARM_L
  - id: ARM_R
  - id: THIGH_L
  - id: THIGH_R
  - id: SHIN_FOOT_L
  - id: SHIN_FOOT_R

# 每台機甲另有 parts.<mech_id>.yaml，內容為：
#   bbox: [x, y, w, h]        # 在 master_front.png 上的位置，先由我手動標
#   occluders: [[x,y,w,h],..] # 選用，遮擋區
#   notes: "white '7' marking on thigh armor, dark grey"  # 必須保留的特徵
```

- 先由我手動提供 bbox；**SAM3 自動分割列為之後的選配**，不在初期範圍。
- `notes` 欄位會被代入提示詞，用來明確描述要保留的配色與標記。
- 左右部件分開生成，不要自動鏡像。

---

## 7. 驗收用對照表

`contact_sheet.py` 產生一個靜態 HTML，每列一個部件，欄位為：
裁切參考圖 ｜ 各種子或各路線的結果 ｜ metadata 摘要。

方便我一眼比較、圈選。HTML 不需要任何外部資源，雙擊即可開啟。

---

## 8. 里程碑

| 編號 | 內容 | 完成條件 |
|---|---|---|
| **M0** | 環境盤點：查 VRAM、系統記憶體、磁碟空間；列出要下載的所有檔案、大小與下載網址 | 交出清單給我確認，**尚未下載任何東西** |
| **M1** | 在 WSL 內安裝 ComfyUI 與外掛、下載基礎模型；跑通官方 2511 範本（GGUF＋Lightning） | 一張成功的編輯結果圖，以及記錄的生成時間與峰值顯存 |
| **M2** | 建立 Python 客戶端、workflow 替換機制、metadata、`rerun`；完成 S1 | 用我提供的圖產出 4 張正面基準圖與對照表 |
| **M3** | 查證並記錄所有 S2 LoRA；以 `THIGH_L` 為例跑 A、B 兩條路線 | 對照表，並附上你的觀察：配色、標記、斷面，哪條路線較穩 |
| **M4** | 依我選定的路線，批次跑完整台機甲的所有部件 | 全部部件的對照表 |
| **M5** | 完成 S3（渲染→打光→轉角度）與 S4 去背 | `deliver/` 資料夾與 `manifest.json` |
| **M6** | 整理 README、CLI 使用說明、常見錯誤排除 | 我能自己從頭跑一台新機甲 |

每個里程碑結束時，請回報：做了什麼、產出路徑、遇到的問題、下一步建議。

---

## 9. 已知風險與待驗證事項

- **ICEdit LoRA** 的範例基於 DiffSynth-Studio 框架，在 ComfyUI＋GGUF 下是否可用未知。M3 若無法載入，回報後只保留 A 路線。
- **2509 的 LoRA** 不保證適用於 2511；本清單以標示 2511 或兩版相容者為主。
- **GGUF 量化**可能損失細節（刻線、小型文字標記），M1 之後要與 Q3_K_M 做比較（12GB 顯存下 Q5_K_S 只會更慢）。
- **模型放在 `/mnt/c`**：WSL 讀取 Windows 磁碟較慢，M1 實測載入時間；太慢則改放 WSL ext4，以 `extra_model_paths.yaml` 指向。
- **cam_object** 是以 512px 合成渲染圖訓練，對 1024px 輸入的效果需實測。
- **Anything2Real** 目標是照片感，可能加入環境或景深；需要控制強度。
- **SAM3** 對機甲部位的文字分割準確度未知，所以初期改用手動 bbox。
- 部分 LoRA 發布在 Civitai，授權條款各異；本專案為公司用途，請在 `docs/lora-notes.md` 記錄每顆 LoRA 的授權。

---

## 10. 不在範圍內

- LoRA 訓練（任何形式）
- 3D 生成、TRELLIS.2 的安裝或修改
- 網頁 UI（初期以 CLI ＋ HTML 對照表為主）
- 自動化的部件分割（SAM3），列為之後的選配
