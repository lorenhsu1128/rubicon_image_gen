# ComfyUI 範本庫：機甲轉圖流程

開啟 ComfyUI（`start_comfyui.bat`）→ 左側「範本」→ 左欄最下方「擴充功能」→ **0_MechPipeline**。
每個範本左上角都有「使用說明」卡。

| 範本 | 做什麼 | 需要先準備 |
|---|---|---|
| 01_改色或修改（姿勢不動） | 原圖＋一句修改，構圖不變 | 上傳原圖 |
| 02_轉成正面A-pose | 改好的圖＋A-pose 骨架圖（姿勢參考）→ 正面立正 A-pose（直式 864×1152） | 01 的結果 |
| 03_全身轉45度 | camera-angle LoRA 把整台轉到左前方 45° | 02 的結果 |
| 04_抽取頭或四肢（45度） | 從 45° 全身圖抽出頭、整隻手臂、整條腿 | `master_45.png`＋框選 |
| 05_細分部位（整件切段補完） | 抽整件 → 依框切出細分段 → 補完切口 | `master_45.png`＋框選 |
| 06_軀幹（刪臂裁切補完轉45度） | 刪雙臂 → 塗除頭腿 → 裁切 → 補完 → 轉 45° | `master_front.png`＋框選 |
| 07_單一部件轉45度 | 正面部件 → 45°（文字與 LoRA 兩種結果） | 一張正面白底部件圖 |
| 08_修圖（擦除與補畫） | 遮罩編輯器塗範圍 → 只擦除／擦除並補畫（mode：刪除（補背景）或補畫結構） | 一張要修的圖 |
| 09_完整流程（機甲圖到18個部位） | 01→02→03→自動預框→頭與四肢→細分部位→軀幹組，一次跑完 | 上傳原圖 |

- 04–06 從 `runs/<機甲>/master/` 讀圖、從 `config/mechs/<機甲>.boxes.yaml` 讀框，所以要先用 CLI 的
  `mechpipe pick` 與 `mechpipe boxes`（見 `CLAUDE.md`）。節點上的 `mech_id` 填機甲代號。
- 範本只跑一次、不做自動 QC 與重試；要批次產出 18 個部位並自動挑選，仍用 `mechpipe s2`。
- 結果存在 `ComfyUI/output/mech/`。
- 每個「提示詞」節點（以及 08 的「修圖模式」）下方的 `full_prompt` 方框會自動顯示實際使用的完整提示詞
  （`web/mech_prompt.js` 向 `/mech/prompt` 取得）；改 template、part、view、text 會重新產生，也可以直接改方框內容，
  執行時就用方框的文字。方框留空則依上面的欄位產生。
- 範本更新後若在範本庫看到舊版，按 Ctrl+F5 重新整理瀏覽器（範本檔會被瀏覽器快取）。
- 草稿模式（Lightning 4 步）；KSampler 的種子預設每次隨機。

## 檔案

- `comfy_nodes/0_MechPipeline/__init__.py`：自訂節點（分類「機甲轉圖」），沿用 `mechpipe` 的提示詞、部位清單與框。
  `scripts/setup_comfyui.sh` 會把它連結到 `ComfyUI/custom_nodes/0_MechPipeline`。
- `templates_api/*.api.json`：API 格式（`scripts/build_comfy_templates.py` 產生），`notes.json` 是說明卡內容。
- `example_workflows/*.json`＋同名 `.jpg`：範本庫實際顯示的 UI 格式與縮圖。

## 修改範本

1. 改 `scripts/build_comfy_templates.py`（或 `notes.json`），執行它產生 API 格式。
2. `python scripts/test_comfy_templates.py` 把每個範本實際跑一次（需要 ComfyUI 執行中）。
3. 轉成 UI 格式：在 ComfyUI 頁面用 `app.loadApiJson()` 載入（或載入舊版 UI 檔再改）、排版、加說明卡後 `app.graph.serialize()` 存成
   `example_workflows/<名稱>.json`（2026-10-05 由 Claude 透過瀏覽器自動化完成）。

外掛資料夾名稱必須是英文：ComfyUI v0.38.0 的範本網址路徑含中文資料夾名稱時會回 404（範本檔名可以是中文）。
`0_` 開頭讓它排在「擴充功能」的第一個；要排到整個範本庫最上面只能改 ComfyUI 前端，未採用。
