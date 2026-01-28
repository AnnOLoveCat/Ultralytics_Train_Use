# YOLOv11 Object Detection Project: From Auto-labeling to Advanced Training

## 本專案提供一套完整的物件偵測工作流 (Workflow)。
## 從環境建置、使用預訓練模型進行自動標記、人工修正、到最後的進階模型訓練與預測視覺化。

## 1. Environment Setup (環境建置)

為了確保專案的獨立性與重現性，強烈建議使用 Python 虛擬環境 (`venv`) 進行開發。

### 1.1 建立與啟動虛擬環境

**Windows:**
```bash
# 建立名為 .venv 的虛擬環境
python -m venv .venv

# 啟動虛擬環境
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv

# 啟動虛擬環境
source .venv/bin/activate
```
成功啟動後，您的終端機前方會出現 (.venv) 

### 1.2 安裝必要套件
在虛擬環境啟動的狀態下，安裝 YOLO 核心庫、標記工具與影像處理套件：
```bash
# 更新 pip
python -m pip install --upgrade pip

# 安裝 Ultralytics (YOLO), LabelImg (標記工具), OpenCV (影像處理)
pip install ultralytics labelimg opencv-python
```

## 2. Auto-labeling (自動標記)
為了節省從零開始畫框的時間，我們先使用 YOLO 官方強大的預訓練模型 (如 yolo11x.pt) 對原始圖片進行初步推論，自動產生標記檔。\
[auto_label.py](auto_label.py)

```bash
from ultralytics import YOLO
import os

# 1. 載入模型 (建議使用最強的官方模型以獲得最佳預標記品質)
model = YOLO('yolo11x.pt') 

# 2. 設定路徑
image_dir = './images'       # 原始圖片資料夾
output_dir = './auto_labels' # 輸出根目錄

os.makedirs(output_dir, exist_ok=True)

# 3. 執行自動標註
# save_txt=True: 儲存標記檔
# conf=0.25: 信心度門檻 (建議設低一點以免漏抓，之後人工刪除比補畫容易)
model.predict(
    source=image_dir,
    save=True,
    save_txt=True,
    project=output_dir,
    exist_ok=True,
    conf=0.25 
)

print(f"自動標記完成！請至 {output_dir}/predict/labels/ 取得 txt 檔")
```

## 3. Data Annotation & Correction (人工修正)
使用標註工具修正 Auto-labeling 的結果。

啟動工具：在終端機輸入 labelimg。

**修正重點：**

1. 修正誤判 (False Positive)：刪除把背景誤認為物件的框。
2. 補上漏抓 (False Negative)：人工補上模型沒抓到的物件。
3. 調整邊界：確保框線緊貼物件邊緣 (Tight Bounding Box)。

## 4. Dataset Preparation (資料集準備)
**[關鍵步驟]** 

將標註好的資料整理成 YOLO 訓練格式。\
YOLO 強制要求圖片 (images) 與標籤 (labels) 分開，且需區分 train (訓練用) 與 val (驗證用)。

### 4.1 資料夾結構
```bash
dataset/
├── images/
│   ├── train/  # 80% 圖片
│   └── val/    # 20% 圖片
└── labels/
    ├── train/  # 對應的 train 標註檔 (.txt)
    └── val/    # 對應的 val 標註檔 (.txt)
```

### 4.2 建立設定檔 dataset/data.yaml

```bash
path: ../dataset  # 資料集根目錄 (相對路徑)
train: images/train
val: images/val

# 類別設定 (請依實際情況修改)
nc: 1            # 類別數量 (Number of Classes)
names:
  0: steel_beam  # 類別名稱
```
## 5. Stage 1: Baseline Training (基礎訓練)
目標：快速跑通流程，確認資料標記格式正確，硬體能跑。此階段不追求高準確率。\
[train_basic.py](train_basic.py)
```bash
from ultralytics import YOLO

# 載入輕量版模型 (Nano version) 進行快速測試
model = YOLO('yolo11n.pt')

model.train(
    data='./dataset/data.yaml',
    epochs=50,      # 少輪數，快速驗證
    imgsz=640,
    batch=8,
    project='Runs',
    name='Basic_Model'
)
```

驗收標準：檢查 Runs/Baseline_Model/results.png。mAP50 曲線應呈現上升趨勢。\
若低於 50%，請回頭檢查 Step 3 的標註品質。\
詳細請看：[chart_analysis.md](docs/chart_analysis.md)

## 6. Stage 2: Advanced Training (進階訓練)
目標：透過資料增強與超參數調整，追求 90% 以上 的穩定性與準確率。此腳本整合了幾何變換、光影增強與收斂策略。\
[train_final.py](train_final.py)
```bash
from ultralytics import YOLO

# 建議：正式訓練使用 Small (s) 或 Medium (m) 版本以平衡速度與準度
model = YOLO('yolo11s.pt') 

# ==========================================
# 最終優化訓練腳本
# ==========================================
model.train(
    data='./dataset/data.yaml',
    project='Runs',
    name='Final_Stage_Model',
    
    # --- 1. 硬體與基礎設定 ---
    epochs=150,          # [訓練輪數] 建議至少 100-300 輪，讓模型充分收斂。
    imgsz=640,           # [影像尺寸] 若硬體許可且物件細節多，可開至 1280。
    batch=6,             # [批次大小] 因開啟下方增強功能較吃顯存，建議調保守一點避免 OOM。
    
    # --- 2. 驗證指標設定 ---
    conf=0.25,           # [信心度] 推論時過濾掉雜訊框。
    iou=0.5,             # [NMS] 去除重疊框的嚴格程度。

    # --- 3. 光影與環境增強 (省去拍攝不同環境的時間) ---
    hsv_h=0.015,         # 色相 (Hue)：模擬冷暖光變化。
    hsv_s=0.7,           # 飽和度 (Saturation)：模擬生鏽、褪色。
    hsv_v=0.4,           # 明度 (Value)：模擬陰天或陰影下。

    # --- 4. 幾何型態增強 (適應不同距離與角度) ---
    scale=0.5,           # 隨機縮放：適應大小物件。
    translate=0.1,       # 隨機平移：適應物件在邊緣的情況。
    degrees=10.0,        # 隨機旋轉：模擬鏡頭不正或物件歪斜 (工業檢測重要參數！)。

    # --- 5. 進階特徵增強 (提升抗干擾力) ---
    mosaic=1.0,          # [馬賽克] 拼圖增強，大幅提升小物件偵測能力。
    mixup=0.1,           # [混合] 疊圖增強，增加對透明/重疊物件的適應力。
    
    # --- 6. 收斂優化策略 ---
    close_mosaic=10,     # [最後衝刺] 在最後 10 輪關閉馬賽克，讓模型看清楚完整的圖片，
                         # 通常能讓 mAP 在最後時刻再提升 1-2%。
)

print("訓練完成！最佳權重檔位於 Runs/Final_Stage_Model/weights/best.pt")
```

## 7. Inference & Visualization (預測結果顯示)
訓練完成後，使用訓練好的最佳權重檔 (best.pt) 對新圖片進行測試，並將結果繪製出來。\
[predict_result.py](predict_result.py)

```bash
from ultralytics import YOLO
import cv2
import os

# 1. 載入訓練好的模型
# 請確認路徑指向您 Advanced Training 跑出來的 best.pt
model_path = "./Runs/Final_Stage_Model/weights/best.pt" 
model = YOLO(model_path)

# 2. 設定路徑
source_images = "./images/test"  # 待測圖片資料夾 (請自行準備測試圖)
output_folder = "./results"      # 結果輸出資料夾

# 3. 執行預測
results = model(source_images)
os.makedirs(output_folder, exist_ok=True)

# 4. 繪製並儲存結果
for i, result in enumerate(results):
    # result.plot() 會將偵測框畫在圖上，並回傳 BGR 格式圖片 (適合 cv2 使用)
    img = result.plot()
    
    # 儲存圖片
    save_path = os.path.join(output_folder, f"detect_result_{i+1}.jpg")
    cv2.imwrite(save_path, img) 

print(f"預測完成！結果已儲存至: {output_folder}")
```