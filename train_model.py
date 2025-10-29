# train_yolo.py
from ultralytics import YOLO
import os
import cv2
import numpy as np

# ========= 1️⃣ 設定路徑 =========
MODEL_PATH = "./model/yolo11x.pt"       # 預訓練模型
DATA_PATH = "./config.yaml"             # 資料集設定檔
SAVE_DIR = "./train_results"            # 儲存輸出結果資料夾

os.makedirs(SAVE_DIR, exist_ok=True)

# ========= 2️⃣ 載入模型並開始訓練 =========
model = YOLO(MODEL_PATH)

results = model.train(
    data=DATA_PATH,
    epochs=10,
    imgsz=640,
    project=SAVE_DIR,
    name="exp_cat",
    save=True
)

print("✅ 模型訓練完成！")

# ========= 3️⃣ 取出訓練後的最佳模型 =========
best_model_path = f"{SAVE_DIR}/exp_cat/weights/best.pt"
trained_model = YOLO(best_model_path)

# ========= 4️⃣ 用訓練後模型對驗證集做推論 =========
val_results = trained_model("./data/images/val/")

# ========= 5️⃣ 輸出標註結果 (image + txt) =========
output_img_dir = f"{SAVE_DIR}/pred_images"
output_txt_dir = f"{SAVE_DIR}/pred_labels"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_txt_dir, exist_ok=True)

for i, result in enumerate(val_results):
    # 取得畫好框框的圖片
    img = result.plot()
    img_name = os.path.basename(result.path)
    img_save_path = os.path.join(output_img_dir, img_name)
    cv2.imwrite(img_save_path, img)

    # 取得偵測框的 xywh、類別與信心分數
    boxes = result.boxes.xywh.cpu().numpy()       # [x, y, w, h]
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()

    # 將結果輸出成 YOLO txt 格式：class x y w h conf
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_save_path = os.path.join(output_txt_dir, txt_name)

    with open(txt_save_path, "w") as f:
        for c, box, s in zip(classes, boxes, scores):
            x, y, w, h = box
            f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {s:.6f}\n")

    print(f"✅ 已輸出：{img_save_path} 和 {txt_save_path}")

print("🎉 全部完成！圖片與標註檔已輸出。")
