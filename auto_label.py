from ultralytics import YOLO
import os

# 載入模型
model = YOLO('./model/yolo11x.pt')

# 設定圖片來源與輸出路徑
image_dir = './images'  # 放原始圖片的資料夾
output_dir = './auto_labels'  # 自動標記結果會存這裡
os.makedirs("./output", exist_ok=True)

# 執行預測（自動標註）
model.predict(
    source=image_dir,
    save=True,
    save_txt=True,
    project=output_dir,
    exist_ok=True
)

print("自動標記完成，圖片與 txt 已儲存於 auto_labels/labels/")
