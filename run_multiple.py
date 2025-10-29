from ultralytics import YOLO
import cv2
import os

model = YOLO("./model/yolo11x.pt")

# 偵測多張圖片
results = model("./images/")
os.makedirs("./results", exist_ok=True)

# 逐張儲存
for i, result in enumerate(results):
    img = result.plot()  # 對每張圖畫上框框
    cv2.imwrite(f"./results/detect_result{i+1}.jpg", img) 

print(f"已儲存到: results資料夾")
