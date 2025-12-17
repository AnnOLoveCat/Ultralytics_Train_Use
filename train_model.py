from ultralytics import YOLO

model = YOLO('./model/yolov8s.pt') # 替換為 's' 或 'n' 版本，筆電友善

# 開始訓練 (目標: 穩定性與準確性)
model.train(data='./dataset/data.yaml',
            epochs=100,                     # 訓練週期減少
            imgsz=720,                      # 影像尺寸減少
            batch=8,                        # batch size 稍微增加，如果 VRAM 允許# 確保使用預設的 lr0 即可，但如果發現 mAP 不升，可手動調整 lr0
            augment=True,
            mosaic=1.0,                     # 極端數據下維持高增強
            mixup=0.2,
            degrees=10,
            scale=0.5,                      # 維持較大的縮放，增加多樣性
            translate=0.1,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            name='retrain_small_data')

print("訓練完成，模型儲存於 runs/detect/retrain_simple/")
