from ultralytics import YOLO

model = YOLO('./model/yolo11x.pt')

# 開始訓練
model.train(data='./dataset/data.yaml',
            epochs=50, 
            imgsz=640, 
            batch=8, 
            name='retrain_simple')

print("✅ 訓練完成，模型儲存於 runs/detect/retrain_simple/")
