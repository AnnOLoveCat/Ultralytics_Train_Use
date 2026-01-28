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