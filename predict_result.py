from ultralytics import YOLO
import cv2
import os

# 1. 載入訓練好的模型
# 請確認路徑指向您 Advanced Training 跑出來的 best.pt
model_path = "./runs/Final_Stage_Model/weights/best.pt" 
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