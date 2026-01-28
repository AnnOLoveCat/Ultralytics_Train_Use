from ultralytics import YOLO

# 建議：若要進行最終訓練，建議使用預訓練模型作為基底
model = YOLO('./model/yolov8s.pt') 

# ==========================================
# 最終優化訓練腳本 (Final Optimized Training)
# 結合了從基礎驗證到進階增強的所有參數
# ==========================================
model.train(
    data='./dataset/data.yaml',
    project='Runs',
    name='Final_Stage_Model',
    
    # --- 1. 硬體與基礎設定 (Infrastructure) ---
    epochs=150,          # [訓練輪數] 
                         # 進階訓練建議至少 100-300 輪，讓模型有足夠時間收斂。
    
    imgsz=640,           # [影像尺寸] 
                         # 640 是標準速度與準度的平衡點；若工廠電腦夠強可開 1280 抓細節。
    
    batch=6,             # [批次大小] 
                         # 注意：因為開啟了下方的 Mosaic/Mixup 增強，VRAM 消耗會變大。
                         # 若發生 OOM (Out Of Memory)，請將此數值調小 (如 4 或 2)。
    
    # --- 2. 驗證指標設定 (Validation Metrics) ---
    conf=0.25,           # [信心度] 推論時過濾掉信心低於 25% 的雜訊框。
    iou=0.5,             # [NMS 閾值] 控制刪除重複框的嚴格程度。

    # --- 3. 光影與環境增強 (Light Augmentation) ---
    # 目標：讓模型適應不同燈光色溫、陰影，省去蒐集不同環境資料的時間。
    hsv_h=0.015,         # 色相 (Hue)：模擬冷暖光變化。
    hsv_s=0.7,           # 飽和度 (Saturation)：模擬生鏽、褪色或鮮豔度。
    hsv_v=0.4,           # 明度 (Value)：模擬白天、陰天或陰影下的亮度。

    # --- 4. 幾何型態增強 (Geometry Augmentation) ---
    # 目標：讓模型適應物件距離遠近、位置偏移。
    scale=0.5,           # 隨機縮放 (+/- 50%)：讓模型學會看大物件與小物件。
    translate=0.1,       # 隨機平移 (10%)：模擬物件在畫面邊緣只露出一半。
    degrees=10.0,        # 隨機旋轉 (+/- 10度)：模擬鏡頭不正或物件歪斜，這對工業檢測極重要！

    # --- 5. 進階特徵增強 (Advanced Robustness) ---
    # 目標：針對遮擋、堆疊進行極限訓練。
    mosaic=1.0,          # [馬賽克] 拼圖增強。大幅提升小物件偵測能力與背景適應力。
    mixup=0.1,           # [混合] 疊圖增強。增加對透明、重疊物件的抗干擾力 (設 0.1-0.2 即可)。
    
    # --- 6. 收斂優化 (Final Polish) ---
    close_mosaic=10,     # [關閉馬賽克] 
                         # 在最後 10 輪關閉 Mosaic，讓模型看清楚「完整的圖片」，
                         # 這通常能讓準確度 (mAP) 在最後時刻再往上衝刺 1%~2%。
)

print("訓練完成！最佳權重檔位於 runs/Runs/Final_Stage_Model/weights/best.pt")