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