from ultralytics import YOLO
import cv2

model = YOLO("./model/yolo11x.pt")                           #載入模型
results = model("./images/anypic.jpg")[0]                    #放任何一張圖片

results_img = results.plot()                                 #是從模型推論結果中取出第一張圖片，並把偵測到的物件畫出來，生成可視化後的圖片

cv2.imwrite("./results/detect_result.jpg", results_img)      #儲存結果
print("✅ 已儲存 detect_result.jpg")