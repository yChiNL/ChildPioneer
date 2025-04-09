from celery import shared_task
import time

@shared_task
def image_recognition_task(image_path):
    # 在這裡寫你的影像辨識程式碼
    # 使用 image_path 參數來讀取影像檔案

    # 這裡只是一個示例，模擬執行時間
    time.sleep(10)
    result = "Image recognition result"

    return result