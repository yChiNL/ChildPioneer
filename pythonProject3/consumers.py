import json
from channels.generic.websocket import AsyncWebsocketConsumer
import subprocess



class DetectionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)

        # 實行影像辨識程式 dance.py，將輸入的影像路徑作為參數
        input_image_path = data.get('image_path', '')
        if input_image_path:
            command = f'python dance.py {input_image_path}'
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            stdout, stderr = process.communicate()

            # 獲取影像辨識結果，這裡假設辨識結果是 stdout
            result = stdout.decode('utf-8')

            # 將影像辨識結果回傳至前端
            await self.send(text_data=json.dumps({'result': result}))