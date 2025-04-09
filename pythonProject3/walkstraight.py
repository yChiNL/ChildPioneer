import cv2
import mediapipe as mp
import time
import django
import os
import numpy as np
from threading import Thread
import queue
from PIL import ImageFont, ImageDraw, Image  # 載入 PIL 相關函式庫

def removebg(**kwargs):
    img = kwargs['img']
    selfie_segmentation = kwargs['selfie_segmentation']
    resized_background = kwargs['resized_background']

    results = selfie_segmentation.process(img)  # 取得自拍分割結果
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1  # 如果滿足模型判斷條件 ( 表示要換成背景 )，回傳 True
    segmented_frame = np.where(condition, img, resized_background)
    segmented_frame = cv2.flip(segmented_frame, 1)
    return segmented_frame

def play():
    global  flipped_image, time_started, time1, failure_count, end_point, target_point, distance_cm, foot_distance, font_time, font, NormalMode1, NormalMode2, mp_drawing, mp_pose, results
    cv2.line(flipped_image, (0, end_point), (640, end_point), (0, 0, 255), 3)
    cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
    pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text((0, 50), f"人物距離: {distance_cm:.1f} 公分", fill=(0, 255, 0), font=font_time)
    draw.text((0, 80), f"雙腳距離: {foot_distance:.2f} ", fill=(0, 255, 0), font=font_time)
    flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # 270以上動作正確
    if (foot_distance <= NormalMode1 and float(distance_cm) > 270):
        time_started = False
        if not time_started:
            time1 = 0
            time2 = 0
            time1 = time.time()
            time_started = True

    # 270以上動作錯誤
    elif (foot_distance > NormalMode1 and float(distance_cm) > 270):
        mp_drawing.draw_landmarks(
            image=flipped_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                         circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                           circle_radius=2)
        )
        if time_started:
            time2 = time.time()
            pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            draw.text((0, 110), f"失敗時間: {(int(time2 - time1))} 秒", fill=(0, 255, 0),
                      font=font_time)
            flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            if (int(time2 - time1) == 3):
                failure_count = failure_count + 1
                pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                draw.text((270, 205), f"失敗", fill=(0, 255, 0), font=font)
                flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                time_started = False

    # 270以下動作正確
    elif (foot_distance <= NormalMode2 and float(distance_cm) < 270):
        time_started = False
        if not time_started:
            time1 = 0
            time2 = 0
            time1 = time.time()
            time_started = True

    # 270以下動作錯誤
    elif (foot_distance > NormalMode2 and float(distance_cm) < 270):
        mp_drawing.draw_landmarks(
            image=flipped_image,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                         circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                           circle_radius=2)
        )
        if time_started:
            time2 = time.time()
            pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            draw.text((0, 110), f"失敗時間: {(int(time2 - time1))} 秒", fill=(0, 255, 0),
                      font=font_time)
            flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            if (int(time2 - time1) == 3):
                failure_count = failure_count + 1
                pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(pil_image)
                draw.text((270, 205), f"失敗", fill=(0, 255, 0), font=font)
                flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                time_started = False

def walkstraight_Ther(user, mode,id, background_path):
    global flipped_image, time_started, time1, failure_count, end_point, target_point, distance_cm, foot_distance, font_time, font, NormalMode1, NormalMode2, mp_drawing, mp_pose, results
    from pythonProject3.models import TrainingWalkStraight, CustomUser
    from pythonProject3.models import TreatmentContent

    # 初始化 MediaPipe 的姿態估計模型
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    try:
        fontpath = "font/Arial Unicode MS.ttf"
        open(fontpath, 'r+')
    except:
        fontpath = "../font/Arial Unicode MS.ttf"
        open(fontpath, 'r+')

    font_size = 50
    font_size_time = 30
    font = ImageFont.truetype(fontpath, font_size)  # 設定字型與文字大小
    font_time = ImageFont.truetype(fontpath, font_size_time)  # 成功次數與秒數字型大小

    # 初始化變數
    train_started = False
    show_text = True

    # EasyMode1 = 0.5
    # EasyMode2 = 0.79
    NormalMode1 = 0.19
    NormalMode2 = 0.4
    # HardMode1 = 0.04
    # HardMode2 = 0.18

    Mode = mode
    time1 = 0
    time2 = 0
    end_point = 420
    t_time1 = 0
    t_time2 = 0
    t_time = 0
    time_started = False
    failure_count = 0

    # 定點座標 (假設定點在畫面中央)
    target_point = (320, 320)  # 左腳點
    r_target_point = (320, 360)  # 右腳點
    center_point = [320, 205]

    # 初始化視訊處理
    try:
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise Exception("error")
    except Exception:
        cap = cv2.VideoCapture(0)

    success, image = cap.read()
    background = cv2.imread(background_path)
    resized_background = cv2.resize(background, (image.shape[1], image.shape[0]))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            # 將圖像轉換成 RGB 格式
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            segmented_frame = removebg(
                selfie_segmentation=selfie_segmentation,
                img=image_rgb,
                resized_background=resized_background
            )
            image_rgb_flip = cv2.flip(image_rgb, 1)
            results = pose.process(image_rgb_flip)

            # 繪製姿態估計結果和定點
            flipped_image = segmented_frame


            # 檢測使用者是否站在定點上
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                target_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * flipped_image.shape[1])
                target_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                r_target_x = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * flipped_image.shape[1])
                r_target_y = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])

                if abs(target_x - target_point[0]) < 10 and abs(target_y - target_point[1]) < 10 and abs(
                        r_target_x - (target_point[0] + 8)) < 10:
                    show_text = False
                    if not train_started:
                        train_started = True
                        t_time1 = time.time()
                else:
                    if show_text:
                        mp_drawing.draw_landmarks(
                            image=flipped_image,
                            landmark_list=results.pose_landmarks,
                            connections=mp_pose.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                                         circle_radius=2),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                           circle_radius=2)
                        )
                        cv2.circle(flipped_image, (target_point[0] - 12, target_point[1]), 10, (0, 255, 0), -1)
                        cv2.circle(flipped_image, (target_point[0] + 12, target_point[1]), 10, (0, 255, 255), -1)
                        pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        draw.text((195, 180), "右腳碰黃點", fill=(0, 255, 0), font=font)
                        draw.text((195, 230), "左腳碰綠點", fill=(0, 255, 0), font=font)
                        flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            '''
                if Mode =="Hard":
                if abs(target_x - target_point[0]) < 10 and abs(target_y - target_point[1]) < 10 and abs(r_target_x - r_target_point[0])<10 and abs(r_target_y - r_target_point[1])<10:
                    show_text = False
                    if not train_started:
                        train_started = True
                else:
                    if show_text:
                        cv2.circle(flipped_image, target_point, 10, (0, 255, 0), -1)
                        cv2.circle(flipped_image, r_target_point, 10, (0, 255, 255), -1)
                        pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        draw.text((195, 180), "右腳碰黃點", fill=(0, 255, 0), font=font)
                        draw.text((195, 230), "左腳碰綠點", fill=(0, 255, 0), font=font)
                        flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)             
                elif Mode =="Easy":
                    if abs(target_x - target_point[0]) < 10 and abs(target_y - target_point[1]) < 10 and abs(r_target_x - (target_point[0] + 30))<10:
                        show_text = False
                        if not train_started:
                            train_started = True
                    else:
                        if show_text:
                            cv2.circle(flipped_image, target_point, 10, (0, 255, 0), -1)
                            cv2.circle(flipped_image, (target_point[0] + 30, target_point[1]), 10, (0, 255, 255), -1)
                            cv2.putText(flipped_image, "Please stand on point", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)'''

            # 顯示計時器
            if train_started:

                    # 計算人物距離
                    left_hip_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                    right_hip_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                    neck_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]

                    #
                    left_ankle_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
                    right_ankle_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
                    foot_distance = 10 * abs(left_ankle_landmark.x - right_ankle_landmark.x)

                    # 取得畫面寬高
                    image_height, image_width, _ = flipped_image.shape

                    # 計算人物在畫面中的距離
                    distance_pixels = abs(right_hip_landmark.x - left_hip_landmark.x) * image_width

                    # 計算人物距離（以公分為單位）
                    # 假設攝像頭與人物之間的距離為30公分
                    distance_cm = 10000 / distance_pixels

                    # 簡單模式
                    if (Mode == "Easy"):
                        play()

                    # 普通模式
                    if (Mode == "Normal"):
                        play()

                    # 困難模式
                    if (Mode == "Hard"):
                        play()

                    if (target_y > end_point):
                        t_time2 = time.time()
                        t_time = t_time2 - t_time1

                        plot_data.put(False)
                        break

            # 顯示圖像
            plot_data.put(flipped_image)

        cap.release()
        cv2.destroyAllWindows()

        def import_data(user, move_count, failure_count, t_time, difficulty, grade):
            training_data = TrainingWalkStraight(user=user, move_count=move_count, failure_count=failure_count, t_time=t_time,
                                                 difficulty=difficulty,
                                                 grade=grade)
            training_data.save()


        if Mode == "Easy":
            if t_time <= 100:
                grade = 'A'
            elif t_time > 100 and t_time <= 140:
                grade = 'B'
            else:
                grade = 'C'
        elif Mode == "Normal":
            if t_time <= 140:
                grade = 'A'
            elif t_time > 140 and t_time <= 180:
                grade = 'B'
            else:
                grade = 'C'
        elif Mode == "Hard":
            if t_time <= 180:
                grade = 'A'
            elif t_time > 180 and t_time <= 220:
                grade = 'B'
            else:
                grade = 'C'
        # 打印评分
        print(f"在 {Mode} 模式下，您的评分是 {grade}")

        if grade == 'A':
            coins_earned = 10  # Example: Assign 10 coins for grade A
        elif grade == 'B':
            coins_earned = 6  # Example: Assign 8 coins for grade B
        else:
            coins_earned = 4  # Example: Assign 4 coins for grade D

        try:
            # Retrieve the user object from the database and update the coins field
            user_instance = CustomUser.objects.get(id=user.id)
            user_instance.coins += coins_earned  # Increment the existing coins by the earned coins
            user_instance.save()
        except CustomUser.DoesNotExist:
            # Handle the case where the user doesn't exist
            pass
        difficulty = Mode
        test_results = {
            "move_count": 1,
            "failure_count": 0,
            "t_time": "",
            "difficulty": Mode,
            "grade": ""
        }
        test_results["move_count"] = 1
        test_results["failure_count"] = failure_count
        test_results["t_time"] = t_time
        test_results["difficulty"] = difficulty
        test_results["grade"] = grade
        import_data(user, test_results["move_count"], test_results["failure_count"], test_results["t_time"], test_results["difficulty"],
                    test_results["grade"])
        plot_data.put(False)

def main(user, mode, id):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pythonProject3.settings")
    django.setup()
    global plot_data
    plot_data = queue.LifoQueue()

    if (mode == 'Easy'):
        background_name = 'background_images/back_walk_easy.jpg'
    elif (mode == 'Normal'):
        background_name = 'background_images/back_walk_normal.jpg'
    elif (mode == 'Hard'):
        background_name = 'background_images/back_walk_hard.jpg'

    try:  # server端
        background_path = 'media/' + background_name
        open(background_path, 'r+')
    except:  # 離線端
        background_path = '../media/' + background_name
        open(background_path, 'r+')

    thread = Thread(target=walkstraight_Ther, args=(user, mode, id, background_path), daemon=True)
    thread.start()

    while plot_data.empty():
        pass
    while True:
        data = plot_data.get()
        if type(data) == type(False) or cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break
        cv2.imshow('walkstraight', data)

if __name__ == '__main__':
    main('stest', 'Hard',  0)
