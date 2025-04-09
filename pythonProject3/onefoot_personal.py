import cv2
import mediapipe as mp
import os
import django
import numpy as np
import copy
import time
from threading import Thread
import queue
from PIL import ImageFont,ImageDraw,Image  # 載入 PIL 相關函式庫

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

def calculateTextPoint(show_text, center_point, font_size):
    text_len = len(show_text)
    start_point = copy.copy(center_point)
    start_point[0] = int(center_point[0] - text_len / 2 * font_size)
    return tuple(start_point)

def doAction(show_text, font, font_size, newline_index=None, point=None, vertical_offset=0):
    global flipped_image, center_point
    pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    if point is not None:
        center = point
    else:
        center = [center_point[0], center_point[1] - vertical_offset]
    if newline_index is not None:
        draw.text(calculateTextPoint(show_text[:newline_index], [center_point[0], center_point[1] - int(font_size / 2)- vertical_offset],
                                     font_size), show_text[:newline_index], fill=(0, 255, 0), font=font)
        draw.text(calculateTextPoint(show_text[newline_index:], [center_point[0], center_point[1] + int(font_size / 2)- vertical_offset],
                                     font_size), show_text[newline_index:], fill=(0, 255, 0), font=font)
    else:
        draw.text(calculateTextPoint(show_text, center, font_size), show_text, fill=(0, 255, 0), font=font)
    flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def readImage(path):
    try:
        open(path)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except:
        image = cv2.imread('../'+path, cv2.IMREAD_UNCHANGED)

    return image

def onefoot_Per(user, mode, background_path):
    global flipped_image, center_point
    from pythonProject3.models import CustomUser
    from pythonProject3.models import TrainingOneFoot

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

    hint_time_started = False
    hint_time = 0
    hint_time1 = 0
    hint_time2 = 0
    # 初始化變數
    train_started = False
    time_started = False
    fail_started = False
    fail = False
    time1 = 99999999999
    time2 = 0
    t_time1 = 0
    t_time2 = 0
    t_time = 0
    time_count = 0
    fail_count = 0
    count = 0
    left_foot = True
    Mode = mode
    m_started = False
    move_count = 0
    show_text = True

    # 定點座標
    target_point = None
    target_point = (320, 400)
    center_point = [320, 205]

    t_timestarted = False
    success_displayed = False


    # 以下新加
    mouse_move = False
    # 老鼠的起始位置 (x, y)
    mouse_position = (320, 480)
    mouse_position_y = mouse_position[1]
    # 老鼠的移動速度
    mouse_speed = 3
    # 老鼠的移動方向，1 表示向右，-1 表示向左
    mouse_direction = 1
    if mode == "Hard_Right" or mode == 'VeryEasy_Right' or mode == 'Easy_Right':
        left_foot = False
        mouse_position = (360, 480)

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
            flipped_image = segmented_frame
            # 檢測使用者是否站在定點上
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                target_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * flipped_image.shape[1])
                target_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                target_x1 = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * flipped_image.shape[1])
                target_y1 = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                if mode == "Hard_Right":
                    left_foot = False
                    target_point = (360, 420)
                    if abs(target_x1 - target_point[0]) < 10 and abs(target_y1 - target_point[1]) < 10:
                        mouse_move = True
                        if not hint_time_started:
                            hint_time1 = time.time()
                            hint_time_started = True
                        if not t_timestarted:
                            t_time1 = time.time()
                            t_timestarted = True
                        show_text = False
                        if not train_started:
                            train_started = True
                    else:
                        if show_text:
                            if results.pose_landmarks:
                                mp_drawing.draw_landmarks(
                                    image=flipped_image,
                                    landmark_list=results.pose_landmarks,
                                    connections=mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                                                 circle_radius=2),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                                   circle_radius=2)
                                )
                            cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                            doAction("請站在點上", font=font, font_size=font_size, vertical_offset=100)

                if left_foot:
                    target_point = (320, 420)
                    if abs(target_x - target_point[0]) < 10 and abs(target_y - target_point[1]) < 10:
                        mouse_move = True
                        if not hint_time_started:
                            hint_time1 = time.time()
                            hint_time_started = True
                        if not t_timestarted:
                            t_time1 = time.time()
                            t_timestarted = True
                        show_text = False
                        if not train_started:
                            train_started = True
                    else:
                        if show_text:
                            if results.pose_landmarks:
                                mp_drawing.draw_landmarks(
                                    image=flipped_image,
                                    landmark_list=results.pose_landmarks,
                                    connections=mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                                                 circle_radius=2),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                                   circle_radius=2)
                                )
                            cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                            doAction("請站在點上", font=font, font_size=font_size, vertical_offset=100)
                else:
                    target_point = (360, 420)
                    if abs(target_x1 - target_point[0]) < 10 and abs(target_y1 - target_point[1]) < 10:
                        mouse_move = True
                        show_text = False
                        if not hint_time_started:
                            hint_time1 = time.time()
                            hint_time_started = True
                        if not train_started:
                            train_started = True
                    else:
                        if show_text:
                            if results.pose_landmarks:
                                mp_drawing.draw_landmarks(
                                    image=flipped_image,
                                    landmark_list=results.pose_landmarks,
                                    connections=mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                                                 circle_radius=2),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                                   circle_radius=2)
                                )
                            cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                            doAction("請站在點上", font=font, font_size=font_size, vertical_offset=100)
                    if mode == 'VeryEasy_Right' or mode == 'Easy_Right':
                        target_point = (360, 420)
                        if abs(target_x1 - target_point[0]) < 10 and abs(target_y1 - target_point[1]) < 10:
                            mouse_move = True
                            if not hint_time_started:
                                hint_time1 = time.time()
                                hint_time_started = True
                            if not t_timestarted:
                                t_time1 = time.time()
                                t_timestarted = True
                            show_text = False
                            if not train_started:
                                train_started = True
                        else:
                            if show_text:
                                if results.pose_landmarks:
                                    mp_drawing.draw_landmarks(
                                        image=flipped_image,
                                        landmark_list=results.pose_landmarks,
                                        connections=mp_pose.POSE_CONNECTIONS,
                                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                                                     circle_radius=2),
                                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                                       circle_radius=2)
                                    )
                                cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                                doAction("請站在點上", font=font, font_size=font_size, vertical_offset=100)

            # 成功
            if success_displayed:
                success_time2 = time.time()
                success_ttime = success_time2 - success_time1
                if int(success_ttime) == 2:
                    show_text = True
                    success_displayed = False
                doAction("鱷魚休息中...", font=font, font_size=font_size, vertical_offset=50)
                """success_image = readImage('media/successcroc.png')
                re_success = cv2.resize(success_image, (40, 40))
                success_height, success_width, _ = re_success.shape"""
                success_image = readImage('media/successcroc.png')  # 以彩色模式读取老鼠图像
                re_success = cv2.resize(success_image[:], (100, 100))

                success_height, success_width, _ = re_success.shape
                # 创建一个带有透明通道的空白图像
                success_with_alpha = np.zeros((success_height, success_width, 4), dtype=np.uint8)
                # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                success_with_alpha = re_success
                # 设置透明通道
                alpha_threshold = 0  # 设置透明度阈值
                # 使用蓝色通道作为透明度
                alpha_channel = re_success[:, :, 0]  # 使用蓝色通道作为透明度
                success_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                # 直接将带有透明通道的老鼠图像合成到背景图像上
                alpha = success_with_alpha[:, :, 3] / 255.0
                success_position = (success_position[0], success_position[1] - (1 * success_direction))
                success_x = target_point[0] - int(success_height / 2)
                success_y = int(success_position[1]) - success_width
                if success_position[1] <= goal_point[1] - 10 or success_position[1] >= goal_point[1] + 10:
                    success_direction *= -1
                """flipped_image[success_y:success_y + success_height,
                success_x:success_x + success_width] = re_success"""
                for c in range(3):
                    flipped_image[success_y:success_y + success_height, success_x:success_x + success_width, c] = (
                            (1.0 - alpha) * flipped_image[success_y:success_y + success_height,
                                            success_x:success_x + success_width, c] +
                            alpha * success_with_alpha[:, :, c]
                    )
            # 顯示計時器
            if train_started:
                if (Mode == "VeryEasy_Left"):
                    cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                    distance = 20
                    pixel_cm = 2.4
                    point_y = target_point[1] - int(distance * pixel_cm)
                    cv2.circle(flipped_image, (target_point[0], point_y), 8, (0, 255, 0), -1)
                    goal_point = (target_point[0], point_y)
                    # 成功
                    success_position = (target_point[0], point_y)
                    # 成功
                    success_direction = -1
                    doAction(f'成功次數: {move_count} / 3', font=font_time, font_size=font_size_time,
                             point=[170, 0])
                    if hint_time_started:
                        hint_time2 = time.time()
                        hint_time = int(hint_time2-hint_time1)
                        if hint_time >= 3:
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
                        time_limit_minutes = 6  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 6 minutes. Exiting program.")
                            break
                    # 新加
                    if mouse_move:
                        doAction("鱷魚出沒！抬起左腳！", font=font, font_size=font_size)

                        mouse_image = cv2.imread('media/croc.png', cv2.IMREAD_COLOR)  # 以彩色模式读取老鼠图像
                        re_image = cv2.resize(mouse_image, (90, 90))

                        mouse_height, mouse_width, _ = re_image.shape

                        # 创建一个带有透明通道的空白图像
                        mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)

                        # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                        mouse_with_alpha[:, :, :3] = re_image

                        # 设置透明通道
                        alpha_threshold = 0  # 设置透明度阈值
                        # 使用蓝色通道作为透明度
                        alpha_channel = re_image[:, :, 1]  # 使用綠色通道作为透明度
                        mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)

                        mouse_position = (mouse_position[0], mouse_position[1] - (mouse_speed * mouse_direction))
                        # 检查是否越界并改变方向
                        mouse_x = target_point[0] - int(mouse_height / 2)
                        mouse_y = int(mouse_position[1]) - mouse_width

                        if mouse_position[1] - mouse_height <= goal_point[1] or mouse_position[1] == mouse_position_y:
                            mouse_direction *= -1

                        # 直接将带有透明通道的老鼠图像合成到背景图像上
                        alpha = mouse_with_alpha[:, :, 3] / 255.0
                        for c in range(3):
                            flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                    (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                    mouse_x:mouse_x + mouse_width, c] +
                                    alpha * mouse_with_alpha[:, :, c]
                            )
                    if move_count == 3:  # 之後傳送數據
                        doAction("結束", font=font, font_size=font_size)
                        t_time2 = time.time()
                        t_time = t_time + (t_time2 - t_time1)
                        plot_data.put(False)
                        break
                    if fail:
                        if abs(target_y > goal_point[1]):
                            time_started = False
                            if fail_started:
                                fail_count = fail_count + 1
                                fail_started = False
                                fail = False
                    if abs(target_y < goal_point[1]):
                        fail = True
                        if hint_time_started:
                            hint_time = 0
                            hint_time1 = 0
                            hint_time2 = 0
                            hint_time_started=False
                        if not fail_started:
                            fail_started = True
                        if not time_started:
                            time1 = 99999999999
                            time2 = 0
                            time_count = 0
                            time1 = time.time()
                            time_started = True
                        if time_started:
                            time2 = time.time()
                            time_count = time2 - time1
                            if int(time_count) == 2:
                                fail = False
                                move_count = move_count + 1
                                time_started = False
                                # 成功
                                success_time1 = time.time()
                                # 成功
                                success_displayed = True
                                if move_count < 3:  # 重置状态
                                    # 成功
                                    train_started = False

                        doAction(f'持續時間: {time_count:.1f} 秒', font=font_time, font_size=font_size_time,
                                 point=[170, 410])
                        doAction(f'失敗次數: {fail_count} 次', font=font_time, font_size=font_size_time,
                                 point=[140, 440])

                if (Mode == "VeryEasy_Right"):
                    cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                    distance = 20
                    pixel_cm = 2.4
                    point_y = target_point[1] - int(distance * pixel_cm)
                    cv2.circle(flipped_image, (target_point[0], point_y), 8, (0, 255, 0), -1)
                    goal_point = (target_point[0], point_y)
                    # 成功
                    success_position = (target_point[0], point_y)
                    # 成功
                    success_direction = -1
                    doAction(f'成功次數: {move_count} / 3', font=font_time, font_size=font_size_time,
                             point=[170, 0])
                    if hint_time_started:
                        hint_time2 = time.time()
                        hint_time = int(hint_time2-hint_time1)
                        if hint_time >= 3:
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
                        time_limit_minutes = 6  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 6 minutes. Exiting program.")
                            break
                    # 新加
                    if mouse_move:
                        doAction("鱷魚出沒！抬起右腳！", font=font, font_size=font_size)

                        mouse_image = cv2.imread('media/croc.png', cv2.IMREAD_COLOR)  # 以彩色模式读取老鼠图像
                        re_image = cv2.resize(mouse_image, (90, 90))

                        mouse_height, mouse_width, _ = re_image.shape

                        # 创建一个带有透明通道的空白图像
                        mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)

                        # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                        mouse_with_alpha[:, :, :3] = re_image

                        # 设置透明通道
                        alpha_threshold = 0  # 设置透明度阈值
                        # 使用蓝色通道作为透明度
                        alpha_channel = re_image[:, :, 1]  # 使用綠色通道作为透明度
                        mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)

                        mouse_position = (mouse_position[0], mouse_position[1] - (mouse_speed * mouse_direction))
                        # 檢查是否越界並改變方向
                        mouse_x = target_point[0] - int(mouse_height / 2)
                        mouse_y = int(mouse_position[1]) - mouse_width

                        if mouse_position[1] - mouse_height <= goal_point[1] or mouse_position[1] == mouse_position_y:
                            mouse_direction *= -1

                        # 直接将带有透明通道的老鼠图像合成到背景图像上
                        alpha = mouse_with_alpha[:, :, 3] / 255.0
                        for c in range(3):
                            flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                    (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                    mouse_x:mouse_x + mouse_width, c] +
                                    alpha * mouse_with_alpha[:, :, c]
                            )
                    if move_count == 3:  # 之後傳送數據
                        doAction("結束", font=font, font_size=font_size)
                        t_time2 = time.time()
                        t_time = t_time + (t_time2 - t_time1)
                        plot_data.put(False)
                        break
                    if fail:
                        if abs(target_y1 > goal_point[1]):
                            time_started = False
                            if fail_started:
                                fail_count = fail_count + 1
                                fail_started = False
                                fail = False
                    if abs(target_y1 < goal_point[1]):
                        fail = True
                        if hint_time_started:
                            hint_time = 0
                            hint_time1 = 0
                            hint_time2 = 0
                            hint_time_started = False
                        if not fail_started:
                            fail_started = True
                        if not time_started:
                            time1 = 99999999999
                            time2 = 0
                            time_count = 0
                            time1 = time.time()
                            time_started = True
                        if time_started:
                            time2 = time.time()
                            time_count = time2 - time1
                            if int(time_count) == 2:
                                fail = False
                                move_count = move_count + 1
                                time_started = False
                                # 成功
                                success_time1 = time.time()
                                # 成功
                                success_displayed = True
                                if move_count < 3:  # 重置状态
                                    # 成功
                                    train_started = False

                            doAction(f'持續時間: {time_count:.1f} 秒', font=font_time, font_size=font_size_time,
                                     point=[170, 410])
                            doAction(f'失敗次數: {fail_count} 次', font=font_time, font_size=font_size_time,
                                     point=[140, 440])

                if (Mode == "Easy_Left"):
                    cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                    distance = 20
                    pixel_cm = 2.4
                    point_y = target_point[1] - int(distance * pixel_cm)
                    cv2.circle(flipped_image, (target_point[0], point_y), 8, (0, 255, 0), -1)
                    goal_point = (target_point[0], point_y)
                    # 成功
                    success_position = (target_point[0], point_y)
                    # 成功
                    success_direction = -1
                    doAction(f'成功次數: {move_count} / 5', font=font_time, font_size=font_size_time,
                             point=[170, 0])
                    if hint_time_started:
                        hint_time2 = time.time()
                        hint_time = int(hint_time2-hint_time1)
                        if hint_time >= 3:
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
                        time_limit_minutes = 8  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 8 minutes. Exiting program.")
                            break
                    # 新加
                    if mouse_move:
                        doAction("鱷魚出沒！抬起左腳！", font=font, font_size=font_size)


                        mouse_image = readImage('media/croc.png')  # 以彩色模式读取老鼠图像
                        re_image = cv2.resize(mouse_image[:], (90, 90))

                        mouse_height, mouse_width, _ = re_image.shape
                        # 创建一个带有透明通道的空白图像
                        mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)
                        # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                        mouse_with_alpha = re_image
                        # 设置透明通道
                        alpha_threshold = 0  # 设置透明度阈值
                        # 使用蓝色通道作为透明度
                        alpha_channel = re_image[:, :, 1]  # 使用蓝色通道作为透明度
                        mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                        # 直接将带有透明通道的老鼠图像合成到背景图像上
                        alpha = mouse_with_alpha[:, :, 3] / 255.0
                        mouse_position = (mouse_position[0], mouse_position[1] - (mouse_speed * mouse_direction))
                        # 檢查是否越界並改變方向
                        mouse_x = target_point[0] - int(mouse_height / 2)
                        mouse_y = int(mouse_position[1]) - mouse_width

                        if mouse_position[1] - mouse_height <= goal_point[1] or mouse_position[1] == mouse_position_y:
                            mouse_direction *= -1

                        for c in range(3):
                            flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                    (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                    mouse_x:mouse_x + mouse_width, c] +
                                    alpha * mouse_with_alpha[:, :, c]
                            )
                    if move_count == 5:  # 之後傳送數據
                        doAction("結束", font=font, font_size=font_size)
                        t_time2 = time.time()
                        t_time = t_time + (t_time2 - t_time1)
                        plot_data.put(False)
                        break
                    if fail:
                        if abs(target_y > goal_point[1]):
                            time_started = False
                            if fail_started:
                                fail_count = fail_count + 1
                                fail_started = False
                                fail = False
                    if abs(target_y < goal_point[1]):
                        fail = True
                        if hint_time_started:
                            hint_time = 0
                            hint_time1 = 0
                            hint_time2 = 0
                            hint_time_started = False
                        if not fail_started:
                            fail_started = True
                        if not time_started:
                            time1 = 99999999999
                            time2 = 0
                            time_count = 0
                            time1 = time.time()
                            time_started = True
                        if time_started:
                            time2 = time.time()
                            time_count = time2 - time1
                            if int(time_count) == 3:
                                fail = False
                                move_count = move_count + 1
                                time_started = False
                                # 成功
                                success_time1 = time.time()
                                # 成功
                                success_displayed = True
                                if move_count < 5:  # 重置状态
                                    # 成功
                                    train_started = False

                            doAction(f'持續時間: {time_count:.1f} 秒', font=font_time, font_size=font_size_time,
                                     point=[170, 410])
                            doAction(f'失敗次數: {fail_count} 次', font=font_time, font_size=font_size_time,
                                     point=[140, 440])

                if (Mode == "Easy_Right"):
                    cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                    distance = 20
                    pixel_cm = 2.4
                    point_y = target_point[1] - int(distance * pixel_cm)
                    cv2.circle(flipped_image, (target_point[0], point_y), 8, (0, 255, 0), -1)
                    goal_point = (target_point[0], point_y)
                    # 成功
                    success_position = (target_point[0], point_y)
                    # 成功
                    success_direction = -1
                    doAction(f'成功次數: {move_count} / 5', font=font_time, font_size=font_size_time,
                             point=[170, 0])
                    if hint_time_started:
                        hint_time2 = time.time()
                        hint_time = int(hint_time2-hint_time1)
                        if hint_time >= 3:
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
                        time_limit_minutes = 8  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 8 minutes. Exiting program.")
                            break
                    # 新加
                    if mouse_move:
                        doAction("鱷魚出沒！抬起右腳！", font=font, font_size=font_size)
                        mouse_image = readImage('media/croc.png')  # 以彩色模式读取老鼠图像
                        re_image = cv2.resize(mouse_image[:], (90, 90))

                        mouse_height, mouse_width, _ = re_image.shape
                        # 创建一个带有透明通道的空白图像
                        mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)
                        # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                        mouse_with_alpha = re_image
                        # 设置透明通道
                        alpha_threshold = 0  # 设置透明度阈值
                        # 使用蓝色通道作为透明度
                        alpha_channel = re_image[:, :, 1]  # 使用蓝色通道作为透明度
                        mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                        # 直接将带有透明通道的老鼠图像合成到背景图像上
                        alpha = mouse_with_alpha[:, :, 3] / 255.0
                        mouse_position = (mouse_position[0], mouse_position[1] - (mouse_speed * mouse_direction))
                        # 檢查是否越界並改變方向
                        mouse_x = target_point[0] - int(mouse_height / 2)
                        mouse_y = int(mouse_position[1]) - mouse_width

                        if mouse_position[1] - mouse_height <= goal_point[1] or mouse_position[1] == mouse_position_y:
                            mouse_direction *= -1

                        for c in range(3):
                            flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                    (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                    mouse_x:mouse_x + mouse_width, c] +
                                    alpha * mouse_with_alpha[:, :, c]
                            )
                    if move_count == 5:  # 之後傳送數據
                        doAction("結束", font=font, font_size=font_size)
                        t_time2 = time.time()
                        t_time = t_time + (t_time2 - t_time1)
                        plot_data.put(False)
                        break
                    if fail:
                        if abs(target_y1 > goal_point[1]):
                            time_started = False
                            if fail_started:
                                fail_count = fail_count + 1
                                fail_started = False
                                fail = False
                    if abs(target_y1 < goal_point[1]):
                        fail = True
                        if hint_time_started:
                            hint_time = 0
                            hint_time1 = 0
                            hint_time2 = 0
                            hint_time_started = False
                        if not fail_started:
                            fail_started = True
                        if not time_started:
                            time1 = 99999999999
                            time2 = 0
                            time_count = 0
                            time1 = time.time()
                            time_started = True
                        if time_started:
                            time2 = time.time()
                            time_count = time2 - time1
                            if int(time_count) == 3:
                                fail = False
                                move_count = move_count + 1
                                time_started = False
                                # 成功
                                success_time1 = time.time()
                                # 成功
                                success_displayed = True
                                if move_count < 5:  # 重置状态
                                    # 成功
                                    train_started = False

                            doAction(f'持續時間: {time_count:.1f} 秒', font=font_time, font_size=font_size_time,
                                     point=[170, 410])
                            doAction(f'失敗次數: {fail_count} 次', font=font_time, font_size=font_size_time,
                                     point=[140, 440])

                if (Mode == "Normal"):
                    if left_foot:
                        # 新加
                        mouse_move = True
                        cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                        distance = 20
                        pixel_cm = 2.4
                        point_y = target_point[1] - int(distance * pixel_cm)
                        cv2.circle(flipped_image, (target_point[0], point_y), 8, (0, 255, 0), -1)
                        goal_point = (target_point[0], point_y)
                        # 成功
                        success_position = (target_point[0], point_y)
                        # 成功
                        success_direction = -1
                        doAction(f'成功次數: {move_count} / 6', font=font_time, font_size=font_size_time,
                                 point=[170, 0])
                        if hint_time_started:
                            hint_time2 = time.time()
                            hint_time = int(hint_time2 - hint_time1)
                            if hint_time >= 3:
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
                            time_limit_minutes = 10  # 时间限制（分钟）
                            time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                            time2 = time.time()
                            time_count = time2 - time1
                            if time_count > time_limit_seconds:
                                print("Training time exceeded 10 minutes. Exiting program.")
                                break
                        # 新加
                        if mouse_move:
                            doAction("鱷魚出沒！抬起左腳！", font=font, font_size=font_size)
                            mouse_image = readImage('media/croc.png')  # 以彩色模式读取老鼠图像
                            re_image = cv2.resize(mouse_image[:], (90, 90))

                            mouse_height, mouse_width, _ = re_image.shape
                            # 创建一个带有透明通道的空白图像
                            mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)
                            # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                            mouse_with_alpha = re_image
                            # 设置透明通道
                            alpha_threshold = 0  # 设置透明度阈值
                            # 使用蓝色通道作为透明度
                            alpha_channel = re_image[:, :, 1]  # 使用蓝色通道作为透明度
                            mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                            # 直接将带有透明通道的老鼠图像合成到背景图像上
                            alpha = mouse_with_alpha[:, :, 3] / 255.0

                            mouse_position = (mouse_position[0], mouse_position[1] - (mouse_speed * mouse_direction))
                            # 檢查是否越界並改變方向
                            mouse_x = target_point[0] - int(mouse_height / 2)
                            mouse_y = int(mouse_position[1]) - mouse_width

                            if mouse_position[1] - mouse_height <= goal_point[1] or mouse_position[
                                1] == mouse_position_y:
                                mouse_direction *= -1

                            for c in range(3):
                                flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                        (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                        mouse_x:mouse_x + mouse_width, c] +
                                        alpha * mouse_with_alpha[:, :, c]
                                )
                        if fail:
                            if abs(target_y > goal_point[1]):
                                time_started = False
                                if fail_started:
                                    fail_count = fail_count + 1
                                    fail_started = False
                                    fail = False
                        if abs(target_y < goal_point[1]):
                            fail = True
                            if hint_time_started:
                                hint_time = 0
                                hint_time1 = 0
                                hint_time2 = 0
                                hint_time_started = False
                            if not fail_started:
                                fail_started = True
                            if not time_started:
                                time1 = 99999999999
                                time2 = 0
                                time_count = 0
                                time1 = time.time()
                                time_started = True
                            if time_started:
                                time2 = time.time()
                                time_count = time2 - time1
                                if int(time_count) == 5:
                                    fail = False
                                    move_count = move_count + 1
                                    time_started = False
                                    train_started = False
                                    show_text = True
                                    left_foot = False
                                    # 成功
                                    success_time1 = time.time()
                                    # 成功
                                    success_displayed = True
                                    if move_count < 6:  # 重置状态
                                        # 成功
                                        train_started = False

                                doAction(f'持續時間: {time_count:.1f} 秒', font=font_time, font_size=font_size_time,
                                         point=[170, 410])
                                doAction(f'失敗次數: {fail_count} 次', font=font_time, font_size=font_size_time,
                                         point=[140, 440])
                    if not left_foot:
                        # 新加
                        mouse_move = True
                        cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                        distance = 20
                        pixel_cm = 2.4
                        point_y = target_point[1] - int(distance * pixel_cm)
                        cv2.circle(flipped_image, (target_point[0], point_y), 8, (0, 255, 0), -1)
                        goal_point = (target_point[0], point_y)
                        # 成功
                        success_position = (target_point[0], point_y)
                        # 成功
                        success_direction = -1
                        doAction(f'成功次數: {move_count} / 6', font=font_time, font_size=font_size_time,
                                 point=[170, 0])
                        if hint_time_started:
                            hint_time2 = time.time()
                            hint_time = int(hint_time2 - hint_time1)
                            if hint_time >= 3:
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
                            time_limit_minutes = 10  # 时间限制（分钟）
                            time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                            time2 = time.time()
                            time_count = time2 - time1
                            if time_count > time_limit_seconds:
                                print("Training time exceeded 10 minutes. Exiting program.")
                                break
                        # 新加
                        if mouse_move:
                            doAction("鱷魚出沒！抬起右腳！", font=font, font_size=font_size)
                            mouse_image = readImage('media/croc.png')  # 以彩色模式读取老鼠图像
                            re_image = cv2.resize(mouse_image[:], (90, 90))

                            mouse_height, mouse_width, _ = re_image.shape
                            # 创建一个带有透明通道的空白图像
                            mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)
                            # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                            mouse_with_alpha = re_image
                            # 设置透明通道
                            alpha_threshold = 0  # 设置透明度阈值
                            # 使用蓝色通道作为透明度
                            alpha_channel = re_image[:, :, 1]  # 使用蓝色通道作为透明度
                            mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                            # 直接将带有透明通道的老鼠图像合成到背景图像上
                            alpha = mouse_with_alpha[:, :, 3] / 255.0

                            mouse_position = (mouse_position[0], mouse_position[1] - (mouse_speed * mouse_direction))
                            # 檢查是否越界並改變方向
                            mouse_x = target_point[0] - int(mouse_height / 2)
                            mouse_y = int(mouse_position[1]) - mouse_width

                            if mouse_position[1] - mouse_height <= goal_point[1] or mouse_position[
                                1] == mouse_position_y:
                                mouse_direction *= -1

                            for c in range(3):
                                flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                        (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                        mouse_x:mouse_x + mouse_width, c] +
                                        alpha * mouse_with_alpha[:, :, c]
                                )
                        if fail:
                            if abs(target_y1 > goal_point[1]):
                                time_started = False
                                if fail_started:
                                    fail_count = fail_count + 1
                                    fail_started = False
                                    fail = False
                        if abs(target_y1 < goal_point[1]):
                            fail = True
                            if hint_time_started:
                                hint_time = 0
                                hint_time1 = 0
                                hint_time2 = 0
                                hint_time_started = False
                            if not fail_started:
                                fail_started = True
                            if not time_started:
                                time1 = 99999999999
                                time2 = 0
                                time_count = 0
                                time1 = time.time()
                                time_started = True
                            if time_started:
                                time2 = time.time()
                                time_count = time2 - time1
                                if int(time_count) == 5:
                                    fail = False
                                    move_count = move_count + 1
                                    time_started = False
                                    train_started = False
                                    show_text = True
                                    left_foot = True
                                    # 成功
                                    success_time1 = time.time()
                                    # 成功
                                    success_displayed = True
                                    if move_count < 6:  # 重置状态
                                        # 成功
                                        train_started = False

                                doAction(f'持續時間: {time_count:.1f} 秒', font=font_time, font_size=font_size_time,
                                         point=[170, 410])
                                doAction(f'失敗次數: {fail_count} 次', font=font_time, font_size=font_size_time,
                                         point=[140, 440])
                    if move_count == 6:  # 之後傳送數據
                        doAction("結束", font=font, font_size=font_size)
                        t_time2 = time.time()
                        t_time = t_time + (t_time2 - t_time1)
                        plot_data.put(False)
                        break

                if (Mode == "Hard_Left"):
                    cv2.line(flipped_image, (0, target_point[1]), (640, target_point[1]), (255, 0, 0), 5)
                    distance = 5
                    pixel_cm = 2.4
                    point_y = target_point[1] - int(distance * pixel_cm)
                    goal_point = (target_point[0], point_y)
                    # 成功
                    success_position = (target_point[0], point_y)
                    # 成功
                    success_direction = -1
                    doAction(f'成功次數: {move_count} / 3', font=font_time, font_size=font_size_time,
                             point=[170, 0])
                    if hint_time_started:
                        hint_time2 = time.time()
                        hint_time = int(hint_time2-hint_time1)
                        if hint_time >= 3:
                            mp_drawing.draw_landmarks(
                                image=flipped_image,
                                landmark_list=results.pose_landmarks,
                                connections=mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                                             circle_radius=2),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                               circle_radius=2)
                            )
                    left_foot = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                    right_foot = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                    left_knee = int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * flipped_image.shape[0])
                    right_knee = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * flipped_image.shape[0])

                    if move_count == 3:  # 之後傳送數據
                        doAction("結束", font=font, font_size=font_size)
                        t_time2 = time.time()
                        t_time = t_time + (t_time2 - t_time1)
                        plot_data.put(False)
                        break

                    if abs(left_foot - left_knee) > abs(right_foot - right_knee):
                        # 新加
                        if not time_started:
                            time1 = time.time()
                            time_started = True
                        if time_started:
                            time2 = time.time()
                            time_count = time2 - time1
                            if int(time_count) >= 2 and int(time_count) <= 4:
                                doAction("鱷魚將出沒！左腳準備跳！", font=font, font_size=font_size, newline_index=6)
                                if mouse_move:
                                    mouse_image = cv2.imread('media/croc.png', cv2.IMREAD_COLOR)  # 以彩色模式读取老鼠图像
                                    re_image = cv2.resize(mouse_image, (80, 80))

                                    mouse_height, mouse_width, _ = re_image.shape

                                    # 创建一个带有透明通道的空白图像
                                    mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)

                                    # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                                    mouse_with_alpha[:, :, :3] = re_image

                                    # 设置透明通道
                                    alpha_threshold = 0  # 设置透明度阈值
                                    # 使用蓝色通道作为透明度
                                    alpha_channel = re_image[:, :, 1]  # 使用綠色通道作为透明度
                                    mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)

                                    mouse_position = (
                                        mouse_position[0], mouse_position[1] - (mouse_speed * mouse_direction))
                                    # 檢查是否越界並改變方向
                                    mouse_x = target_point[0] - int(mouse_height / 2)
                                    mouse_y = int(mouse_position[1]) - mouse_width

                                    if mouse_position[1] - mouse_height <= goal_point[1] or mouse_position[
                                        1] == mouse_position_y:
                                        mouse_direction *= -1

                                    # 直接将带有透明通道的老鼠图像合成到背景图像上
                                    alpha = mouse_with_alpha[:, :, 3] / 255.0
                                    for c in range(3):
                                        flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width,
                                        c] = (
                                                (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                                mouse_x:mouse_x + mouse_width, c] +
                                                alpha * mouse_with_alpha[:, :, c]
                                        )
                            if int(time_count) >= 5:
                                cv2.line(flipped_image, (0, point_y), (640, point_y), (0, 0, 255), 5)
                                doAction("鱷魚出沒！左腳跳起！", font=font, font_size=font_size)

                                mouse_image = cv2.imread('media/croc.png', cv2.IMREAD_COLOR)  # 以彩色模式读取老鼠图像
                                re_image = cv2.resize(mouse_image, (80, 80))

                                mouse_height, mouse_width, _ = re_image.shape

                                # 创建一个带有透明通道的空白图像
                                mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)

                                # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                                mouse_with_alpha[:, :, :3] = re_image

                                # 设置透明通道
                                alpha_threshold = 0  # 设置透明度阈值
                                # 使用蓝色通道作为透明度
                                alpha_channel = re_image[:, :, 1]  # 使用綠色通道作为透明度
                                mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)

                                # 根据 "mouse_position" 计算绘制的位置
                                mouse_x = 280
                                mouse_y = target_point[1] - mouse_height  # 让老鼠位于定点上方

                                alpha = mouse_with_alpha[:, :, 3] / 255.0
                                for c in range(3):
                                    flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width,
                                    c] = (
                                            (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                            mouse_x:mouse_x + mouse_width, c] +
                                            alpha * mouse_with_alpha[:, :, c]
                                    )

                                if left_foot < point_y:
                                    if hint_time_started:
                                        hint_time = 0
                                        hint_time1 = 0
                                        hint_time2 = 0
                                        hint_time_started = False
                                    move_count = move_count + 1

                                    # 成功
                                    success_time1 = time.time()
                                    # 成功
                                    success_displayed = True
                                    if move_count < 3:  # 重置状态
                                        # 成功
                                        train_started = False
                                        doAction("做得好！", font=font, font_size=font_size)
                                    if not fail_started:
                                        fail_started = True
                                    time_started = False
                                else:
                                    if fail_started:
                                        fail_count = fail_count + 1
                                        fail_started = False

                if (Mode == "Hard_Right"):
                    cv2.line(flipped_image, (0, target_point[1]), (640, target_point[1]), (255, 0, 0), 5)
                    distance = 5
                    pixel_cm = 2.4
                    point_y = target_point[1] - int(distance * pixel_cm)
                    goal_point = (target_point[0], point_y)
                    # 成功
                    success_position = (target_point[0], point_y)
                    # 成功
                    success_direction = -1
                    doAction(f'成功次數: {move_count} / 3', font=font_time, font_size=font_size_time,
                             point=[170, 0])
                    if hint_time_started:
                        hint_time2 = time.time()
                        hint_time = int(hint_time2-hint_time1)
                        if hint_time >= 3:
                            mp_drawing.draw_landmarks(
                                image=flipped_image,
                                landmark_list=results.pose_landmarks,
                                connections=mp_pose.POSE_CONNECTIONS,
                                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2,
                                                                             circle_radius=2),
                                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2,
                                                                               circle_radius=2)
                            )
                    left_foot = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                    right_foot = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                    left_knee = int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * flipped_image.shape[0])
                    right_knee = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * flipped_image.shape[0])

                    if move_count == 3:  # 之後傳送數據
                        doAction("結束", font=font, font_size=font_size)
                        t_time2 = time.time()
                        t_time = t_time + (t_time2 - t_time1)
                        plot_data.put(False)
                        break
                    if abs(right_foot - right_knee) > abs(left_foot - left_knee):
                        # 新加
                        if not time_started:
                            time1 = time.time()
                            time_started = True
                        if time_started:
                            time2 = time.time()
                            time_count = time2 - time1
                            if int(time_count) >= 2 and int(time_count) <= 4:
                                doAction("鱷魚將出沒！右腳準備跳！", font=font, font_size=font_size)
                                if mouse_move:
                                    mouse_image = cv2.imread('media/croc.png', cv2.IMREAD_COLOR)  # 以彩色模式读取老鼠图像
                                    re_image = cv2.resize(mouse_image, (80, 80))

                                    mouse_height, mouse_width, _ = re_image.shape

                                    # 创建一个带有透明通道的空白图像
                                    mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)

                                    # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                                    mouse_with_alpha[:, :, :3] = re_image

                                    # 设置透明通道
                                    alpha_threshold = 0  # 设置透明度阈值
                                    # 使用蓝色通道作为透明度
                                    alpha_channel = re_image[:, :, 1]  # 使用綠色通道作为透明度
                                    mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)

                                    mouse_position = (
                                        mouse_position[0], mouse_position[1] - (mouse_speed * mouse_direction))
                                    # 檢查是否越界並改變方向
                                    mouse_x = target_point[0] - int(mouse_height / 2)
                                    mouse_y = int(mouse_position[1]) - mouse_width

                                    if mouse_position[1] - mouse_height <= goal_point[1] or mouse_position[
                                        1] == mouse_position_y:
                                        mouse_direction *= -1

                                    alpha = mouse_with_alpha[:, :, 3] / 255.0
                                    for c in range(3):
                                        flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width,
                                        c] = (
                                                (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                                mouse_x:mouse_x + mouse_width, c] +
                                                alpha * mouse_with_alpha[:, :, c]
                                        )
                            if int(time_count) >= 5:
                                cv2.line(flipped_image, (0, point_y), (640, point_y), (0, 0, 255), 5)
                                doAction("鱷魚出沒！右腳跳起！", font=font, font_size=font_size)

                                mouse_image = cv2.imread('media/croc.png', cv2.IMREAD_COLOR)  # 以彩色模式读取老鼠图像
                                re_image = cv2.resize(mouse_image, (80, 80))

                                mouse_height, mouse_width, _ = re_image.shape

                                # 创建一个带有透明通道的空白图像
                                mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)

                                # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                                mouse_with_alpha[:, :, :3] = re_image

                                # 设置透明通道
                                alpha_threshold = 0  # 设置透明度阈值
                                # 使用蓝色通道作为透明度
                                alpha_channel = re_image[:, :, 1]  # 使用綠色通道作为透明度
                                mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)

                                # 根据 "mouse_position" 计算绘制的位置
                                mouse_x = 320
                                mouse_y = target_point[1] - mouse_height  # 让老鼠位于定点上方

                                alpha = mouse_with_alpha[:, :, 3] / 255.0
                                for c in range(3):
                                    flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width,
                                    c] = (
                                            (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                            mouse_x:mouse_x + mouse_width, c] +
                                            alpha * mouse_with_alpha[:, :, c]
                                    )

                                if right_foot < point_y:
                                    move_count = move_count + 1
                                    if hint_time_started:
                                        hint_time = 0
                                        hint_time1 = 0
                                        hint_time2 = 0
                                        hint_time_started = False
                                    # 成功
                                    success_time1 = time.time()
                                    # 成功
                                    success_displayed = True
                                    if move_count < 3:  # 重置状态
                                        # 成功
                                        train_started = False
                                        doAction("做得好！", font=font, font_size=font_size)
                                    if not fail_started:
                                        fail_started = True
                                    time_started = False
                                else:
                                    if fail_started:
                                        fail_count = fail_count + 1
                                        fail_started = False

            # 顯示圖像
            plot_data.put(flipped_image)

        cap.release()
        cv2.destroyAllWindows()

        def import_data(user, move_count, fail_count, t_time, difficulty, grade):
            training_data = TrainingOneFoot(user=user, move_count=move_count, fail_count=fail_count, t_time=t_time,
                                            difficulty=difficulty, grade=grade)
            training_data.save()

        if Mode == "VeryEasy_Left" or Mode == "VeryEasy_Right":
            if move_count == 3 and t_time <= 60:
                grade = 'A'
            elif move_count == 3 and t_time > 60 and t_time <= 100:
                grade = 'B'
            else:
                grade = 'C'
        elif Mode == "Easy_Left" or Mode == "Easy_Right":
            if move_count == 5 and t_time <= 100:
                grade = 'A'
            elif move_count == 5 and t_time > 100 and t_time <= 140:
                grade = 'B'
            else:
                grade = 'C'
        elif Mode == "Normal":
            if move_count == 6 and t_time <= 140:
                grade = 'A'
            elif move_count == 6 and t_time > 140 and t_time <= 180:
                grade = 'B'
            else:
                grade = 'C'
        elif Mode == "Hard_Left" or Mode == "Hard_Right":
            if move_count == 3 and t_time <= 180:
                grade = 'A'
            elif move_count == 3 and t_time > 180 and t_time <= 220:
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
            "move_count": 0,
            "fail_count": 0,
            "time": "",
            "difficulty": Mode,
            "grade": ""
        }
        test_results["move_count"] = move_count
        test_results["fail_count"] = fail_count
        test_results["t_time"] = t_time
        test_results["difficulty"] = difficulty
        test_results["grade"] = grade

        import_data(user, test_results["move_count"], test_results["fail_count"], test_results["t_time"],
                    test_results["difficulty"],
                    test_results["grade"])

        plot_data.put(False)

def main(user, mode, background_name):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "pythonProject3.settings")
    django.setup()
    global plot_data
    plot_data = queue.LifoQueue()

    if (mode == 'Hard_Left'):
        background_name = 'background_images/back_onefoot_Left.jpg'
    elif (mode == 'Hard_Right'):
        background_name = 'background_images/back_onefoot_Right.jpg'

    try:  # server端
        background_path = 'media/' + background_name
        open(background_path, 'r+')
    except:  # 離線端
        background_path = '../media/' + background_name
        open(background_path, 'r+')
    thread = Thread(target=onefoot_Per, args=(user, mode, background_path), daemon=True)
    thread.start()

    while plot_data.empty():
        pass
    while True:
        data = plot_data.get()
        if type(data) == type(False) or cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break
        cv2.imshow('onefoot', data)

if __name__ == '__main__':
    main('stest', 'Easy_Left',  'background_images/remove.png')