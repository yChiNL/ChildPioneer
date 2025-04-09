import random
import random
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import django
from threading import Thread
import queue
import copy
import pygame
import math
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


def calculateTextPoint(show_text, center_point, font_size, vertical_offset=0):
    text_len = len(show_text)
    start_point = copy.copy(center_point)
    start_point[0] = int(center_point[0] - text_len / 2 * font_size)
    start_point[1] = int(center_point[1] - font_size / 2 - vertical_offset)
    return tuple(start_point)


def calculatePose(landmarks, mp_pose, flipped_image):
    l_elbow_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * flipped_image.shape[1])
    l_elbow_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * flipped_image.shape[0])
    r_elbow_x = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * flipped_image.shape[1])
    r_elbow_y = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * flipped_image.shape[0])
    l_shoulder_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * flipped_image.shape[0])
    r_shoulder_y = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * flipped_image.shape[0])
    return l_elbow_x, l_elbow_y, r_elbow_x, r_elbow_y, l_shoulder_y, r_shoulder_y


def win(end=False):
    global flipped_image, center_point, font_size, move_count, m_started, time_started, font
    pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    show_text = "成功"
    draw.text(calculateTextPoint(show_text, center_point, font_size,), show_text, fill=(0, 255, 0), font=font)
    flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    if end:
        move_count = move_count + 1
        m_started = False
        time_started = False


def doAction(show_text, font, font_size, newline_index=None, point=None):
    global flipped_image, center_point
    pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    if point is not None:
        center = point
    else:
        center = center_point
    if newline_index is not None:
        draw.text(calculateTextPoint(show_text[:newline_index], [center_point[0], center_point[1] - int(font_size / 2)],
                                     font_size), show_text[:newline_index], fill=(0, 255, 0), font=font)
        draw.text(calculateTextPoint(show_text[newline_index:], [center_point[0], center_point[1] + int(font_size / 2)],
                                     font_size), show_text[newline_index:], fill=(0, 255, 0), font=font)
    else:
        draw.text(calculateTextPoint(show_text, center, font_size), show_text, fill=(0, 255, 0), font=font)
    flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def readImage(path):
    try:
        open(path)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except:
        image = cv2.imread('../' + path, cv2.IMREAD_UNCHANGED)

    return image


def dance_Per(user, mode, background_path):
    global flipped_image, center_point, font_size, font_size_time, move_count, m_started, time_started, font, font_time, state

    from ChildPioneer.models import TrainingDance, CustomUser

    # 初始化 MediaPipe 的姿态估计模型
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    pygame.mixer.init()
    try:
        pygame.mixer.music.load("music/funny-tropical-house.mp3")
    except:
        pygame.mixer.music.load("../music/funny-tropical-house.mp3")

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
    time_started = False
    show_text = True
    time1 = 99999999999
    time2 = 0
    t_time1 = 0
    t_time2 = 0
    t_time = 0
    time_count = 0
    count = 0
    left_foot = True
    left_elbow_x = None
    right_elbow_x = None
    left_elbow_y = None
    right_elbow_y = None
    state = False
    movement = ['Dance_LeftFly', 'Dance_RightFly', 'Dance_FlFl', 'Dance_UpUp', 'Dance_DoDo', 'Dance_UpDo', 'Dance_DoUp',
                'Dance_UpUp_wrist', 'Dance_DoDo_wrist', 'Dance_UpDo_wrist',
                'Dance_DoUp_wrist']
    m_started = False
    move_count = 0
    Mode = mode
    hint_time1 = 0
    hint_time2 = 0
    hint_time = 0

    # 定點座標 (假設定點在畫面中央)
    target_point = (320, 460)
    center_point = [320, 205]
    time_point = (0, 400)

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

    def hint(time1):
        hint_time2 = time.time()
        hint_time = int(hint_time2 - time1)
        if hint_time >= 5:
            mp_drawing.draw_landmarks(
                image=flipped_image,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

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
            window_height, window_width, _ = flipped_image.shape
            image_w = 150
            image_h = 150
            image_x = window_width - image_w
            image_y = window_height - image_h

            # 檢測使用者是否站在定點上
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                target_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * flipped_image.shape[1])
                target_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                l_wrist_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x * flipped_image.shape[1])
                l_wrist_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y * flipped_image.shape[0])
                l_hip_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * flipped_image.shape[1])
                l_distance_x = abs(l_wrist_x - l_hip_x)
                r_wrist_x = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * flipped_image.shape[1])
                r_wrist_y = int(landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * flipped_image.shape[0])
                r_hip_x = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * flipped_image.shape[1])
                r_distance_x = abs(r_hip_x - r_wrist_x)
                if abs(target_x - target_point[0]) < 10 and abs(
                        target_y - target_point[1]) < 10:
                    show_text = False
                    left_elbow_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x * flipped_image.shape[1])
                    right_elbow_x = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * flipped_image.shape[1])
                    left_elbow_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y * flipped_image.shape[0])
                    right_elbow_y = int(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * flipped_image.shape[0])
                    pygame.mixer.music.play(-1)  # -1 表示循環播放音樂

                    t_time1 = time.time()  # 算訓練時長

                    if not train_started:
                        train_started = True
                else:
                    if show_text:
                        cv2.circle(flipped_image, target_point, 10, (0, 255, 0), -1)
                        pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        draw.text(calculateTextPoint("請站在點上", center_point, font_size, vertical_offset=100), "請站在點上",
                                  fill=(0, 255, 0), font=font)
                        flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # 顯示計時器
            if train_started:
                if results.pose_landmarks:
                    # 取得畫面寬高
                    image_height, image_width, _ = flipped_image.shape

                if (Mode == "VeryEasy"):
                    doAction(f"成功次數 {move_count} / 3", point=[150, 410], font=font_time, font_size=font_size_time)
                    l_elbow_x, l_elbow_y, r_elbow_x, r_elbow_y, l_shoulder_y, r_shoulder_y = calculatePose(landmarks,
                                                                                                           mp_pose,
                                                                                                           flipped_image)
                    if time_started:
                        time_limit_minutes = 6  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 6 minutes. Exiting program.")
                            break
                    if not m_started:
                        hint_time1 = 0
                        hint_time2 = 0
                        hint_time = 0
                        if move_count == 3:
                            t_time2 = time.time()
                            t_time = t_time2 - t_time1
                            plot_data.put(False)
                            break
                        if not time_started:
                            time1 = time.time()
                            time_started = True
                        if time_started:
                            time2 = time.time()
                            time_count = time2 - time1
                            cv2.putText(flipped_image, str(int(time_count)) + 'sec', time_point,
                                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0), 2)
                            if int(time_count) == 3:
                                move = int(random.randrange(0, 3))
                                # cv2.putText(flipped_image, movement[move], (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                #            (0, 255, 0), 2)
                                hint_time1 = time.time()
                                m_started = True
                                win_time = None

                    if m_started:
                        hint(hint_time1)
                        if movement[move] == 'Dance_FlFl':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image

                            if l_elbow_y <= abs(l_shoulder_y + 30) and l_elbow_y >= abs(
                                    l_shoulder_y - 30) and l_wrist_y <= abs(l_elbow_y + 30) and r_wrist_y >= abs(
                                r_elbow_y - 30) and l_wrist_x < l_elbow_x and r_wrist_x > r_elbow_x or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("雙手平舉", font=font, font_size=font_size)
                        if movement[move] == 'Dance_RightFly':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image

                            if (l_wrist_y < l_elbow_y < l_shoulder_y and
                                r_wrist_y > r_elbow_y > r_shoulder_y) or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("正確地舉起雙手", font=font, font_size=font_size)
                        if movement[move] == 'Dance_LeftFly':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image

                            if (l_wrist_y > l_elbow_y > l_shoulder_y and
                                r_wrist_y < r_elbow_y < r_shoulder_y) or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()

                            else:
                                doAction("正確的舉起你的雙手", font=font, font_size=font_size)

                if (Mode == "Easy"):
                    doAction(f"成功次數 {move_count} / 5", point=[150, 410], font=font_time, font_size=font_size_time)
                    l_elbow_x, l_elbow_y, r_elbow_x, r_elbow_y, l_shoulder_y, r_shoulder_y = calculatePose(landmarks,
                                                                                                           mp_pose,
                                                                                                           flipped_image)
                    if time_started:
                        time_limit_minutes = 8  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 8 minutes. Exiting program.")
                            break
                    if not m_started:
                        hint_time1 = 0
                        hint_time2 = 0
                        hint_time = 0
                        if move_count == 5:
                            t_time2 = time.time()
                            t_time = t_time2 - t_time1
                            plot_data.put(False)
                            break
                        if not time_started:
                            time1 = time.time()
                            time_started = True
                        if time_started:
                            time2 = time.time()
                            time_count = time2 - time1
                            cv2.putText(flipped_image, str(int(time_count)) + 'sec', time_point,
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            if int(time_count) == 3:
                                move = int(random.randrange(3, 5))
                                # cv2.putText(flipped_image, movement[move], (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                #            (0, 255, 0), 2)
                                hint_time1 = time.time()
                                m_started = True
                                win_time = None

                    if m_started:
                        hint(hint_time1)
                        if movement[move] == 'Dance_UpUp':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image

                            if l_elbow_y <= abs(l_shoulder_y + 30) and l_elbow_y >= abs(
                                    l_shoulder_y - 30) and l_wrist_x <= abs(l_elbow_x + 30) and l_wrist_x >= abs(
                                l_elbow_x - 30) and l_wrist_y < l_elbow_y and r_elbow_y <= abs(
                                r_shoulder_y + 30) and r_elbow_y >= abs(r_shoulder_y - 30) and r_wrist_x <= abs(
                                r_elbow_x + 30) and r_wrist_x >= abs(
                                r_elbow_x - 30) and r_wrist_y < r_elbow_y or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("舉起你的雙手", font=font, font_size=font_size)
                        if movement[move] == 'Dance_DoDo':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resized_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resized_image

                            if l_elbow_y <= abs(l_shoulder_y + 30) and l_elbow_y >= abs(
                                    l_shoulder_y - 30) and l_wrist_x <= abs(l_elbow_x + 30) and l_wrist_x >= abs(
                                l_elbow_x - 30) and l_wrist_y > l_elbow_y and r_elbow_y <= abs(
                                r_shoulder_y + 30) and r_elbow_y >= abs(r_shoulder_y - 30) and r_wrist_x <= abs(
                                r_elbow_x + 30) and r_wrist_x >= abs(
                                r_elbow_x - 30) and r_wrist_y > r_elbow_y or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("放下你的雙手", font=font, font_size=font_size)

                if (Mode == "Normal"):
                    doAction(f"成功次數 {move_count} / 5", point=[150, 410], font=font_time, font_size=font_size_time)
                    l_elbow_x, l_elbow_y, r_elbow_x, r_elbow_y, l_shoulder_y, r_shoulder_y = calculatePose(landmarks,
                                                                                                           mp_pose,
                                                                                                           flipped_image)
                    if time_started:
                        time_limit_minutes = 10  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 10 minutes. Exiting program.")
                            break
                    if not m_started:
                        hint_time1 = 0
                        hint_time2 = 0
                        hint_time = 0
                        if move_count == 5:
                            t_time2 = time.time()
                            t_time = int(t_time2 - t_time1)
                            plot_data.put(False)
                            break
                        if not time_started:
                            time1 = time.time()
                            time_started = True
                        if time_started:
                            time2 = time.time()
                            time_count = time2 - time1
                            cv2.putText(flipped_image, str(int(time_count)) + 'sec', time_point,
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            if int(time_count) == 3:
                                move = int(random.randrange(5, 7))
                                # cv2.putText(flipped_image, movement[move], (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                #            (0, 255, 0), 2)
                                hint_time1 = time.time()
                                m_started = True
                                win_time = None

                    if m_started:
                        hint(hint_time1)
                        if movement[move] == 'Dance_UpDo':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image

                            if l_elbow_y <= abs(l_shoulder_y + 30) and l_elbow_y >= abs(
                                    l_shoulder_y - 30) and l_wrist_x <= abs(l_elbow_x + 30) and l_wrist_x >= abs(
                                l_elbow_x - 30) and l_wrist_y < l_elbow_y and r_elbow_y <= abs(
                                r_shoulder_y + 30) and r_elbow_y >= abs(r_shoulder_y - 30) and r_wrist_x <= abs(
                                r_elbow_x + 30) and r_wrist_x >= abs(
                                r_elbow_x - 30) and r_wrist_y > r_elbow_y or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("請左手向上右手向下", font=font, font_size=font_size)
                        if movement[move] == 'Dance_DoUp':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image

                            if l_elbow_y <= abs(l_shoulder_y + 30) and l_elbow_y >= abs(
                                    l_shoulder_y - 30) and l_wrist_x <= abs(l_elbow_x + 30) and l_wrist_x >= abs(
                                l_elbow_x - 30) and l_wrist_y > l_elbow_y and r_elbow_y <= abs(
                                r_shoulder_y + 30) and r_elbow_y >= abs(r_shoulder_y - 30) and r_wrist_x <= abs(
                                r_elbow_x + 30) and r_wrist_x >= abs(
                                r_elbow_x - 30) and r_wrist_y < r_elbow_y or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("請左手向下右手向上", font=font, font_size=font_size)

                if (Mode == "Hard"):
                    doAction(f"成功次數 {move_count} / 8", point=[150, 410], font=font_time, font_size=font_size_time)
                    l_elbow_x, l_elbow_y, r_elbow_x, r_elbow_y, l_shoulder_y, r_shoulder_y = calculatePose(landmarks,
                                                                                                           mp_pose,
                                                                                                           flipped_image)
                    if time_started:
                        time_limit_minutes = 12  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 12 minutes. Exiting program.")
                            break
                    if not m_started:
                        hint_time1 = 0
                        hint_time2 = 0
                        hint_time = 0
                        if move_count == 7:
                            t_time2 = time.time()
                            t_time = int(t_time2 - t_time1)
                            plot_data.put(False)
                            break
                        if not time_started:
                            time1 = time.time()
                            time_started = True
                        if time_started:
                            time2 = time.time()
                            time_count = time2 - time1
                            cv2.putText(flipped_image, str(int(time_count)) + 'sec', time_point,
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            if int(time_count) == 3:
                                move = int(random.randrange(3, 7))
                                # cv2.putText(flipped_image, movement[move], (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                #            (0, 255, 0), 2)
                                hint_time1 = time.time()
                                m_started = True
                                win_time = None
                    if m_started:
                        hint(hint_time1)
                        if movement[move] == 'Dance_UpUp':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image

                            if l_elbow_y <= abs(l_shoulder_y + 30) and l_elbow_y >= abs(
                                    l_shoulder_y - 30) and l_wrist_x <= abs(l_elbow_x + 30) and l_wrist_x >= abs(
                                l_elbow_x - 30) and l_wrist_y < l_elbow_y and r_elbow_y <= abs(
                                r_shoulder_y + 30) and r_elbow_y >= abs(r_shoulder_y - 30) and r_wrist_x <= abs(
                                r_elbow_x + 30) and r_wrist_x >= abs(
                                r_elbow_x - 30) and r_wrist_y < r_elbow_y or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("舉起你的雙手", font=font, font_size=font_size)
                        if movement[move] == 'Dance_DoDo':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image

                            if l_elbow_y <= abs(l_shoulder_y + 30) and l_elbow_y >= abs(
                                    l_shoulder_y - 30) and l_wrist_x <= abs(l_elbow_x + 30) and l_wrist_x >= abs(
                                l_elbow_x - 30) and l_wrist_y > l_elbow_y and r_elbow_y <= abs(
                                r_shoulder_y + 30) and r_elbow_y >= abs(r_shoulder_y - 30) and r_wrist_x <= abs(
                                r_elbow_x + 30) and r_wrist_x >= abs(
                                r_elbow_x - 30) and r_wrist_y > r_elbow_y or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("放下你的雙手", font=font, font_size=font_size)
                        if movement[move] == 'Dance_UpDo':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image

                            if l_elbow_y <= abs(l_shoulder_y + 30) and l_elbow_y >= abs(
                                    l_shoulder_y - 30) and l_wrist_x <= abs(l_elbow_x + 30) and l_wrist_x >= abs(
                                l_elbow_x - 30) and l_wrist_y < l_elbow_y and r_elbow_y <= abs(
                                r_shoulder_y + 30) and r_elbow_y >= abs(r_shoulder_y - 30) and r_wrist_x <= abs(
                                r_elbow_x + 30) and r_wrist_x >= abs(r_elbow_x - 30) and r_wrist_y > r_elbow_y or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("請左手向上右手向下", font=font, font_size=font_size)
                        if movement[move] == 'Dance_DoUp':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image

                            if l_elbow_y <= abs(l_shoulder_y + 30) and l_elbow_y >= abs(
                                    l_shoulder_y - 30) and l_wrist_x <= abs(l_elbow_x + 30) and l_wrist_x >= abs(
                                l_elbow_x - 30) and l_wrist_y > l_elbow_y and r_elbow_y <= abs(
                                r_shoulder_y + 30) and r_elbow_y >= abs(r_shoulder_y - 30) and r_wrist_x <= abs(
                                r_elbow_x + 30) and r_wrist_x >= abs(r_elbow_x - 30) and r_wrist_y < r_elbow_y or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("請左手向上右手向下", font=font, font_size=font_size)

                if (Mode == "VeryHard"):
                    doAction(f"成功次數 {move_count} / 10", point=[150, 410], font=font_time, font_size=font_size_time)
                    l_elbow_x, l_elbow_y, r_elbow_x, r_elbow_y, l_shoulder_y, r_shoulder_y = calculatePose(landmarks,
                                                                                                           mp_pose,
                                                                                                           flipped_image)
                    l_index_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_INDEX].x * flipped_image.shape[1])
                    r_index_x = int(landmarks[mp_pose.PoseLandmark.LEFT_INDEX].x * flipped_image.shape[1])
                    l_pinky_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_PINKY].x * flipped_image.shape[1])
                    r_pinky_x = int(landmarks[mp_pose.PoseLandmark.LEFT_PINKY].x * flipped_image.shape[1])
                    l_thumb_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_THUMB].x * flipped_image.shape[1])
                    r_thumb_x = int(landmarks[mp_pose.PoseLandmark.LEFT_THUMB].x * flipped_image.shape[1])
                    #cv2.putText(flipped_image, Index:'+str(r_index_x), (300,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #cv2.putText(flipped_image, 'Pinky:'+str(r_pinky_x), (300,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
                    #cv2.putText(flipped_image, 'Thunb:'+str(r_thumb_x), (300,150), cv2.FONT_HERSHEY_SIMPLEX, 1,(0, 255, 0), 2)
                    if time_started:
                        time_limit_minutes = 14  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 14 minutes. Exiting program.")
                            break
                    if not m_started:
                        hint_time1 = 0
                        hint_time2 = 0
                        hint_time = 0
                        if move_count == 10:
                            t_time2 = time.time()
                            t_time = int(t_time2 - t_time1)
                            plot_data.put(False)
                            break
                        if not time_started:
                            time1 = time.time()
                            time_started = True
                        if time_started:
                            time2 = time.time()
                            time_count = time2 - time1
                            cv2.putText(flipped_image, str(int(time_count)) + 'sec', time_point,
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            if int(time_count) == 3:
                                move = int(random.randrange(7, 11))
                                # cv2.putText(flipped_image, movement[move], (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                #            (0, 255, 0), 2)
                                hint_time1 = time.time()
                                m_started = True
                                win_time = None
                    if m_started:
                        hint(hint_time1)
                        if movement[move] == 'Dance_UpUp_wrist':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image


                            if (l_elbow_y <= abs(l_shoulder_y + 30) and l_elbow_y >= abs(l_shoulder_y - 30) and
                                l_wrist_x <= abs(l_elbow_x + 30) and l_wrist_x >= abs(l_elbow_x - 30) and
                                l_wrist_y < l_elbow_y and
                                r_elbow_y <= abs(r_shoulder_y + 30) and r_elbow_y >= abs(r_shoulder_y - 30) and
                                r_wrist_x <= abs(r_elbow_x + 30) and r_wrist_x >= abs(r_elbow_x - 30) and
                                r_wrist_y < r_elbow_y and r_pinky_x > r_thumb_x and r_pinky_x > r_index_x and r_index_x <= r_thumb_x and l_index_x <= l_thumb_x and l_pinky_x < l_index_x and l_pinky_x < l_thumb_x) or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("舉起你的雙手並手腕內凹", font=font, font_size=font_size)
                        if movement[move] == 'Dance_DoDo_wrist':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image
                            # Additional code to get palm positions

                            if (l_elbow_y <= abs(l_shoulder_y + 30) and l_elbow_y >= abs(
                                    l_shoulder_y - 30) and l_wrist_x <= abs(l_elbow_x + 30) and l_wrist_x >= abs(
                                l_elbow_x - 30) and l_wrist_y > l_elbow_y and r_elbow_y <= abs(
                                r_shoulder_y + 30) and r_elbow_y >= abs(r_shoulder_y - 30) and r_wrist_x <= abs(
                                r_elbow_x + 30) and r_wrist_x >= abs(
                                r_elbow_x - 30) and r_wrist_y > r_elbow_y and r_pinky_x > r_thumb_x and r_pinky_x > r_index_x and l_pinky_x < l_index_x and l_pinky_x < l_thumb_x and r_index_x <= r_thumb_x and l_index_x <= l_thumb_x) or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("放下你的雙手並手腕內凹", font=font, font_size=font_size)
                        if movement[move] == 'Dance_UpDo_wrist':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image
                            # Additional code to get palm positions

                            if (l_elbow_y <= abs(l_shoulder_y + 30) and l_elbow_y >= abs(
                                    l_shoulder_y - 30) and l_wrist_x <= abs(l_elbow_x + 30) and l_wrist_x >= abs(
                                l_elbow_x - 30) and l_wrist_y < l_elbow_y and r_elbow_y <= abs(
                                r_shoulder_y + 30) and r_elbow_y >= abs(r_shoulder_y - 30) and r_wrist_x <= abs(
                                r_elbow_x + 30) and r_wrist_x >= abs(
                                r_elbow_x - 30) and r_wrist_y > r_elbow_y and r_pinky_x > r_thumb_x and r_pinky_x > r_index_x and l_pinky_x < l_index_x and l_pinky_x < l_thumb_x and r_index_x <= r_thumb_x and l_index_x <= l_thumb_x) or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("請左手向上右手向下並手腕內凹", font=font, font_size=font_size,
                                         newline_index=5)
                        if movement[move] == 'Dance_DoUp_wrist':
                            image_to_insert = readImage('media/' + movement[move] + '.jpg')
                            resize_image = cv2.resize(image_to_insert, (image_w, image_h))
                            flipped_image[image_y:image_y + image_h, image_x:image_x + image_w] = resize_image
                            # Additional code to get palm positions

                            if (l_elbow_y <= abs(l_shoulder_y + 30) and l_elbow_y >= abs(
                                    l_shoulder_y - 30) and l_wrist_x <= abs(l_elbow_x + 30) and l_wrist_x >= abs(
                                l_elbow_x - 30) and l_wrist_y > l_elbow_y and r_elbow_y <= abs(
                                r_shoulder_y + 30) and r_elbow_y >= abs(r_shoulder_y - 30) and r_wrist_x <= abs(
                                r_elbow_x + 30) and r_wrist_x >= abs(
                                r_elbow_x - 30) and r_wrist_y < r_elbow_y and r_pinky_x > r_thumb_x and r_pinky_x > r_index_x and l_pinky_x < l_index_x and l_pinky_x < l_thumb_x and r_index_x <= r_thumb_x and l_index_x <= l_thumb_x) or state:
                                if win_time is not None:
                                    win()
                                    if time.time() - win_time >= 2:
                                        win(True)
                                        state = False
                                else:
                                    state = True
                                    win_time = time.time()
                            else:
                                doAction("請左手向下右手向上並手腕內凹", font=font, font_size=font_size,
                                         newline_index=5)
            # 顯示圖像
            plot_data.put(flipped_image)

        # 停止音樂
        pygame.mixer.music.stop()

        def import_data(user, move_count, t_time, difficulty, grade):
            training_data = TrainingDance(user=user, move_count=move_count, t_time=t_time, difficulty=difficulty,
                                          grade=grade)
            training_data.save()

        if Mode == "VeryEasy":
            if move_count == 3 and t_time <= 60:
                grade = 'A'
            elif move_count == 3 and t_time > 60 and t_time <= 100:
                grade = 'B'
            else:
                grade = 'C'
        elif Mode == "Easy":
            if move_count == 5 and t_time <= 100:
                grade = 'A'
            elif move_count == 5 and t_time > 100 and t_time <= 140:
                grade = 'B'
            else:
                grade = 'C'
        elif Mode == "Normal":
            if move_count == 5 and t_time <= 140:
                grade = 'A'
            elif move_count == 5 and t_time > 140 and t_time <= 180:
                grade = 'B'
            else:
                grade = 'C'
        elif Mode == "Hard":
            if move_count == 7 and t_time <= 180:
                grade = 'A'
            elif move_count == 7 and t_time > 180 and t_time <= 220:
                grade = 'B'
            else:
                grade = 'C'

        elif Mode == "VeryHard":
            if move_count == 10 and t_time <= 220:
                grade = 'A'
            elif move_count >= 10 and t_time > 220 and t_time <= 260:
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
            "t_time": "",
            "difficulty": Mode,
            "grade": ""
        }
        test_results["move_count"] = move_count
        test_results["t_time"] = t_time
        test_results["difficulty"] = difficulty
        test_results["grade"] = grade

        import_data(user, test_results["move_count"], test_results["t_time"], test_results["difficulty"],
                    test_results["grade"])
        plot_data.put(False)


def main(user, mode, background_name):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ChildPioneer.settings")
    django.setup()
    global plot_data
    plot_data = queue.LifoQueue()
    try:  # server端
        background_path = 'media/' + background_name
        open(background_path, 'r+')
    except:  # 離線端
        background_path = '../media/background_images/' + background_name
        open(background_path, 'r+')

    thread = Thread(target=dance_Per, args=(user, mode, background_path), daemon=True)
    thread.start()

    while plot_data.empty():
        pass
    while True:
        data = plot_data.get()
        if type(data) == type(False) or cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break
        cv2.imshow('dance', data)


if __name__ == '__main__':
    main('stest', 'VeryEasy', 'remove.png')
