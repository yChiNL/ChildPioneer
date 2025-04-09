import cv2
import mediapipe as mp
import time
import math
import os
import django
import numpy as np
import copy
from threading import Thread
import queue
from PIL import ImageFont, ImageDraw, Image  # 載入 PIL 相關函式庫


t_time1 = 0
t_time2 = 0
t_time = 0

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


def readImage(path):
    try:
        open(path)
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except:
        image = cv2.imread('../' + path, cv2.IMREAD_UNCHANGED)

    return image


def jump_Ther(user, mode, end, id,background_path):
    from ChildPioneer.models import TreatmentContent
    from ChildPioneer.models import CustomUser
    from ChildPioneer.models import TrainingJump

    # 初始化 MediaPipe 的姿態估計模型
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    distance = None
    pixel_cm = None

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
    show_text = True
    time_started = False
    text_left = True
    text_right = True
    Jump_Right = False
    Mode = mode
    countdown = 30
    t1 = None
    run_count = 0
    time1 = 99999999999
    time2 = 0
    count = 0
    end_count = end

    t_timestarted = False
    mouse_move_easy = False
    mtv = True
    move_count = 0
    # 老鼠的移動速度
    mouse_speed_easy = 5
    # 老鼠的移動方向，1 表示向右，-1 表示向左
    mouse_direction_easy = 1

    success_displayed = False
    success_display_start_time = 0
    success_display_duration = 3  # 切换图片显示时间，单位：秒

    run_OK = False
    x = True
    # 老鼠的起始位置 (x, y)
    mouse_position = (320, 480)
    # 老鼠的移動速度
    mouse_speed = 2
    mouse_speed1 = 1.2
    # 老鼠的移動方向，1 表示向右，-1 表示向左
    mouse_direction1 = -1
    # 老鼠的移動方向，1 表示向下，-1 表示向上
    mouse_direction2 = -1
    direction_change = 0
    mouse_move = False
    fg = False
    # 定點座標
    target_point = (320, 400)
    center_point = [320, 205]

    if Mode == "Hard":
        mouse_speed = 3
        mouse_speed1 = 1
        distance = 37
        pixel_cm = 2.4
        target_point = (400, 420)
        mouse_position = (450, 420)
    if Mode == "VeryHard":
        mouse_speed = 4
        mouse_speed1 = 1
        target_point = (400, 420)
        mouse_position = (450, 420)

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
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # 檢測使用者是否站在定點上
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                target_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * flipped_image.shape[1])
                target_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                target_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                target_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                if abs(target_x - target_point[0]) < 10 and abs(target_y - target_point[1]) < 10:
                    show_text = False
                    mouse_move_easy = True
                    if not hint_time_started:
                        hint_time1 = time.time()
                        hint_time_started = True
                    if not t_timestarted:
                        t_time1 = time.time()
                        t_timestarted = True
                    if Mode == 'Hard' and x or Mode == 'VeryHard' and x:
                        if not mouse_move:
                            mouse_move = True
                            fg = True
                    if not time_started:
                        time1 = time.time()
                        time_started = True
                    if not train_started:
                        train_started = True
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
                        cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                        pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        draw.text(calculateTextPoint("請站在點上", center_point, font_size, vertical_offset=100), "請站在點上",
                                  fill=(0, 255, 0), font=font)
                        flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

            # 顯示計時器
            if train_started:
                if (Mode == "VeryEasy"):
                    cv2.line(flipped_image, (0, target_point[1]), (640, target_point[1]), (0, 255, 0), 5)
                    distance = 10
                    pixel_cm = 2.4
                    point_y = target_point[1] - int(distance * pixel_cm)
                    cv2.line(flipped_image, (0, point_y), (640, point_y), (0, 0, 255), 5)
                    goal_point = (target_point[0], point_y)
                    target_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                    target_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                    pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    draw.text((25, 50), "成功次數 " + str(move_count) + "/" + str(end_count), fill=(0, 255, 0), font=font_time)
                    flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    img = flipped_image
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
                    if move_count == end_count:  # 之後傳送數據
                        pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        draw.text((150, 150), "結束", fill=(0, 255, 0), font=font)
                        t_time2 = time.time()
                        t_time = int(t_time2 - t_time1)
                        plot_data.put(False)
                        break
                    if time_started:
                        time_limit_minutes = 6  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 6 minutes. Exiting program.")
                            break

                    if mouse_move_easy:
                        if mtv:
                            pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_image)
                            draw.text(calculateTextPoint("跳過老鼠!", center_point, font_size), "跳過老鼠!",
                                      fill=(0, 255, 0), font=font)

                            mouse_image = cv2.imread('media/jump_mouse.png', cv2.IMREAD_COLOR)  # 以彩色模式读取老鼠图像
                            re_image = cv2.resize(mouse_image[:], (100, 100))
                            mouse_height, mouse_width, _ = re_image.shape
                            # 创建一个带有透明通道的空白图像
                            mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)
                            # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                            mouse_with_alpha[:, :, :3] = re_image
                            # 设置透明通道
                            alpha_threshold = 0  # 设置透明度阈值
                            # 使用蓝色通道作为透明度
                            alpha_channel = re_image[:, :, 0]  # 使用蓝色通道作为透明度
                            mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                            # 直接将带有透明通道的老鼠图像合成到背景图像上
                            alpha = mouse_with_alpha[:, :, 3] / 255.0

                            # 根据 "mouse_position" 计算绘制的位置
                            mouse_x = 310
                            mouse_y = 370  # 让老鼠位于定点上方

                            for c in range(3):
                                flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                        (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                        mouse_x:mouse_x + mouse_width, c] +
                                        alpha * mouse_with_alpha[:, :, c]
                                )
                        if success_displayed:
                            flipped_image = img
                            pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_image)
                            draw.text((270, 205), "成功", fill=(0, 255, 0), font=font)
                            flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        if abs(target_b < goal_point[1]) and abs(target_a < goal_point[1]):
                            if not success_displayed:
                                if hint_time_started:
                                    hint_time = 0
                                    hint_time1 = 0
                                    hint_time2 = 0
                                    hint_time_started = False
                                move_count = move_count + 1
                                success_displayed = True
                                success_display_start_time = time.time()

                                # 读取带有透明通道的图像
                                another_success_image = cv2.imread('media/Successgirl.png', cv2.IMREAD_UNCHANGED)

                                # 缩放图像
                                re_another_success_image = cv2.resize(another_success_image, (100, 100))

                                # 获取图像的高度和宽度
                                another_success_height, another_success_width, _ = re_another_success_image.shape

                                # 创建一个带有透明通道的空白图像
                                success_with_alpha = np.zeros((another_success_height, another_success_width, 4),
                                                              dtype=np.uint8)

                                # 将RGB通道复制到带有透明通道的图像中
                                success_with_alpha[:, :, :3] = re_another_success_image[:, :, :3]

                                # 设置透明通道
                                alpha_threshold = 0  # 设置透明度阈值
                                alpha_channel = re_another_success_image[:, :, 0]  # 使用蓝色通道作为透明度
                                success_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)

                                alpha = success_with_alpha[:, :, 3] / 255.0
                                # 计算另一张成功跳起图片的位置
                                another_success_x = 305
                                another_success_y = target_point[1] - another_success_height + 10
                                # 绘制另一张成功跳起图片（包含透明背景）
                                for c in range(3):  # 这里是4通道，包括RGB和透明通道
                                    flipped_image[another_success_y:another_success_y + another_success_height,
                                    another_success_x:another_success_x + another_success_width, c] = (
                                            (1.0 - alpha) * flipped_image[
                                                            another_success_y:another_success_y + another_success_height,
                                                            another_success_x:another_success_x + another_success_width,
                                                            c] +
                                            alpha * success_with_alpha[:, :, c]
                                    )

                                success_animation_start_time = time.time()  # 记录动画开始的时间戳
                            else:
                                show_text = True

                if (Mode == "Easy"):
                    cv2.line(flipped_image, (0, target_point[1]), (640, target_point[1]), (0, 255, 0), 5)
                    distance = 20
                    pixel_cm = 2.4
                    point_y = target_point[1] - int(distance * pixel_cm)
                    cv2.line(flipped_image, (0, point_y), (640, point_y), (0, 0, 255), 5)
                    goal_point = (target_point[0], point_y)
                    target_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                    target_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                    pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    draw.text((25, 50), "成功次數 " + str(move_count) + "/" + str(end_count), fill=(0, 255, 0), font=font_time)
                    flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    img = flipped_image
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

                    if move_count == end_count:  # 之後傳送數據
                        pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        draw.text((150, 150), "結束", fill=(0, 255, 0), font=font)
                        t_time2 = time.time()
                        t_time = int(t_time2 - t_time1)
                        plot_data.put(False)
                        break
                    if time_started:
                        time_limit_minutes = 8  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 8 minutes. Exiting program.")
                            break

                    if mouse_move_easy:
                        if mtv:
                            pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_image)
                            draw.text(calculateTextPoint("跳過貓咪!", center_point, font_size), "跳過貓咪!",
                                      fill=(0, 255, 0), font=font)

                            mouse_image = cv2.imread('media/cat.png', cv2.IMREAD_COLOR)  # 以彩色模式读取老鼠图像
                            re_image = cv2.resize(mouse_image[:], (100, 100))
                            mouse_height, mouse_width, _ = re_image.shape
                            # 创建一个带有透明通道的空白图像
                            mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)
                            # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                            mouse_with_alpha[:, :, :3] = re_image
                            # 设置透明通道
                            alpha_threshold = 0  # 设置透明度阈值
                            # 使用蓝色通道作为透明度
                            alpha_channel = re_image[:, :, 0]  # 使用蓝色通道作为透明度
                            mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                            # 直接将带有透明通道的老鼠图像合成到背景图像上
                            alpha = mouse_with_alpha[:, :, 3] / 255.0

                            # 根据 "mouse_position" 计算绘制的位置
                            mouse_x = 310
                            mouse_y = 360  # 让老鼠位于定点上方

                            for c in range(3):
                                flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                        (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                        mouse_x:mouse_x + mouse_width, c] +
                                        alpha * mouse_with_alpha[:, :, c]
                                )

                        if success_displayed:
                            flipped_image = img
                            pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_image)
                            draw.text((270, 205), "成功", fill=(0, 255, 0), font=font)
                            flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                        if abs(target_b < goal_point[1]) and abs(target_a < goal_point[1]):
                            if not success_displayed:
                                if hint_time_started:
                                    hint_time = 0
                                    hint_time1 = 0
                                    hint_time2 = 0
                                    hint_time_started = False
                                move_count = move_count + 1
                                success_displayed = True
                                success_display_start_time = time.time()

                                # 读取带有透明通道的图像
                                another_success_image = cv2.imread('media/successgirl.png', cv2.IMREAD_UNCHANGED)

                                # 缩放图像
                                re_another_success_image = cv2.resize(another_success_image, (100, 100))

                                # 获取图像的高度和宽度
                                another_success_height, another_success_width, _ = re_another_success_image.shape

                                # 创建一个带有透明通道的空白图像
                                success_with_alpha = np.zeros((another_success_height, another_success_width, 4),
                                                              dtype=np.uint8)

                                # 将RGB通道复制到带有透明通道的图像中
                                success_with_alpha[:, :, :3] = re_another_success_image[:, :, :3]

                                # 设置透明通道
                                alpha_threshold = 0  # 设置透明度阈值
                                alpha_channel = re_another_success_image[:, :, 0]  # 使用蓝色通道作为透明度
                                success_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)

                                alpha = success_with_alpha[:, :, 3] / 255.0
                                # 计算另一张成功跳起图片的位置
                                another_success_x = 305
                                another_success_y = target_point[1] - another_success_height + 10
                                # 绘制另一张成功跳起图片（包含透明背景）
                                for c in range(3):  # 这里是4通道，包括RGB和透明通道
                                    flipped_image[another_success_y:another_success_y + another_success_height,
                                    another_success_x:another_success_x + another_success_width, c] = (
                                            (1.0 - alpha) * flipped_image[
                                                            another_success_y:another_success_y + another_success_height,
                                                            another_success_x:another_success_x + another_success_width,
                                                            c] +
                                            alpha * success_with_alpha[:, :, c]
                                    )

                                success_animation_start_time = time.time()  # 记录动画开始的时间戳


                            else:


                                show_text = True

                if (Mode == "Normal"):
                    cv2.line(flipped_image, (0, target_point[1]), (640, target_point[1]), (0, 255, 0), 5)
                    distance = 30
                    pixel_cm = 2.4
                    point_y = target_point[1] - int(distance * pixel_cm)
                    cv2.line(flipped_image, (0, point_y), (640, point_y), (0, 0, 255), 5)
                    goal_point = (target_point[0], point_y)
                    target_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                    target_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                    pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    draw.text((25, 50), "成功次數 " + str(move_count) + "/" + str(end_count), fill=(0, 255, 0), font=font_time)
                    flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    img = flipped_image

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

                    if move_count == end_count:  # 之後傳送數據
                        pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        draw.text((150, 150), "結束", fill=(0, 255, 0), font=font)
                        t_time2 = time.time()
                        t_time = int(t_time2 - t_time1)
                        plot_data.put(False)
                        break
                    if time_started:
                        time_limit_minutes = 10  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 10 minutes. Exiting program.")
                            break
                    if mouse_move_easy:
                        if mtv:
                            pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_image)
                            draw.text(calculateTextPoint("跳過長頸鹿!", center_point, font_size), "跳過長頸鹿!",
                                      fill=(0, 255, 0), font=font)
                            mouse_image = cv2.imread('media/234.png', cv2.IMREAD_COLOR)  # 以彩色模式读取老鼠图像
                            re_image = cv2.resize(mouse_image[:], (100, 100))
                            mouse_height, mouse_width, _ = re_image.shape
                            # 创建一个带有透明通道的空白图像
                            mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)
                            # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                            mouse_with_alpha[:, :, :3] = re_image
                            # 设置透明通道
                            alpha_threshold = 0  # 设置透明度阈值
                            # 使用蓝色通道作为透明度
                            alpha_channel = re_image[:, :, 0]  # 使用蓝色通道作为透明度
                            mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                            # 直接将带有透明通道的老鼠图像合成到背景图像上
                            alpha = mouse_with_alpha[:, :, 3] / 255.0

                            # 根据 "mouse_position" 计算绘制的位置
                            mouse_x = 290
                            mouse_y = 330  # 让老鼠位于定点上方

                            for c in range(3):
                                flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                        (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                        mouse_x:mouse_x + mouse_width, c] +
                                        alpha * mouse_with_alpha[:, :, c]
                                )

                        if success_displayed:
                            flipped_image = img
                            pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                            draw = ImageDraw.Draw(pil_image)
                            draw.text((270, 205), "成功", fill=(0, 255, 0), font=font)
                            flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                        if abs(target_b < goal_point[1]) and abs(target_a < goal_point[1]):
                            if not success_displayed:
                                if hint_time_started:
                                    hint_time = 0
                                    hint_time1 = 0
                                    hint_time2 = 0
                                    hint_time_started = False
                                move_count = move_count + 1
                                success_displayed = True
                                success_display_start_time = time.time()

                                # 读取带有透明通道的图像
                                another_success_image = cv2.imread('media/successgirl.png', cv2.IMREAD_UNCHANGED)

                                # 缩放图像
                                re_another_success_image = cv2.resize(another_success_image, (100, 100))

                                # 获取图像的高度和宽度
                                another_success_height, another_success_width, _ = re_another_success_image.shape

                                # 创建一个带有透明通道的空白图像
                                success_with_alpha = np.zeros((another_success_height, another_success_width, 4),
                                                              dtype=np.uint8)

                                # 将RGB通道复制到带有透明通道的图像中
                                success_with_alpha[:, :, :3] = re_another_success_image[:, :, :3]

                                # 设置透明通道
                                alpha_threshold = 0  # 设置透明度阈值
                                alpha_channel = re_another_success_image[:, :, 0]  # 使用蓝色通道作为透明度
                                success_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)

                                alpha = success_with_alpha[:, :, 3] / 255.0
                                # 计算另一张成功跳起图片的位置
                                another_success_x = 305
                                another_success_y = target_point[1] - another_success_height + 10
                                # 绘制另一张成功跳起图片（包含透明背景）
                                for c in range(3):  # 这里是4通道，包括RGB和透明通道
                                    flipped_image[another_success_y:another_success_y + another_success_height,
                                    another_success_x:another_success_x + another_success_width, c] = (
                                            (1.0 - alpha) * flipped_image[
                                                            another_success_y:another_success_y + another_success_height,
                                                            another_success_x:another_success_x + another_success_width,
                                                            c] +
                                            alpha * success_with_alpha[:, :, c]
                                    )

                                success_animation_start_time = time.time()  # 记录动画开始的时间戳


                            else:


                                show_text = True
                # 在 Successmouse 动画持续时间内绘制动画效果
                if success_displayed:
                    time_since_animation = time.time() - success_animation_start_time

                    if time_since_animation < success_display_duration:
                        mtv = False
                        # 计算动画效果，可以使用 sin 函数来模拟上下浮动效果
                        animation_amplitude = 10  # 上下浮动的幅度
                        animation_frequency = 5  # 浮动的频率
                        animation_offset = animation_amplitude * math.sin(animation_frequency * time_since_animation)

                        # 读取带有透明通道的图像
                        another_success_image = cv2.imread('media/Successgirl.png', cv2.IMREAD_UNCHANGED)

                        # 缩放图像
                        re_another_success_image = cv2.resize(another_success_image, (100, 100))

                        # 获取图像的高度和宽度
                        another_success_height, another_success_width, _ = re_another_success_image.shape

                        # 创建一个带有透明通道的空白图像
                        success_with_alpha = np.zeros((another_success_height, another_success_width, 4),
                                                      dtype=np.uint8)

                        # 将RGB通道复制到带有透明通道的图像中
                        success_with_alpha[:, :, :3] = re_another_success_image[:, :, :3]

                        # 设置透明通道
                        alpha_threshold = 0  # 设置透明度阈值
                        alpha_channel = re_another_success_image[:, :, 0]  # 使用蓝色通道作为透明度
                        success_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)

                        alpha = success_with_alpha[:, :, 3] / 255.0

                        # 计算成功跳起图片的位置（垂直方向上进行浮动）
                        success_image_y_with_animation = another_success_y + int(animation_offset)

                        for c in range(3):  # 这里是4通道，包括RGB和透明通道
                            flipped_image[
                            success_image_y_with_animation:success_image_y_with_animation + another_success_height,
                            another_success_x:another_success_x + another_success_width, c] = (
                                    (1.0 - alpha) * flipped_image[
                                                    success_image_y_with_animation:success_image_y_with_animation + another_success_height,
                                                    another_success_x:another_success_x + another_success_width,
                                                    c] +
                                    alpha * success_with_alpha[:, :, c]
                            )
                    else:
                        success_displayed = False  # 重置状态
                        mtv = True

                if (Mode == "Hard"):
                    cv2.line(flipped_image, (target_point[0], 480), (target_point[0], 0), (0, 255, 0), 5)
                    distance = 37
                    pixel_cm = 2.4
                    point_y = target_point[0] - int(distance * pixel_cm)
                    cv2.line(flipped_image, (point_y, 480), (point_y, 0), (0, 0, 255), 5)
                    pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    draw.text((25, 50), "成功次數 " + str(move_count) + "/" + str(end_count), fill=(0, 255, 0), font=font_time)
                    flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

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

                    if move_count == end_count:  # 之後傳送數據
                        pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        draw.text((150, 150), "結束", fill=(0, 255, 0), font=font)
                        t_time2 = time.time()
                        t_time = int(t_time2 - t_time1)
                        plot_data.put(False)
                        break
                    if time_started:
                        time_limit_minutes = 12  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 12 minutes. Exiting program.")
                            break
                    if not mouse_move:
                        mouse_image = readImage('media/jump_mouse.png')  # 以彩色模式读取老鼠图像
                        re_image = cv2.resize(mouse_image[:], (100, 100))

                        flipped_image = cv2.cvtColor(flipped_image, cv2.COLOR_RGB2RGBA)

                        mouse_height, mouse_width = re_image.shape[:2]
                        # 创建一个带有透明通道的空白图像
                        mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)
                        # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                        mouse_with_alpha = re_image.copy()
                        # 设置透明通道
                        alpha_threshold = 0  # 设置透明度阈值
                        # 使用蓝色通道作为透明度
                        alpha_channel = re_image[:, :, 0]  # 使用蓝色通道作为透明度
                        mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                        # 直接将带有透明通道的老鼠图像合成到背景图像上
                        alpha = mouse_with_alpha[:, :, 3] / 255.0
                        mouse_x = int(mouse_position[0]) - mouse_width
                        mouse_y = int(mouse_position[1]) - mouse_height
                        for c in range(3):
                            flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                    (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                    mouse_x:mouse_x + mouse_width, c] +
                                    alpha * mouse_with_alpha[:, :, c]
                            )
                    if not Jump_Right:
                        goal_point = (point_y, target_point[1])
                        target_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * flipped_image.shape[1])
                        foot_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                        target_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * flipped_image.shape[1])
                        foot_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                        pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        draw.text(calculateTextPoint("跟著老鼠向左跳", center_point, font_size), "跟著老鼠向左跳",
                                  fill=(0, 255, 0), font=font)
                        flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        if abs(target_a - target_point[0]) < 10 and abs(foot_a - target_point[1]) < 10 and x:
                            if not mouse_move:
                                fg = True
                                mouse_move = True
                            text_left = False
                            x = False
                        else:
                            if text_left:
                                cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                                pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                                draw = ImageDraw.Draw(pil_image)
                                draw.text(calculateTextPoint("請站在點上", center_point, font_size, vertical_offset=100), "請站在點上",
                                          fill=(0, 255, 0), font=font)
                                flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                        if mouse_move and fg:
                            distance = 60
                            pixel_cm = 2.4
                            mouse_image = readImage('media/mousejump.png')  # 以彩色模式读取老鼠图像
                            re_image = cv2.resize(mouse_image[:], (100, 100))
                            mouse_height, mouse_width, _ = re_image.shape
                            # 创建一个带有透明通道的空白图像
                            mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)
                            # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                            mouse_with_alpha = re_image
                            # 设置透明通道
                            alpha_threshold = 0  # 设置透明度阈值
                            # 使用蓝色通道作为透明度
                            alpha_channel = re_image[:, :, 0]  # 使用蓝色通道作为透明度
                            mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                            # 直接将带有透明通道的老鼠图像合成到背景图像上
                            alpha = mouse_with_alpha[:, :, 3] / 255.0
                            mouse_position = (
                                mouse_position[0] + (mouse_speed * mouse_direction1),
                                mouse_position[1] + (mouse_speed1 * mouse_direction2))
                            # 檢查是否越界並改變方向
                            mouse_x = int(mouse_position[0]) - mouse_width
                            mouse_y = int(mouse_position[1]) - mouse_height  # 讓老鼠位於定點上方
                            mouse_position_X = int(mouse_position[1])
                            if int(mouse_position[0]) <= target_point[0] - int(distance * pixel_cm):
                                fg = False
                                mouse_move = False
                                mouse_position = (point_y, 420)
                            if mouse_position_X <= target_point[1] - 20 or mouse_position_X >= target_point[1]:
                                mouse_direction2 *= -1
                                direction_change += 1
                                if direction_change == 2:
                                    fg = False
                                    mouse_move = False
                                    mouse_position = (point_y, 420)
                                    direction_change = 0

                            for c in range(3):
                                flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                        (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                        mouse_x:mouse_x + mouse_width, c] +
                                        alpha * mouse_with_alpha[:, :, c]
                                )

                        if abs(target_b < goal_point[0]) and abs(target_a < goal_point[0]):
                            # and foot_a < target_point[1] and foot_b < target_point[1]
                            if hint_time_started:
                                hint_time = 0
                                hint_time1 = 0
                                hint_time2 = 0
                                hint_time_started = False
                            move_count = move_count + 1
                            mouse_move = False
                            fg = False
                            text_right = True
                            Jump_Right = True
                            direction_change = 0
                            mouse_direction1 = 1
                            mouse_direction2 = -1
                            x = True
                            mouse_position = (point_y, 420)

                    if Jump_Right:
                        goal_point = target_point
                        target_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * flipped_image.shape[1])
                        foot_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                        target_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * flipped_image.shape[1])
                        foot_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                        pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        draw.text(calculateTextPoint("跟著老鼠向右跳", center_point, font_size), "跟著老鼠向右跳",
                                  fill=(0, 255, 0), font=font)
                        flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                        if abs(target_b - point_y) < 10 and abs(foot_b - target_point[1]) < 10 and x:
                            if not hint_time_started:
                                hint_time1 = time.time()
                                hint_time_started = True
                            if not mouse_move:
                                mouse_direction1 = 1
                                mouse_direction2 = -1
                                mouse_move = True
                            text_right = False
                            x = False
                        else:
                            if text_right:
                                cv2.circle(flipped_image, (point_y, target_point[1]), 8, (0, 255, 0), -1)
                                pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                                draw = ImageDraw.Draw(pil_image)
                                draw.text(calculateTextPoint("請站在點上", center_point, font_size, vertical_offset=100), "請站在點上",
                                          fill=(0, 255, 0), font=font)
                                flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        if mouse_move and not fg:
                            mouse_image = readImage('media/mousejump1.png')  # 以彩色模式读取老鼠图像
                            re_image = cv2.resize(mouse_image[:], (100, 100))

                            mouse_height, mouse_width, _ = re_image.shape
                            # 创建一个带有透明通道的空白图像
                            mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)
                            # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                            mouse_with_alpha = re_image
                            # 设置透明通道
                            alpha_threshold = 0  # 设置透明度阈值
                            # 使用蓝色通道作为透明度
                            alpha_channel = re_image[:, :, 0]  # 使用蓝色通道作为透明度
                            mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                            # 直接将带有透明通道的老鼠图像合成到背景图像上
                            alpha = mouse_with_alpha[:, :, 3] / 255.0
                            mouse_position = (mouse_position[0] + (mouse_speed * mouse_direction1),
                                              mouse_position[1] + (mouse_speed1 * mouse_direction2))
                            # 檢查是否越界並改變方向
                            mouse_x = int(mouse_position[0]) - mouse_width
                            mouse_y = int(mouse_position[1]) - mouse_height  # 讓老鼠位於定點上方
                            mouse_position_X = int(mouse_position[1])
                            if int(mouse_position[0]) >= target_point[0] + mouse_width:
                                mouse_move = False
                                mouse_position = (400 + mouse_width, 420)
                            if mouse_position_X <= target_point[1] - 20 or mouse_position_X >= target_point[1]:
                                mouse_direction2 *= -1
                                direction_change += 1
                                if direction_change == 2:
                                    mouse_move = False
                                    mouse_position = (400 + mouse_width, 420)
                                    direction_change = 0
                            for c in range(3):
                                flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                        (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                        mouse_x:mouse_x + mouse_width, c] +
                                        alpha * mouse_with_alpha[:, :, c]
                                )
                        if abs(target_b > goal_point[0]) and abs(target_a > goal_point[0]):
                            # and foot_a < target_point[1] and foot_b < target_point[1]
                            if hint_time_started:
                                hint_time = 0
                                hint_time1 = 0
                                hint_time2 = 0
                                hint_time_started = False
                            move_count = move_count + 1
                            mouse_move = False
                            Jump_Right = False
                            text_left = True
                            direction_change = 0
                            mouse_direction1 = -1
                            mouse_direction2 = -1
                            x = True
                            mouse_position = (400 + mouse_width, 420)

                if (Mode == "VeryHard"):
                    cv2.line(flipped_image, (target_point[0], 480), (target_point[0], 0), (0, 255, 0), 5)
                    distance = 60
                    pixel_cm = 2.4
                    point_y = target_point[0] - int(distance * pixel_cm)
                    cv2.line(flipped_image, (point_y, 480), (point_y, 0), (0, 0, 255), 5)
                    pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_image)
                    draw.text((25, 50), "成功次數 " + str(move_count) + "/" + str(end_count), fill=(0, 255, 0), font=font_time)
                    flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


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

                    if move_count == end_count:  # 之後傳送數據
                        pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        draw.text((150, 150), "結束", fill=(0, 255, 0), font=font)
                        t_time2 = time.time()
                        t_time = int(t_time2 - t_time1)
                        plot_data.put(False)
                        break
                    if time_started:
                        time_limit_minutes = 14  # 时间限制（分钟）
                        time_limit_seconds = time_limit_minutes * 60  # 转换为秒
                        time2 = time.time()
                        time_count = time2 - time1
                        if time_count > time_limit_seconds:
                            print("Training time exceeded 14 minutes. Exiting program.")
                            break
                    if not mouse_move:
                        mouse_image = readImage('media/jump_mouse.png')  # 以彩色模式读取老鼠图像
                        re_image = cv2.resize(mouse_image[:], (100, 100))

                        flipped_image = cv2.cvtColor(flipped_image, cv2.COLOR_RGB2RGBA)

                        mouse_height, mouse_width = re_image.shape[:2]
                        # 创建一个带有透明通道的空白图像
                        mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)
                        # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                        mouse_with_alpha = re_image.copy()
                        # 设置透明通道
                        alpha_threshold = 0  # 设置透明度阈值
                        # 使用蓝色通道作为透明度
                        alpha_channel = re_image[:, :, 0]  # 使用蓝色通道作为透明度
                        mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                        # 直接将带有透明通道的老鼠图像合成到背景图像上
                        alpha = mouse_with_alpha[:, :, 3] / 255.0
                        mouse_x = int(mouse_position[0]) - mouse_width
                        mouse_y = int(mouse_position[1]) - mouse_height
                        for c in range(3):
                            flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                    (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                    mouse_x:mouse_x + mouse_width, c] +
                                    alpha * mouse_with_alpha[:, :, c]
                            )
                    if not Jump_Right:
                        goal_point = (point_y, target_point[1])
                        target_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * flipped_image.shape[1])
                        foot_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                        target_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * flipped_image.shape[1])
                        foot_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                        pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        draw.text(calculateTextPoint("跟著老鼠向左跳", center_point, font_size), "跟著老鼠向左跳",
                                  fill=(0, 255, 0), font=font)
                        flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        if abs(target_a - target_point[0]) < 10 and abs(foot_a - target_point[1]) < 10 and x:

                            if not mouse_move:
                                fg = True
                                mouse_move = True
                            text_left = False
                            x = False
                        else:
                            if text_left:
                                cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                                pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                                draw = ImageDraw.Draw(pil_image)
                                draw.text(calculateTextPoint("請站在點上", center_point, font_size, vertical_offset=100), "請站在點上",
                                          fill=(0, 255, 0), font=font)
                                flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                        if mouse_move and fg:
                            distance = 60
                            pixel_cm = 2.4
                            mouse_image = readImage('media/mousejump.png')  # 以彩色模式读取老鼠图像
                            re_image = cv2.resize(mouse_image[:], (100, 100))
                            mouse_height, mouse_width, _ = re_image.shape
                            # 创建一个带有透明通道的空白图像
                            mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)
                            # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                            mouse_with_alpha = re_image
                            # 设置透明通道
                            alpha_threshold = 0  # 设置透明度阈值
                            # 使用蓝色通道作为透明度
                            alpha_channel = re_image[:, :, 0]  # 使用蓝色通道作为透明度
                            mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                            # 直接将带有透明通道的老鼠图像合成到背景图像上
                            alpha = mouse_with_alpha[:, :, 3] / 255.0
                            mouse_position = (
                                mouse_position[0] + (mouse_speed * mouse_direction1),
                                mouse_position[1] + (mouse_speed1 * mouse_direction2))
                            # 檢查是否越界並改變方向
                            mouse_x = int(mouse_position[0]) - mouse_width
                            mouse_y = int(mouse_position[1]) - mouse_height  # 讓老鼠位於定點上方
                            mouse_position_X = int(mouse_position[1])
                            if int(mouse_position[0]) <= target_point[0] - int(distance * pixel_cm):
                                fg = False
                                mouse_move = False
                                mouse_position = (265, 420)
                            if mouse_position_X <= target_point[1] - 20 or mouse_position_X >= target_point[1]:
                                mouse_direction2 *= -1
                                direction_change += 1
                                if direction_change == 2:
                                    fg = False
                                    mouse_move = False
                                    mouse_position = (265, 420)
                                    direction_change = 0

                            for c in range(3):
                                flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                        (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                        mouse_x:mouse_x + mouse_width, c] +
                                        alpha * mouse_with_alpha[:, :, c]
                                )

                        if abs(target_b < goal_point[0]) and abs(target_a < goal_point[0]):
                            # and foot_a < target_point[1] and foot_b < target_point[1]
                            if hint_time_started:
                                hint_time = 0
                                hint_time1 = 0
                                hint_time2 = 0
                                hint_time_started = False
                            move_count = move_count + 1
                            mouse_move = False
                            fg = False
                            text_right = True
                            Jump_Right = True
                            direction_change = 0
                            mouse_direction1 = 1
                            mouse_direction2 = -1
                            x = True
                            mouse_position = (265, 420)

                    if Jump_Right:
                        goal_point = target_point
                        target_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * flipped_image.shape[1])
                        foot_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                        target_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * flipped_image.shape[1])
                        foot_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                        pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                        draw = ImageDraw.Draw(pil_image)
                        draw.text(calculateTextPoint("跟著老鼠向右跳", center_point, font_size), "跟著老鼠向右跳",
                                  fill=(0, 255, 0), font=font)
                        flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                        if abs(target_b - point_y) < 10 and abs(foot_b - target_point[1]) < 10 and x:
                            if not hint_time_started:
                                hint_time1 = time.time()
                                hint_time_started = True
                            if not mouse_move:
                                mouse_direction1 = 1
                                mouse_direction2 = -1
                                mouse_move = True
                            text_right = False
                            x = False
                        else:
                            if text_right:
                                cv2.circle(flipped_image, (point_y, target_point[1]), 8, (0, 255, 0), -1)
                                pil_image = Image.fromarray(cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB))
                                draw = ImageDraw.Draw(pil_image)
                                draw.text(calculateTextPoint("請站在點上", center_point, font_size, vertical_offset=100), "請站在點上",
                                          fill=(0, 255, 0), font=font)
                                flipped_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                        if mouse_move and not fg:
                            mouse_image = readImage('media/mousejump1.png')  # 以彩色模式读取老鼠图像
                            re_image = cv2.resize(mouse_image[:], (100, 100))

                            mouse_height, mouse_width, _ = re_image.shape
                            # 创建一个带有透明通道的空白图像
                            mouse_with_alpha = np.zeros((mouse_height, mouse_width, 4), dtype=np.uint8)
                            # 将老鼠图像的颜色通道复制到新的带有透明通道的图像中
                            mouse_with_alpha = re_image
                            # 设置透明通道
                            alpha_threshold = 0  # 设置透明度阈值
                            # 使用蓝色通道作为透明度
                            alpha_channel = re_image[:, :, 0]  # 使用蓝色通道作为透明度
                            mouse_with_alpha[:, :, 3] = np.where(alpha_channel > alpha_threshold, 255, 0)
                            # 直接将带有透明通道的老鼠图像合成到背景图像上
                            alpha = mouse_with_alpha[:, :, 3] / 255.0
                            mouse_position = (mouse_position[0] + (mouse_speed * mouse_direction1),
                                              mouse_position[1] + (mouse_speed1 * mouse_direction2))
                            # 檢查是否越界並改變方向
                            mouse_x = int(mouse_position[0]) - mouse_width
                            mouse_y = int(mouse_position[1]) - mouse_height  # 讓老鼠位於定點上方
                            mouse_position_X = int(mouse_position[1])
                            if int(mouse_position[0]) >= target_point[0] + mouse_width:
                                mouse_move = False
                                mouse_position = (400 + mouse_width, 420)
                            if mouse_position_X <= target_point[1] - 20 or mouse_position_X >= target_point[1]:
                                mouse_direction2 *= -1
                                direction_change += 1
                                if direction_change == 2:
                                    mouse_move = False
                                    mouse_position = (400 + mouse_width, 420)
                                    direction_change = 0
                            for c in range(3):
                                flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width, c] = (
                                        (1.0 - alpha) * flipped_image[mouse_y:mouse_y + mouse_height,
                                                        mouse_x:mouse_x + mouse_width, c] +
                                        alpha * mouse_with_alpha[:, :, c]
                                )
                        if abs(target_b > goal_point[0]) and abs(target_a > goal_point[0]):
                            # and foot_a < target_point[1] and foot_b < target_point[1]
                            if hint_time_started:
                                hint_time = 0
                                hint_time1 = 0
                                hint_time2 = 0
                                hint_time_started = False
                            move_count = move_count + 1
                            mouse_move = False
                            Jump_Right = False
                            text_left = True
                            direction_change = 0
                            mouse_direction1 = -1
                            mouse_direction2 = -1
                            x = True
                            mouse_position = (400 + mouse_width, 420)

            # 顯示圖像
            plot_data.put(flipped_image)

        cap.release()
        cv2.destroyAllWindows()

        def import_data(user, move_count, t_time, difficulty, grade):
            training_data = TrainingJump(user=user, move_count=move_count, t_time=t_time, difficulty=difficulty,
                                         grade=grade)
            train_down = TreatmentContent.objects.filter(id=id, user=user).update(status=True)
            training_data.save()
            train_down.save()

        if Mode == "VeryEasy":
            if move_count == end_count and t_time <= 60:
                grade = 'A'
            elif move_count == end_count and t_time > 60 and t_time <= 120:
                grade = 'B'
            else:
                grade = 'C'
        elif Mode == "Easy":
            if move_count == end_count and t_time <= 100:
                grade = 'A'
            elif move_count == end_count and t_time > 100 and t_time <= 200:
                grade = 'B'
            else:
                grade = 'C'
        elif Mode == "Normal":
            if move_count == end_count and t_time <= 120:
                grade = 'A'
            elif move_count == end_count and t_time > 120 and t_time <= 180:
                grade = 'B'
            else:
                grade = 'C'
        elif Mode == "Hard":
            if move_count == end_count and t_time <= 140:
                grade = 'A'
            elif move_count == end_count and t_time > 140 and t_time <= 220:
                grade = 'B'
            else:
                grade = 'C'
        elif Mode == "VeryHard":
            if move_count == end_count and t_time <= 160:
                grade = 'A'
            elif move_count >= end_count and t_time > 160 and t_time <= 240:
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

def main(user, mode, end, id, background_name):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ChildPioneer.settings")
    django.setup()
    global plot_data
    plot_data = queue.LifoQueue()

    if (mode == 'Hard'):
        background_name = 'background_images/back_jump_Hard.jpg'
    elif (mode == 'VeryHard'):
        background_name = 'background_images/back_jump_VeryHard.jpg'

    try:  # server端
        background_path = 'media/' + background_name
        open(background_path, 'r+')
    except:  # 離線端
        background_path = '../media/' + background_name
        open(background_path, 'r+')
    thread = Thread(target=jump_Ther, args=(user, mode, end, id, background_path), daemon=True)
    thread.start()
    while plot_data.empty():
        pass
    while True:
        data = plot_data.get()
        if type(data) == type(False) or cv2.waitKey(1) == 27:
            cv2.destroyAllWindows()
            break
        cv2.imshow('jump', data)

if __name__ == '__main__':
    main('stest', 'Easy', 12, 0, 'background_images/remove.png')
