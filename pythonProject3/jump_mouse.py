import cv2
import mediapipe as mp
import time
import math
# 初始化 MediaPipe 的姿態估計模型
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 初始化變數
train_started = False
show_text = True
time_started = False
text_left = True
text_right = True
Jump_Right = False
Mode = "Hard"
countdown = 30
t1 = None
time1 = 99999999999
time2 = 0
count = 0
t_time1 = 0
t_time2 = 0
t_time = 0
t_timestarted = False
mouse_move_easy = False
move_count = 0
# 老鼠的移動速度
mouse_speed_easy = 5
# 老鼠的移動方向，1 表示向右，-1 表示向左
mouse_direction_easy = 1
# ...
success_displayed = False
success_display_start_time = 0
success_display_duration = 3  # 切换图片显示时间，单位：秒

run_OK = 0
# 老鼠的起始位置 (x, y)
mouse_position = (320, 480)
# 老鼠的移動速度
mouse_speed = 2
mouse_speed1 = 1.2
# 老鼠的移動方向，1 表示向右，-1 表示向左
mouse_direction1 = 1
# 老鼠的移動方向，1 表示向下，-1 表示向上
mouse_direction2 = -1
direction_change = 0
mouse_move = False
# 定點座標
target_point = (320, 400)

if Mode == "Hard":
    mouse_speed = 3
    target_point = (400, 420)
    mouse_position = (400, 420)
# 初始化視訊處理
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # 將圖像轉換成 RGB 格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb_flip = cv2.flip(image_rgb, 1)

        # 進行姿態估計
        results = pose.process(image_rgb_flip)

        # 繪製姿態估計結果和定點
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        flipped_image = cv2.flip(image, 1)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(flipped_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 檢測使用者是否站在定點上
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            target_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * flipped_image.shape[1])
            target_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])



            if abs(target_x - target_point[0]) < 10 and abs(target_y - target_point[1]) < 10:
                show_text = False
                mouse_move_easy = True
                if Mode == 'Hard':
                    if run_OK == 0:
                        mouse_move = True
                if not time_started:
                    time1 = time.time()
                    time_started = True
                if not train_started:
                    train_started = True
            else:
                if show_text:
                    cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                    cv2.putText(flipped_image, "Please stand on point", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

            if (Mode == "Hard"):
                distance = 60
                pixel_cm = 2.4
                if mouse_move:
                    mouse_image = cv2.imread('media/Dance_DoDo.jpg',)
                    re_image = cv2.resize(mouse_image, (40, 40))
                    mouse_height, mouse_width, _ = re_image.shape
                    mouse_position = (mouse_position[0] - (mouse_speed * mouse_direction1), mouse_position[1] + (mouse_speed1 * mouse_direction2))
                    # 檢查是否越界並改變方向
                    mouse_x = int(mouse_position[0]) - mouse_width
                    mouse_y = int(mouse_position[1]) - mouse_height  # 讓老鼠位於定點上方
                    mouse_position_X = int(mouse_position[1])
                    if int(mouse_position[0]) >= target_point[0] or int(mouse_position[0]) <= target_point[0]-int(distance * pixel_cm) + mouse_width:
                        mouse_direction1 *= -1
                        if mouse_direction1 == -1:
                            mouse_position = (296, 420)
                            mouse_move = False
                        if mouse_direction1 == 1:
                            mouse_position = (400, 420)
                            mouse_move = False
                    if mouse_position_X <= target_point[1]-20 or mouse_position_X >= target_point[1]:
                        mouse_direction2 *= -1
                        direction_change += 1
                        if direction_change == 2:
                            mouse_position = (296, 420)
                            mouse_move = False
                        if direction_change == 4:
                            mouse_position = (400, 420)
                            mouse_move = False
                            direction_change = 0
                    # 繪製老鼠圖片
                    flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width] = re_image


        # 顯示計時器
        if train_started:
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

            if (Mode == "VeryEasy"):

                cv2.line(flipped_image, (0, target_point[1]), (640, target_point[1]), (0, 255, 0), 5)
                distance = 20
                pixel_cm = 2.4
                point_y = target_point[1] - int(distance * pixel_cm)
                cv2.line(flipped_image, (0, point_y), (640, point_y), (0, 0, 255), 5)
                goal_point = (target_point[0], point_y)
                target_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                target_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                if move_count == 3:  # 之後傳送數據
                    cv2.putText(flipped_image, "done", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    t_time2 = time.time()
                    t_time = t_time + (t_time2 - t_time1)
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
                    mouse_image = cv2.imread('../media/mouse.jpg')
                    re_image = cv2.resize(mouse_image, (50, 50))
                    mouse_height, mouse_width, _ = re_image.shape

                    # 根据 "mouse_position" 计算绘制的位置
                    mouse_x = 310
                    mouse_y = target_point[1] - mouse_height  # 让老鼠位于定点上方

                    # 绘制老鼠图片
                    flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width] = re_image

                    if abs(target_b < goal_point[1]) and abs(target_a < goal_point[1]):
                        if not success_displayed:
                            move_count = move_count + 1
                            success_displayed = True
                            success_display_start_time = time.time()

                            # 切换图片为另一张图片
                            another_success_image = cv2.imread('../media/Successmouse.jpg')  # 替换为实际路径
                            re_another_success_image = cv2.resize(another_success_image, (70, 80))
                            another_success_height, another_success_width, _ = re_another_success_image.shape

                            # 计算另一张成功跳起图片的位置
                            another_success_x = 305
                            another_success_y = target_point[1] - another_success_height + 10

                            # 绘制另一张成功跳起图片
                            flipped_image[another_success_y:another_success_y + another_success_height,
                            another_success_x:another_success_x + another_success_width] = re_another_success_image

                            success_animation_start_time = time.time()  # 记录动画开始的时间戳

                        if success_displayed:
                            time_since_display = time.time() - success_display_start_time

                            if time_since_display < success_display_duration:
                                # 计算动画的水平位移，可以使用 sin 函数来模拟左右摆动效果
                                animation_amplitude = 10  # 左右摆动的幅度
                                animation_frequency = 5  # 摆动的频率
                                animation_offset = animation_amplitude * math.sin(
                                    animation_frequency * time_since_display)

                                # 计算另一张图片的位置（水平方向上进行摆动）
                                another_success_x_with_animation = another_success_x + int(animation_offset)

                                # 绘制另一张成功跳起图片（包含摆动效果）
                                flipped_image[another_success_y:another_success_y + another_success_height,
                                another_success_x_with_animation:another_success_x_with_animation + another_success_width] = re_another_success_image

                            else:
                                success_display_ongoing = False
                                success_displayed = False  # 重置状态
                                train_started = False
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
                if move_count == 5:  # 之後傳送數據
                    cv2.putText(flipped_image, "done", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    t_time2 = time.time()
                    t_time = t_time + (t_time2 - t_time1)
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
                    mouse_image = cv2.imread('../media/mouse.jpg')
                    re_image = cv2.resize(mouse_image, (50, 50))
                    mouse_height, mouse_width, _ = re_image.shape

                    # 根据 "mouse_position" 计算绘制的位置
                    mouse_x = 310
                    mouse_y = target_point[1] - mouse_height  # 让老鼠位于定点上方

                    # 绘制老鼠图片
                    flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width] = re_image

                    if abs(target_b < goal_point[1]) and abs(target_a < goal_point[1]):
                        if not success_displayed:
                            move_count = move_count + 1
                            success_displayed = True
                            success_display_start_time = time.time()

                            # 切换图片为另一张图片
                            another_success_image = cv2.imread('../media/Successmouse.jpg')  # 替换为实际路径
                            re_another_success_image = cv2.resize(another_success_image, (70, 80))
                            another_success_height, another_success_width, _ = re_another_success_image.shape

                            # 计算另一张成功跳起图片的位置
                            another_success_x = 305
                            another_success_y = target_point[1] - another_success_height + 10

                            # 绘制另一张成功跳起图片
                            flipped_image[another_success_y:another_success_y + another_success_height,
                            another_success_x:another_success_x + another_success_width] = re_another_success_image

                            success_animation_start_time = time.time()  # 记录动画开始的时间戳

                        if success_displayed:
                            time_since_display = time.time() - success_display_start_time

                            if time_since_display < success_display_duration:
                                # 计算动画的水平位移，可以使用 sin 函数来模拟左右摆动效果
                                animation_amplitude = 10  # 左右摆动的幅度
                                animation_frequency = 5  # 摆动的频率
                                animation_offset = animation_amplitude * math.sin(
                                    animation_frequency * time_since_display)

                                # 计算另一张图片的位置（水平方向上进行摆动）
                                another_success_x_with_animation = another_success_x + int(animation_offset)

                                # 绘制另一张成功跳起图片（包含摆动效果）
                                flipped_image[another_success_y:another_success_y + another_success_height,
                                another_success_x_with_animation:another_success_x_with_animation + another_success_width] = re_another_success_image

                            else:
                                success_display_ongoing = False
                                success_displayed = False  # 重置状态
                                train_started = False
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
                if move_count == 8:  # 之後傳送數據
                    cv2.putText(flipped_image, "done", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    t_time2 = time.time()
                    t_time = t_time + (t_time2 - t_time1)
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
                    mouse_image = cv2.imread('../media/mouse.jpg')
                    re_image = cv2.resize(mouse_image, (60, 70))
                    mouse_height, mouse_width, _ = re_image.shape

                    # 根据 "mouse_position" 计算绘制的位置
                    mouse_x = 310
                    mouse_y = target_point[1] - mouse_height  # 让老鼠位于定点上方

                    # 绘制老鼠图片
                    flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width] = re_image
                    if abs(target_b < goal_point[1]) and abs(target_a < goal_point[1]):
                        cv2.putText(flipped_image, "Success", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        if not success_displayed:
                            move_count = move_count + 1
                            success_displayed = True
                            success_display_start_time = time.time()

                            # 切换图片为另一张图片
                            another_success_image = cv2.imread('../media/Successmouse.jpg')  # 替换为实际路径
                            re_another_success_image = cv2.resize(another_success_image, (80, 90))
                            another_success_height, another_success_width, _ = re_another_success_image.shape

                            # 计算另一张成功跳起图片的位置
                            another_success_x = 305
                            another_success_y = target_point[1] - another_success_height + 10

                            # 绘制另一张成功跳起图片
                            flipped_image[another_success_y:another_success_y + another_success_height,
                            another_success_x:another_success_x + another_success_width] = re_another_success_image

                            success_animation_start_time = time.time()  # 记录动画开始的时间戳

                        if success_displayed:
                            time_since_display = time.time() - success_display_start_time

                            if time_since_display < success_display_duration:
                                # 计算动画的水平位移，可以使用 sin 函数来模拟左右摆动效果
                                animation_amplitude = 10  # 左右摆动的幅度
                                animation_frequency = 5  # 摆动的频率
                                animation_offset = animation_amplitude * math.sin(
                                    animation_frequency * time_since_display)

                                # 计算另一张图片的位置（水平方向上进行摆动）
                                another_success_x_with_animation = another_success_x + int(animation_offset)

                                # 绘制另一张成功跳起图片（包含摆动效果）
                                flipped_image[another_success_y:another_success_y + another_success_height,
                                another_success_x_with_animation:another_success_x_with_animation + another_success_width] = re_another_success_image

                            else:
                                success_display_ongoing = False
                                success_displayed = False  # 重置状态
                                train_started = False
                                show_text = True
            # 在 Successmouse 动画持续时间内绘制动画效果
            if success_displayed:
                time_since_animation = time.time() - success_animation_start_time

                if time_since_animation < success_display_duration:
                    # 计算动画效果，可以使用 sin 函数来模拟上下浮动效果
                    animation_amplitude = 10  # 上下浮动的幅度
                    animation_frequency = 5  # 浮动的频率
                    animation_offset = animation_amplitude * math.sin(animation_frequency * time_since_animation)

                    # 计算成功跳起图片的位置（垂直方向上进行浮动）
                    success_image_y_with_animation = another_success_y + int(animation_offset)

                    # 绘制成功跳起图片（包含浮动效果）
                    flipped_image[
                    success_image_y_with_animation:success_image_y_with_animation + another_success_height,
                    another_success_x:another_success_x + another_success_width] = re_another_success_image
                else:
                    success_displayed = False  # 重置状态

            if (Mode == "Hard"):
                cv2.line(flipped_image, (target_point[0], 480), (target_point[0], 0), (0, 255, 0), 5)
                distance = 60
                pixel_cm = 2.4
                point_y = target_point[0] - int(distance * pixel_cm)
                cv2.line(flipped_image, (point_y, 480), (point_y, 0), (0, 0, 255), 5)
                if not Jump_Right:
                    if not mouse_move:
                        mouse_image = cv2.imread('media/Dance_DoDo.jpg', )
                        re_image = cv2.resize(mouse_image, (40, 40))
                        mouse_height, mouse_width, _ = re_image.shape
                        mouse_x = int(mouse_position[0]) - mouse_width
                        mouse_y = int(mouse_position[1]) - mouse_height
                        flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width] = re_image
                    goal_point = (point_y, target_point[1])
                    target_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * flipped_image.shape[1])
                    target_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * flipped_image.shape[1])
                    foot_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                    foot_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                    if abs(target_a - target_point[0]) < 10 and abs(foot_a - target_point[1]) < 10:
                        text_left = False
                    else:
                        if text_left:
                            cv2.circle(flipped_image, target_point, 8, (0, 255, 0), -1)
                            cv2.putText(flipped_image, "Please stand on point", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0), 2)
                    if abs(target_b < goal_point[0]) and abs(target_a < goal_point[0]):
                        #and foot_a < target_point[1] and foot_b < target_point[1]
                        count = count + 1
                        Jump_Right = True
                        text_right = True
                if Jump_Right:
                    run_OK = True
                    if not mouse_move:
                        mouse_image = cv2.imread('media/Dance_DoDo.jpg', )
                        re_image = cv2.resize(mouse_image, (40, 40))
                        mouse_height, mouse_width, _ = re_image.shape
                        mouse_x = int(mouse_position[0]) - mouse_width
                        mouse_y = int(mouse_position[1]) - mouse_height
                        flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width] = re_image
                    goal_point = target_point
                    target_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * flipped_image.shape[1])
                    target_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * flipped_image.shape[1])
                    foot_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                    foot_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])
                    if abs(target_b - point_y) < 10 and abs(foot_a - target_point[1]) < 10:
                        text_right = False
                        if run_OK:
                            mouse_move = True
                            run_OK = False
                        distance = 60
                        pixel_cm = 2.4
                        if mouse_move:
                            mouse_image = cv2.imread('media/Dance_DoDo.jpg', )
                            re_image = cv2.resize(mouse_image, (40, 40))
                            mouse_height, mouse_width, _ = re_image.shape
                            mouse_position = (mouse_position[0] - (mouse_speed * mouse_direction1),
                                              mouse_position[1] + (mouse_speed1 * mouse_direction2))
                            # 檢查是否越界並改變方向
                            mouse_x = int(mouse_position[0]) - mouse_width
                            mouse_y = int(mouse_position[1]) - mouse_height  # 讓老鼠位於定點上方
                            mouse_position_X = int(mouse_position[1])
                            if int(mouse_position[0]) >= target_point[0] or int(mouse_position[0]) <= target_point[0] - int(distance * pixel_cm) + mouse_width:
                                mouse_direction1 *= -1
                                if mouse_direction1 == -1:
                                    mouse_position = (296, 420)
                                    mouse_move = False
                                if mouse_direction1 == 1:
                                    mouse_position = (400, 420)
                                    mouse_move = False
                            if mouse_position_X <= target_point[1] - 20 or mouse_position_X >= target_point[1]:
                                mouse_direction2 *= -1
                                direction_change += 1
                                if direction_change == 2:
                                    mouse_position = (296, 420)
                                    mouse_move = False
                                if direction_change == 4:
                                    mouse_position = (400, 420)
                                    mouse_move = False
                                    direction_change = 0
                            # 繪製老鼠圖片
                            flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width] = re_image
                    else:
                        if text_right:
                            cv2.circle(flipped_image, (point_y, target_point[1]), 8, (0, 255, 0), -1)
                            cv2.putText(flipped_image, "Please stand on point", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 255, 0), 2)
                    if abs(target_b > goal_point[0]) and abs(target_a > goal_point[0]):
                        #and foot_a < target_point[1] and foot_b < target_point[1]
                        count = count + 1
                        Jump_Right = False
                        mouse_move = True
                        text_left = True

                cv2.putText(flipped_image, str(count), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 顯示圖像
        cv2.imshow("MediaPipe Pose", flipped_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()