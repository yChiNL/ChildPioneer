import cv2
import mediapipe as mp
import time
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
mouse_time1 = 0
mouse_time2 = 0
mouse_time = False
mouse_move = False
run_OK = True
# 定點座標
target_point = (320, 400)

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
                    mouse_image = cv2.imread('../media/Dance_DoDo.jpg', )
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

        if mouse_move and Mode == "Easy" and Mode == "Normal":
            mouse_image = cv2.imread('../media/Dance_DoDo.jpg')
            re_image = cv2.resize(mouse_image, (40, 40))
            mouse_height, mouse_width, _ = re_image.shape

            mouse_position = (mouse_position[0] + (mouse_speed * mouse_direction1), mouse_position[1])
            # 檢查是否越界並改變方向
            mouse_x = int(mouse_position[0]) - mouse_width
            mouse_y = target_point[1] - mouse_height  # 讓老鼠位於定點上方

            if mouse_position[0] == target_point[0]-100 or mouse_position[0] == target_point[0]+100:
                mouse_direction1 *= -1
            # 繪製老鼠圖片
            flipped_image[mouse_y:mouse_y + mouse_height, mouse_x:mouse_x + mouse_width] = re_image

        if time_started:
            time2 = time.time() - time1
            t1 = countdown - time2
            cv2.putText(flipped_image, str(t1), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if (t1 <= 0):
                break

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

            if (Mode == "Easy"):
                cv2.line(flipped_image, (0, target_point[1]), (640, target_point[1]), (0, 255, 0), 5)
                distance = 20
                pixel_cm = 2.4
                point_y = target_point[1] - int(distance * pixel_cm)
                cv2.line(flipped_image, (0, point_y), (640, point_y), (0, 0, 255), 5)
                goal_point = (target_point[0], point_y)
                target_a = int(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * flipped_image.shape[0])
                target_b = int(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * flipped_image.shape[0])

                if abs(target_b < goal_point[1]) and abs(target_a < goal_point[1]):
                    cv2.putText(flipped_image, "Success", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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
                if abs(target_b < goal_point[1]) and abs(target_a < goal_point[1]):
                    cv2.putText(flipped_image, "Success", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    train_started = False
                    show_text = True

            if (Mode == "Hard"):
                cv2.line(flipped_image, (target_point[0], 480), (target_point[0], 0), (0, 255, 0), 5)
                distance = 60
                pixel_cm = 2.4
                point_y = target_point[0] - int(distance * pixel_cm)
                cv2.line(flipped_image, (point_y, 480), (point_y, 0), (0, 0, 255), 5)
                if not Jump_Right:
                    if not mouse_move:
                        mouse_image = cv2.imread('../media/Dance_DoDo.jpg', )
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
                        mouse_image = cv2.imread('../media/Dance_DoDo.jpg', )
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
                            mouse_image = cv2.imread('../media/Dance_DoDo.jpg', )
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