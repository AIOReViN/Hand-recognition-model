import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

# 初始化 MediaPipe hands 模組
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=1,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# 開啟攝像頭
cap = cv2.VideoCapture(0)

# 儲存食指軌跡的隊列，增加儲存點數以更好地判斷圓形
path_points = deque(maxlen=40)  # 增加到40個點
path_points2 = deque(maxlen=40)

# 用於判斷手勢的變數
is_drawing = False
gesture_detected = False
cooldown = 0
COOLDOWN_FRAMES = 100  # 增加冷卻時間

def calculate_circularity(points):
    if len(points) < 15:  # 降低最小需要的點數
        return float('inf')
    
    # 計算中心點
    center_x = sum(p[0] for p in points) / len(points)
    center_y = sum(p[1] for p in points) / len(points)
    
    # 計算平均半徑
    distances = [math.sqrt((p[0] - center_x)**2 + (p[1] - center_y)**2) for p in points]
    avg_radius = sum(distances) / len(distances)
    
    # 檢查起點和終點是否接近（圓形封閉性）
    start_end_distance = math.sqrt(
        (points[0][0] - points[-1][0])**2 + 
        (points[0][1] - points[-1][1])**2
    )
    
    # 計算路徑長度
    path_length = 0
    for i in range(1, len(points)):
        path_length += math.sqrt(
            (points[i][0] - points[i-1][0])**2 + 
            (points[i][1] - points[i-1][1])**2
        )
    
    # 計算圓形度量（考慮多個因素）
    circularity = 0
    
    # 1. 半徑變化的一致性
    radius_variance = sum((d - avg_radius)**2 for d in distances) / len(distances)
    circularity += radius_variance / avg_radius if avg_radius > 0 else float('inf')
    
    # 2. 起點終點距離（應該較小）
    circularity += start_end_distance / avg_radius if avg_radius > 0 else float('inf')
    
    # 3. 路徑長度與圓周的比例（應接近2π）
    expected_circumference = 2 * math.pi * avg_radius
    circularity += abs(path_length - expected_circumference) / expected_circumference
    
    return circularity / 3  # 取平均值

def check_horizontal_line(points):
    if len(points) < 10:
        return False
    
    # 檢查起點和終點的水平距離
    start_x = points[0][0]
    end_x = points[-1][0]
    distance_x = abs(start_x - end_x)
    
    # 檢查垂直偏移
    max_y = max(p[1] for p in points)
    min_y = min(p[1] for p in points)
    vertical_variation = max_y - min_y
    
    return distance_x > 200 and vertical_variation < 100

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    
    if cooldown > 0:
        cooldown -= 1
        gesture_detected = True
    else:
        gesture_detected = False
    
    # 在畫面上顯示提示文字
    cv2.putText(frame, "Spell 1: Draw circle with one finger", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Spell 2: Draw horizontal line", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Spell 3: Draw circles with both hands", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if results.multi_hand_landmarks:
        if not is_drawing:
            path_points.clear()
            path_points2.clear()
            is_drawing = True
        
        num_hands = len(results.multi_hand_landmarks)
        
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 獲取食指指尖座標
            finger_tip = hand_landmarks.landmark[8]
            x = int(finger_tip.x * frame.shape[1])
            y = int(finger_tip.y * frame.shape[0])
            
            # 在食指位置畫一個明顯的圓點
            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
            
            if hand_idx == 0:
                path_points.append((x, y))
            elif hand_idx == 1:
                path_points2.append((x, y))
            
            # 繪製軌跡
            points = list(path_points)
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (0, 255, 0), 2)
            
            if hand_idx == 1:
                points2 = list(path_points2)
                if len(points2) > 1:
                    for i in range(1, len(points2)):
                        cv2.line(frame, points2[i-1], points2[i], (255, 0, 0), 2)
        
        # 手勢辨識（放寬條件）
        if not gesture_detected and len(path_points) > 15:
            # 檢查單手畫圓
            circularity = calculate_circularity(list(path_points))
            if circularity < 1.0 and num_hands == 1:  # 放寬閾值
                print("咒語一成功")
                cv2.putText(frame, "Spell 1 Success!", (200, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cooldown = COOLDOWN_FRAMES
            
            # 檢查水平線
            if check_horizontal_line(list(path_points)) and num_hands == 1:
                print("咒語二成功")
                cv2.putText(frame, "Spell 2 Success!", (200, 200), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cooldown = COOLDOWN_FRAMES
            
            # 檢查雙手畫圓
            if num_hands == 2 and len(path_points2) > 15:
                circularity1 = calculate_circularity(list(path_points))
                circularity2 = calculate_circularity(list(path_points2))
                if circularity1 < 1.2 and circularity2 < 1.2:  # 更寬鬆的閾值
                    print("咒語三成功")
                    cv2.putText(frame, "Spell 3 Success!", (200, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                    cooldown = COOLDOWN_FRAMES
    else:
        is_drawing = False
    
    cv2.imshow("Gesture Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break