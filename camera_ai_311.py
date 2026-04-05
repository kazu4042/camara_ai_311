import cv2
import mediapipe as mp
import time
import numpy as np

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode



options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    black_frame = np.zeros_like(frame)
    
    if not ret:
        break

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    timestamp = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp)
    
    

    if result.hand_landmarks:
        
        print("検出！")
        for hand in result.hand_landmarks:
            points = []
            
            # 点を描く
            for point in hand:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                points.append((x, y))
                cv2.circle(black_frame, (x, y), 5, (0, 255, 0), -1)

            # 線を描く
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                cv2.line(black_frame, points[start_idx], points[end_idx], (255, 0, 0), 2)

    cv2.imshow("Hand Tracking", black_frame)

    key = cv2.waitKey(1)
    if key == 27 or cv2.getWindowProperty("Hand Tracking", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()