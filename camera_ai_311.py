import cv2
import mediapipe as mp
import time

# ===== 初期化 =====
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

# ===== カメラ =====
cap = cv2.VideoCapture(0)

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # mediapipe用に変換
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame
    )

    result = landmarker.detect_for_video(mp_image, frame_id)

    # 検出結果
    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            for point in hand:
                x = int(point.x * frame.shape[1])
                y = int(point.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()