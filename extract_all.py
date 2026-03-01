import cv2
import mediapipe as mp
import numpy as np
import os

DATA_PATH = "data/videos"
SAVE_PATH = "data/landmarks"

# إعداد MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,   # مهم جداً: IMAGE mode
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# المرور على الكلمات
for word in os.listdir(DATA_PATH):

    word_folder = os.path.join(DATA_PATH, word)
    save_word_folder = os.path.join(SAVE_PATH, word)
    os.makedirs(save_word_folder, exist_ok=True)

    # المرور على الفيديوهات
    for video in os.listdir(word_folder):

        video_path = os.path.join(word_folder, video)

        cap = cv2.VideoCapture(video_path)
        sequence = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            results = landmarker.detect(mp_image)

            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    frame_landmarks = []
                    for lm in hand_landmarks:
                        frame_landmarks.extend([lm.x, lm.y, lm.z])
                    sequence.append(frame_landmarks)

        cap.release()

        sequence = np.array(sequence)

        filename = os.path.splitext(video)[0]
        np.save(os.path.join(save_word_folder, filename), sequence)

print("✅ All landmarks extracted successfully!")
