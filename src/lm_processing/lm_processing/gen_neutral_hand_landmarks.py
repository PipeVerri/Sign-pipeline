# Genera los landmarks para la mano neutral. Van a ser usados
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands

img_path = "neutral_hand.png"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with mp_hands.Hands(static_image_mode=True, model_complexity=1) as hands:
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        raise Exception("No hand detected")

    landmarks = []
    hand_landmarks = results.multi_hand_landmarks[0]
    for i, lm in enumerate(hand_landmarks.landmark):
        landmarks.append([lm.x, lm.y, lm.z])

    # Convertir a DataFrame
    landmarks_arr = np.array(landmarks)
    np.save("neutral_hand.npy", landmarks_arr)