# Calcula el ratio entre los vectores codo-muñeca y muñeca-nudillo para escalar la mano
import mediapipe as mp
import cv2
import numpy as np
from src.utils.mediapipe.parse import mp_to_arr

img_path = "person.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with mp.solutions.holistic.Holistic(static_image_mode=True, model_complexity=2) as holistic:
    results = holistic.process(img_rgb)
    if results.left_hand_landmarks is not None:
        handedness = "left"
        hand_landmarks = results.left_hand_landmarks
    else:
        handedness = "right"
        hand_landmarks = results.right_hand_landmarks
    if results.pose_landmarks is None or hand_landmarks is None:
        raise Exception("No poe or hand detected")

    pose_array = mp_to_arr(results.pose_landmarks.landmark)
    hand_array = mp_to_arr(hand_landmarks.landmark)

    if handedness == "left":
        forearm_vec = pose_array[15] - pose_array[13]
    else:
        forearm_vec = pose_array[16] - pose_array[14]

    hand_vec = hand_array[9] - hand_array[0]

    ratio = np.linalg.norm(forearm_vec) / np.linalg.norm(hand_vec)
    print(ratio)