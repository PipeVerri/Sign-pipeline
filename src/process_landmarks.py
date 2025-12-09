import mediapipe as mp
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor
from shared.lm_processing import Landmarks, nn_parser
import numpy as np
import cv2
from utils.video import frame_reader

folders = ["../data/test"]

def process_video(path, out_path):
    FPS = 6

    cap = cv2.VideoCapture(path)
    lm = Landmarks(max_frames_interpolation=18) # 3s de limite entre clips
    with mp.solutions.holistic.Holistic(model_complexity=2, static_image_mode=False) as holistic:
        for frame in frame_reader(cap, fps=FPS):
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm.add(res.pose_landmarks, res.left_hand_landmarks, res.right_hand_landmarks)

    clips = [[]]
    timestamps = [0.0]
    first_frame = True
    for pose, left, right, jumped_frame in lm.get_landmarks():
        if jumped_frame is not None:
            if first_frame:
                timestamps[-1] = jumped_frame / FPS
            else:
                clips.append([])
                timestamps.append(jumped_frame / FPS)

        clips[-1].append(nn_parser(pose, left, right))
        first_frame = False


for folder in folders:
    videos = os.listdir(folder + "/videos")
