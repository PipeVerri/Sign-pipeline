import mediapipe as mp
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor
from shared.lm_processing import Landmarks, nn_parser
import numpy as np

folders = ["../data/test"]

def process_video(path, out_path):


for folder in folders:
    videos = os.listdir(folder + "/videos")
