import numpy as np

def mp_to_arr(landmarks):
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks])