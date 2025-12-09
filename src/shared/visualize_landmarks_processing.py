import mediapipe as mp
from src.lm_processing import Landmarks
from src.utils.video import camera_reader
from src.utils.mediapipe.render import draw_landmarks_from_array
import cv2

lm = Landmarks()

with mp.solutions.holistic.Holistic(model_complexity=2, static_image_mode=False) as holistic:
    lm_generator = lm.get_landmarks(continuous=True)
    for frame in camera_reader(fps=6):
        hol = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        lm.add(hol.pose_landmarks, hol.left_hand_landmarks, hol.right_hand_landmarks)
        pose, left, right = next(lm_generator)
        #frame = np.zeros(frame.shape)
        img = draw_landmarks_from_array(frame, pose, connections=mp.solutions.pose.POSE_CONNECTIONS)
        img = draw_landmarks_from_array(img, left, connections=mp.solutions.hands.HAND_CONNECTIONS)
        img = draw_landmarks_from_array(img, right, connections=mp.solutions.hands.HAND_CONNECTIONS)
        cv2.imshow("frame", img)
        cv2.waitKey(1)


