import cv2
import time

def frame_reader(cap, fps=24):
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    skip_rate = int(round(fps_original/fps))
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_rate == 0:
            yield frame

        frame_count += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset the cap

def camera_reader(fps=24):
    fps_time = 1.0 / fps
    cap = cv2.VideoCapture(0)
    last_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        elapsed = now - last_time
        if elapsed < fps_time:
            continue

        last_time = now
        yield frame