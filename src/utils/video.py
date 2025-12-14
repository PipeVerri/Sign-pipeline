import cv2

def read_cap_segments(cap, fps=6, start=0, end=None):
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    skip_rate = int(round(fps_original / fps))

    start =  int(start * fps_original)
    end = end * fps_original if end else None
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_count = start

    while cap.isOpened():
        if end and frame_count >= end:
            break

        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_rate == 0:
            yield frame, (frame_count / fps_original)

        frame_count += 1