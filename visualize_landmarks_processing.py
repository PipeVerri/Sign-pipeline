from lm_processing import Landmarks
from utils.video import camera_reader
from utils.mediapipe.render import draw_landmarks_from_array
import cv2
import numpy as np
import mediapipe as mp
import threading
import time

stop_event = threading.Event()
lm = Landmarks(max_frames_interpolation=200)
lm_generator = None
generator_lock = threading.Lock()


def processing_thread():
    """Thread 1: Read frames and add landmarks to lm"""
    global lm_generator

    with mp.solutions.holistic.Holistic(model_complexity=2, static_image_mode=False) as holistic:
        # Initialize generator in this thread
        with generator_lock:
            lm_generator = lm.get_landmarks(continuous=True)

        for frame in camera_reader(fps=6):
            if stop_event.is_set():
                break

            # Process frame with MediaPipe
            hol = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Add landmarks to the processor
            lm.add(hol.pose_landmarks, hol.left_hand_landmarks, hol.right_hand_landmarks)

            # Only log transitions (pose present -> not present and vice versa)
            if len(lm.pose) > 1:
                prev_present = lm.pose[-2] is not None
                curr_present = hol.pose_landmarks is not None
    stop_event.set()


def rendering_thread():
    """Thread 2: Get landmarks from generator and render"""
    global lm_generator

    # Wait for generator to be initialized
    while lm_generator is None and not stop_event.is_set():
        time.sleep(0.01)

    frame_count = 0
    last_frame_shape = (480, 640, 3)

    while not stop_event.is_set():
        try:
            # Check if processing thread has added any frames
            if len(lm.pose) == 0:
                time.sleep(0.01)
                continue

            # Get next landmarks from generator with timeout protection
            with generator_lock:
                pose, left, right, jumped = next(lm_generator)
            frame_count += 1

            # Create blank frame and draw landmarks
            frame = np.zeros(last_frame_shape, dtype=np.uint8)
            if jumped:
                frame[:] = (255, 0, 0)
            img = draw_landmarks_from_array(frame, pose, connections=mp.solutions.pose.POSE_CONNECTIONS)
            img = draw_landmarks_from_array(img, left, connections=mp.solutions.hands.HAND_CONNECTIONS)
            img = draw_landmarks_from_array(img, right, connections=mp.solutions.hands.HAND_CONNECTIONS)

            # Display the result
            cv2.imshow("frame", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

        except StopIteration:
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            time.sleep(0.1)

    cv2.destroyAllWindows()


# Create and start threads
proc_thread = threading.Thread(target=processing_thread, daemon=True)
render_thread = threading.Thread(target=rendering_thread, daemon=True)

proc_thread.start()
render_thread.start()

# Wait for threads to complete
try:
    proc_thread.join()
    render_thread.join()
except KeyboardInterrupt:
    stop_event.set()
    proc_thread.join()
    render_thread.join()
    cv2.destroyAllWindows()