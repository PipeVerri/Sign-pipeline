from lm_processing import Landmarks
from utils.video import camera_reader, frame_reader
from utils.mediapipe.render import draw_landmarks_from_array
import cv2
import numpy as np
import mediapipe as mp
import threading
import time
from collections import deque

stop_event = threading.Event()
lm = Landmarks(max_frames_interpolation=200)
lm_generator = None
generator_lock = threading.Lock()

# Queue to pass frames from processing to rendering thread
frame_queue = deque(maxlen=100)  # Limit queue size to prevent memory issues
frame_queue_lock = threading.Lock()


def processing_thread():
    """Thread 1: Read frames and add landmarks to lm"""
    global lm_generator

    cap = cv2.VideoCapture("../../data/raw/test/1.mp4")
    # Start at minute 2 (120 seconds)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(120 * fps)  # 2 minutes * 60 seconds * fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    with mp.solutions.holistic.Holistic(model_complexity=2, static_image_mode=False) as holistic:
        # Initialize generator in this thread
        with generator_lock:
            lm_generator = lm.get_landmarks(continuous=True)

        for frame in frame_reader(cap, fps=6):
            if stop_event.is_set():
                print("Stopping processing thread...")
                break

            # Process frame with MediaPipe
            hol = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Add landmarks to the processor
            lm.add(hol.pose_landmarks, hol.left_hand_landmarks, hol.right_hand_landmarks)

            # Add frame to queue for rendering thread
            with frame_queue_lock:
                frame_queue.append(frame.copy())

            # Only log transitions (pose present -> not present and vice versa)
            if len(lm.pose) > 1:
                prev_present = lm.pose[-2] is not None
                curr_present = hol.pose_landmarks is not None
    stop_event.set()


def rendering_thread():
    """Thread 2: Get landmarks from generator and render on video frames"""
    global lm_generator

    # Wait for generator to be initialized
    while lm_generator is None and not stop_event.is_set():
        time.sleep(0.01)

    frame_count = 0

    while not stop_event.is_set():
        try:
            # Check if processing thread has added any frames
            if len(lm.pose) == 0:
                #time.sleep(0.01)
                continue

            # Get corresponding frame from queue
            with frame_queue_lock:
                if len(frame_queue) == 0:
                    #time.sleep(0.01)
                    continue
                frame = frame_queue.popleft()

            # Get next landmarks from generator with timeout protection
            with generator_lock:
                pose, left, right, jumped = next(lm_generator)
            frame_count += 1

            # Draw landmarks on the actual video frame
            if jumped:
                # Add red tint to indicate jumped frames
                frame = cv2.addWeighted(frame, 0.7, np.full_like(frame, (0, 0, 255)), 0.3, 0)

            #frame = np.zeros(frame.shape, dtype=np.uint8)
            img = draw_landmarks_from_array(frame, pose, connections=mp.solutions.pose.POSE_CONNECTIONS)
            img = draw_landmarks_from_array(img, left, connections=mp.solutions.hands.HAND_CONNECTIONS)
            img = draw_landmarks_from_array(img, right, connections=mp.solutions.hands.HAND_CONNECTIONS)

            scale_factor = 2.0  # Makes image 2x bigger
            new_width = int(img.shape[1] * scale_factor)
            new_height = int(img.shape[0] * scale_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

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
            #time.sleep(0.1)

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