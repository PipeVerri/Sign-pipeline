import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# Module-level cache to track image metadata
_image_metadata = {}

def draw_landmarks_from_array(image, landmarks_array, connections=None,
                              color=(0, 255, 0), radius=2, thickness=1):
    if landmarks_array is None:
        return image

    # Use image id to track metadata across calls
    img_id = id(image)

    # Check if we have metadata from previous calls
    if img_id in _image_metadata:
        orig_h, orig_w = _image_metadata[img_id]['original_shape']
        curr_pad_top, curr_pad_left = _image_metadata[img_id]['current_padding']
    else:
        # First call - store original dimensions
        orig_h, orig_w = image.shape[:2]
        curr_pad_top, curr_pad_left = 0, 0

    # Convert normalized landmarks relative to ORIGINAL dimensions
    points = []
    for lm in landmarks_array:
        if hasattr(lm, "x") and hasattr(lm, "y"):
            x_norm, y_norm = lm.x, lm.y
        else:
            x_norm, y_norm = lm[0], lm[1]

        # Use original dimensions for denormalization
        x, y = int(x_norm * orig_w), int(y_norm * orig_h)
        # Apply current padding offset
        x += curr_pad_left
        y += curr_pad_top
        points.append((x, y))

    # ---- Determine additional padding needed ----
    h, w = image.shape[:2]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    pad_left = max(0, -min_x)
    pad_top = max(0, -min_y)
    pad_right = max(0, max_x - (w - 1))
    pad_bottom = max(0, max_y - (h - 1))

    # ---- Create padded canvas if needed ----
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        new_w = w + pad_left + pad_right
        new_h = h + pad_top + pad_bottom

        padded = np.zeros((new_h, new_w, 3), dtype=image.dtype)
        padded[pad_top:pad_top + h, pad_left:pad_left + w] = image

        # Clean up old image metadata
        if img_id in _image_metadata:
            del _image_metadata[img_id]

        image = padded

        # Update cumulative padding
        curr_pad_top += pad_top
        curr_pad_left += pad_left

        # Shift points by new padding
        points = [(x + pad_left, y + pad_top) for (x, y) in points]

    # ---- Draw landmarks ----
    for (x, y) in points:
        cv2.circle(image, (x, y), radius, color, -1)

    # ---- Draw connections ----
    if connections:
        for start_idx, end_idx in connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(image, points[start_idx], points[end_idx], color, thickness)

    # Store metadata for next call
    _image_metadata[id(image)] = {
        'original_shape': (orig_h, orig_w),
        'current_padding': (curr_pad_top, curr_pad_left)
    }

    return image

def render_frame(frame, pose_array=None, left_hand_array=None,
                 right_hand_array=None, window_name="Render desde np.array"):
    annotated = frame.copy()

    # Pose
    if pose_array is not None:
        annotated = draw_landmarks_from_array(
            annotated, pose_array, connections=POSE_CONNECTIONS, color=(0, 255, 0)
        )

    # Mano izquierda
    if left_hand_array is not None:
        annotated = draw_landmarks_from_array(
            annotated, left_hand_array, connections=HAND_CONNECTIONS, color=(255, 0, 0)
        )

    # Mano derecha
    if right_hand_array is not None:
        annotated = draw_landmarks_from_array(
            annotated, right_hand_array, connections=HAND_CONNECTIONS, color=(0, 0, 255)
        )

    cv2.imshow(window_name, annotated)
    cv2.waitKey(1)
    return annotated


def render_holistic_frame(frame, holistic_res):
    pose_landmarks = None
    left_hand_landmarks = None
    right_hand_landmarks = None

    if holistic_res.pose_landmarks:
        pose_landmarks = holistic_res.pose_landmarks.landmark
    if holistic_res.left_hand_landmarks:
        left_hand_landmarks = holistic_res.left_hand_landmarks.landmark
    if holistic_res.right_hand_landmarks:
        right_hand_landmarks = holistic_res.right_hand_landmarks.landmark

    render_frame(frame, pose_landmarks, left_hand_landmarks, right_hand_landmarks)
