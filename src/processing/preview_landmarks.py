import h5py
import numpy as np
import cv2
from pathlib import Path
import argparse
import json

# MediaPipe pose connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

# MediaPipe hand connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# Simplified face connections (outline and features)
FACE_CONNECTIONS = [
    # Face oval - just key points
    *[(i, i + 1) for i in range(10, 152, 3)],
    # Left eyebrow
    *[(i, i + 1) for i in range(70, 75)],
    # Right eyebrow
    *[(i, i + 1) for i in range(300, 305)],
    # Left eye
    (33, 133), (133, 155), (155, 154), (154, 153), (153, 145), (145, 144), (144, 163), (163, 7),
    # Right eye
    (362, 263), (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382),
    # Nose
    *[(i, i + 1) for i in range(168, 195, 2)],
    # Lips outer
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
]

BOUNDING_BOX_PADDING = 0.2


class LandmarkPreview:
    def __init__(self, h5_path, root_dir=None):
        self.h5_path = Path(h5_path)
        self.file = h5py.File(self.h5_path, 'r')
        self.current_person_idx = 0
        self.current_clip_idx = 0
        self.current_frame_idx = 0
        self.window_name = "Landmark Preview"

        # Determine root directory
        if root_dir:
            self.root_dir = Path(root_dir)
        else:
            # Try to infer from path structure
            self.root_dir = self._infer_root_dir()

        # Parse file structure
        self.people = list(self.file.keys())

        # Load video and bounding boxes
        self.video_path, self.bb_path = self._get_file_paths()
        self.cap = None
        self.bounding_boxes = None

        if self.video_path and self.video_path.exists():
            self.cap = cv2.VideoCapture(str(self.video_path))
            print(f"Loaded video: {self.video_path}")
        else:
            print(f"Warning: Could not find video file at {self.video_path}")

        if self.bb_path and self.bb_path.exists():
            with open(self.bb_path, 'r') as f:
                self.bounding_boxes = json.load(f)
            print(f"Loaded bounding boxes: {self.bb_path}")
        else:
            print(f"Warning: Could not find bounding boxes at {self.bb_path}")

        self.load_current_clip()

        print(f"\nLoaded {len(self.people)} people from {h5_path}")
        print("\nControls:")
        print("  n/p: Next/Previous frame")
        print("  c/C: Next/Previous clip")
        print("  w/W: Next/Previous person")
        print("  Space: Play current clip")
        print("  q: Quit")

    def _infer_root_dir(self):
        """Infer root directory from HDF5 path."""
        # Path structure: ROOT/data/processed/landmarks/[folder]/[file].h5
        path_parts = self.h5_path.parts
        if 'data' in path_parts:
            data_idx = path_parts.index('data')
            return Path(*path_parts[:data_idx])
        return Path.cwd()

    def _get_file_paths(self):
        """Get video and bounding box file paths from HDF5 path."""
        # HDF5 path: ROOT/data/processed/landmarks/[folder]/[file].h5
        # Video path: ROOT/data/raw/[folder]/video/[file].mp4
        # BB path: ROOT/data/processed/bounding_boxes/[folder]/[file].json

        path_parts = self.h5_path.parts

        # Find the folder name and whether it's unlabeled
        if 'landmarks' in path_parts:
            landmarks_idx = path_parts.index('landmarks')
            folder_path = '/'.join(path_parts[landmarks_idx + 1:-1])  # Everything between landmarks/ and filename
            filename = self.h5_path.stem + '.mp4'

            is_unlabeled = 'unlabeled' in folder_path
            base_folder = folder_path.replace('/unlabeled', '')

            # Construct paths
            if is_unlabeled:
                video_path = self.root_dir / 'data' / 'raw' / base_folder / 'unlabeled' / filename
            else:
                video_path = self.root_dir / 'data' / 'raw' / base_folder / 'video' / filename

            bb_path = self.root_dir / 'data' / 'processed' / 'bounding_boxes' / folder_path / (
                        self.h5_path.stem + '.json')

            return video_path, bb_path

        return None, None

    def load_current_clip(self):
        """Load current clip data."""
        person_key = self.people[self.current_person_idx]
        person_group = self.file[person_key]

        clips = [k for k in person_group.keys() if k.isdigit()]
        if not clips:
            self.clips = []
            self.pose_data = None
            return

        self.clips = sorted(clips, key=int)

        if self.current_clip_idx >= len(self.clips):
            self.current_clip_idx = 0

        clip_key = self.clips[self.current_clip_idx]
        clip_group = person_group[clip_key]

        self.pose_data = np.array(clip_group['pose_landmarks'])
        self.left_hand_data = np.array(clip_group['left_hand_landmarks'])
        self.right_hand_data = np.array(clip_group['right_hand_landmarks'])
        self.face_data = np.array(clip_group['face_landmarks'])
        self.timestamps = np.array(clip_group['timestamps'])
        self.clip_start = clip_group.attrs['start']
        self.clip_end = clip_group.attrs['end']

        # Get person ID from the key (e.g., "person_0" -> "0")
        self.current_person_id = person_key.split('_')[1]

        # Calculate max bounding box size for this clip
        self.clip_max_box_size = self._calculate_max_box_size()

        self.num_frames = len(self.timestamps)
        self.current_frame_idx = 0

        print(f"\nPerson: {person_key}, Clip: {clip_key}")
        print(f"  Frames: {self.num_frames}")
        print(f"  Duration: {self.clip_start:.2f}s - {self.clip_end:.2f}s")

    def _calculate_max_box_size(self):
        """Calculate the maximum bounding box size for the current clip."""
        if not self.bounding_boxes:
            return {'x': 0, 'y': 0}

        max_x = 0
        max_y = 0

        # Find all bounding boxes within this clip's time range
        for entry in self.bounding_boxes:
            timestamp = entry['timestamp']
            if self.clip_start <= timestamp <= self.clip_end:
                if self.current_person_id in entry['boxes']:
                    bbox = entry['boxes'][self.current_person_id]
                    x_size = bbox[2] - bbox[0]
                    y_size = bbox[3] - bbox[1]
                    max_x = max(max_x, x_size)
                    max_y = max(max_y, y_size)

        return {'x': max_x, 'y': max_y}

    def get_bounding_box_for_timestamp(self, timestamp):
        """Get the bounding box for a specific timestamp."""
        if not self.bounding_boxes:
            return None

        # Find the closest bounding box entry
        closest_entry = None
        min_diff = float('inf')

        for entry in self.bounding_boxes:
            diff = abs(entry['timestamp'] - timestamp)
            if diff < min_diff and self.current_person_id in entry['boxes']:
                min_diff = diff
                closest_entry = entry

        if closest_entry:
            bbox = closest_entry['boxes'][self.current_person_id]

            # Use the clip's max box size (same as during processing)
            x_size = bbox[2] - bbox[0]
            y_size = bbox[3] - bbox[1]
            x_center = bbox[0] + x_size / 2
            y_center = bbox[1] + y_size / 2

            # Use clip_max_box_size for consistent cropping
            x_distance_center = (self.clip_max_box_size['x'] * (1 + BOUNDING_BOX_PADDING)) / 2
            y_distance_center = (self.clip_max_box_size['y'] * (1 + BOUNDING_BOX_PADDING)) / 2

            x_start = max(0, x_center - x_distance_center)
            y_start = max(0, y_center - y_distance_center)

            crop_region = {
                'x_start': int(x_start),
                'y_start': int(y_start),
                'x_size': int(x_distance_center * 2),
                'y_size': int(y_distance_center * 2)
            }
            return crop_region

        return None

    def transform_landmarks_to_video(self, landmarks, crop_region):
        """Transform normalized landmarks from crop region to video coordinates."""
        if landmarks is None or np.all(np.isnan(landmarks)) or crop_region is None:
            return None

        transformed = landmarks.copy()
        # Landmarks are normalized (0-1) relative to the crop region
        # Transform to video pixel coordinates
        transformed[:, 0] = landmarks[:, 0] * crop_region['x_size'] + crop_region['x_start']
        transformed[:, 1] = landmarks[:, 1] * crop_region['y_size'] + crop_region['y_start']

        return transformed

    def draw_landmarks(self, canvas, landmarks, connections, color, thickness=2, point_radius=3):
        """Draw landmarks and connections on canvas."""
        if landmarks is None or np.all(np.isnan(landmarks)):
            return

        # Filter out NaN landmarks
        valid_mask = ~np.isnan(landmarks[:, 0])

        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                if valid_mask[start_idx] and valid_mask[end_idx]:
                    pt1 = (int(landmarks[start_idx, 0]), int(landmarks[start_idx, 1]))
                    pt2 = (int(landmarks[end_idx, 0]), int(landmarks[end_idx, 1]))
                    cv2.line(canvas, pt1, pt2, color, thickness)

        # Draw landmarks
        for i, point in enumerate(landmarks):
            if valid_mask[i]:
                pt = (int(point[0]), int(point[1]))
                cv2.circle(canvas, pt, point_radius, color, -1)
                # Draw white border for visibility
                cv2.circle(canvas, pt, point_radius + 1, (255, 255, 255), 1)

    def get_video_frame(self, timestamp):
        """Get video frame at specific timestamp."""
        if not self.cap or not self.cap.isOpened():
            return None

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if ret:
            return frame
        return None

    def render_frame(self):
        """Render current frame with landmarks overlaid on video."""
        if self.pose_data is None:
            canvas = np.ones((480, 640, 3), dtype=np.uint8) * 255
            cv2.putText(canvas, "No data available", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            return canvas

        # Get current frame data
        timestamp = self.timestamps[self.current_frame_idx]
        pose = self.pose_data[self.current_frame_idx]
        left_hand = self.left_hand_data[self.current_frame_idx]
        right_hand = self.right_hand_data[self.current_frame_idx]
        face = self.face_data[self.current_frame_idx]

        # Get video frame
        video_frame = self.get_video_frame(timestamp)

        if video_frame is None:
            canvas = np.ones((480, 640, 3), dtype=np.uint8) * 128
            cv2.putText(canvas, "Video frame not available", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            canvas = video_frame.copy()

        # Get bounding box and transform landmarks
        crop_region = self.get_bounding_box_for_timestamp(timestamp)

        if crop_region:
            # Draw bounding box
            x_start = int(crop_region['x_start'])
            y_start = int(crop_region['y_start'])
            x_end = int(crop_region['x_start'] + crop_region['x_size'])
            y_end = int(crop_region['y_start'] + crop_region['y_size'])
            cv2.rectangle(canvas, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)

            # Transform and draw landmarks
            pose_transformed = self.transform_landmarks_to_video(pose, crop_region)
            left_hand_transformed = self.transform_landmarks_to_video(left_hand, crop_region)
            right_hand_transformed = self.transform_landmarks_to_video(right_hand, crop_region)
            face_transformed = self.transform_landmarks_to_video(face, crop_region)

            # Draw all landmarks
            self.draw_landmarks(canvas, pose_transformed, POSE_CONNECTIONS, (0, 255, 255), 2, 4)
            self.draw_landmarks(canvas, left_hand_transformed, HAND_CONNECTIONS, (0, 255, 0), 2, 3)
            self.draw_landmarks(canvas, right_hand_transformed, HAND_CONNECTIONS, (255, 0, 0), 2, 3)
            self.draw_landmarks(canvas, face_transformed, FACE_CONNECTIONS, (255, 0, 255), 1, 2)

        # Info overlay with black background for readability
        person_key = self.people[self.current_person_idx]
        clip_key = self.clips[self.current_clip_idx] if self.clips else "N/A"
        info_text = [
            f"Person: {person_key} ({self.current_person_idx + 1}/{len(self.people)})",
            f"Clip: {clip_key} ({self.current_clip_idx + 1}/{len(self.clips)})",
            f"Frame: {self.current_frame_idx + 1}/{self.num_frames}",
            f"Time: {timestamp:.2f}s"
        ]

        # Draw semi-transparent background for text
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (400, 90), (0, 0, 0), -1)
        canvas = cv2.addWeighted(canvas, 0.7, overlay, 0.3, 0)

        for i, text in enumerate(info_text):
            cv2.putText(canvas, text, (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw legend
        legend_y = canvas.shape[0] - 100
        cv2.putText(canvas, "Pose", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(canvas, "L-Hand", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(canvas, "R-Hand", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(canvas, "Face", (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        return canvas

    def run(self):
        """Main preview loop."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        while True:
            frame = self.render_frame()
            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('n'):  # Next frame
                self.current_frame_idx = min(self.current_frame_idx + 1, self.num_frames - 1)
            elif key == ord('p'):  # Previous frame
                self.current_frame_idx = max(self.current_frame_idx - 1, 0)
            elif key == ord('c'):  # Next clip
                self.current_clip_idx = (self.current_clip_idx + 1) % len(self.clips)
                self.load_current_clip()
            elif key == ord('C'):  # Previous clip
                self.current_clip_idx = (self.current_clip_idx - 1) % len(self.clips)
                self.load_current_clip()
            elif key == ord('w'):  # Next person
                self.current_person_idx = (self.current_person_idx + 1) % len(self.people)
                self.current_clip_idx = 0
                self.load_current_clip()
            elif key == ord('W'):  # Previous person
                self.current_person_idx = (self.current_person_idx - 1) % len(self.people)
                self.current_clip_idx = 0
                self.load_current_clip()
            elif key == ord(' '):  # Space - play clip
                self.play_clip()

        cv2.destroyAllWindows()
        self.file.close()
        if self.cap:
            self.cap.release()

    def play_clip(self):
        """Play through the current clip."""
        fps = self.file.attrs.get('fps', 6)
        delay = int(1000 / fps)

        original_frame = self.current_frame_idx

        for i in range(self.current_frame_idx, self.num_frames):
            self.current_frame_idx = i
            frame = self.render_frame()
            cv2.imshow(self.window_name, frame)

            key = cv2.waitKey(delay) & 0xFF
            if key == ord(' ') or key == ord('q'):
                break

        # Reset to original frame if stopped
        if key != ord('q'):
            self.current_frame_idx = original_frame


def main():
    parser = argparse.ArgumentParser(description='Preview landmark data overlaid on video')
    parser.add_argument('h5_file', type=str, help='Path to HDF5 landmark file')
    parser.add_argument('--root', type=str, default=None,
                        help='Root directory (auto-detected if not provided)')

    args = parser.parse_args()

    if not Path(args.h5_file).exists():
        print(f"Error: File {args.h5_file} does not exist")
        return

    preview = LandmarkPreview(args.h5_file, args.root)
    preview.run()


if __name__ == "__main__":
    main()