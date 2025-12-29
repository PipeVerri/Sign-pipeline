import time

from fontTools.misc.plistlib import end_key
from scipy.linalg.cython_lapack import shgeqz

from src.shared.utils.mediapipe.parse import mp_to_arr
import numpy as np
from src.shared.utils.ds.segment_tree import SegmentTree
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path

np.seterr(divide='raise', invalid='raise')


def make_hip_centric(pose):
    """
    Transforms pose data to be centered around the hip midpoint.

    Parameters:
    pose: np.ndarray
        A NumPy array representing the coordinates of body keypoints.
        Expected to have hip keypoints at indices 23 and 24.

    Returns:
    np.ndarray
        The pose array with all points translated so the hip center is at the origin.
    """
    hip_distance_v = pose[24] - pose[23]
    hip_center = pose[24] - (hip_distance_v / 2)
    return pose - hip_center


def nn_parser(pose, left, right, face):
    """
    Parses and transforms pose, left hand, and right hand data into a standardized,
    hip and wrist centric format. The function ensures the body points are
    centered around the hips, and hand points (excluding the wrist) are centered
    around the respective wrists. It also removes unnecessary points like hands
    and face from the pose data before combining all into a single flattened
    array.

    Parameters:
    pose: np.ndarray
        A NumPy array representing the coordinates of body keypoints. The points
        are processed to be centered around the hip.
    left: np.ndarray
        A NumPy array representing the keypoints of the left hand. The points are
        processed to be centered around the wrist.
    right: np.ndarray
        A NumPy array representing the keypoints of the right hand. The points are
        processed to be centered around the wrist.

    Returns:
    np.ndarray
        A flattened NumPy array combining the processed body, left hand, and right
        hand keypoints in a standardized format.
    """
    # Calculate hip center before modifying pose
    hip_distance_v = pose[24] - pose[23]
    hip_center = pose[24] - (hip_distance_v / 2)

    # Make all points hip-centric(also move the hands so they dont get detached from the pose)
    pose = pose - hip_center
    left = left - hip_center
    right = right - hip_center

    # Wrist-centric for hands (excluding the wrist itself)
    left[1:] = left[1:] - left[0].reshape(1, 3)
    right[1:] = right[1:] - right[0].reshape(1, 3)

    # Remove hand and face points from pose
    mask = np.ones(pose.shape[0], dtype=bool)
    mask[15:23] = False
    mask[0:11] = False
    mask[25:] = False  # Just in case that the pose array hasnt been trimmed
    pose = pose[mask]

    return np.concatenate((pose, left, right, face), axis=0).flatten()

class Landmarks:
    _neutral_hand = None

    @dataclass
    class Hand:
        lm: List[np.ndarray] = field(default_factory=list)
        empty: SegmentTree = field(default_factory=SegmentTree)
        angles: Dict[int, np.ndarray] = field(default_factory=dict)
        ratio = None
        positions: List[np.ndarray] = field(default_factory=list)
        velocities: List[np.ndarray] = field(default_factory=list)

    @dataclass
    class InterpolatedLandmarks:
        lm: List[np.ndarray] = field(default_factory=list)
        empty: SegmentTree = field(default_factory=SegmentTree)
        interpolator = None

    def __init__(self, max_frames_interpolation=48, max_face_frames_interpolation=12):
        self.pose = self.InterpolatedLandmarks()
        self.face = self.InterpolatedLandmarks()
        self.left = self.Hand()
        self.right = self.Hand()
        self.max_frames_interpolation = max_frames_interpolation
        self.max_face_frames_interpolation = max_face_frames_interpolation

        if Landmarks._neutral_hand is None:
            path = Path(__file__).resolve().with_name("neutral_hand.npy")
            Landmarks._neutral_hand = np.load(path)

    def add(self, pose, left, right, face):
        self._mediapipe_parser(pose, self.pose, pose=True)
        self._mediapipe_parser(left, self.left)
        self._mediapipe_parser(right, self.right)
        self._mediapipe_parser(face, self.face)

    def _mediapipe_parser(self, lm, store, pose=False):
        if lm:
            r_idx = 25 if pose else len(lm)
            store.lm.append(mp_to_arr(lm[:r_idx]))
        else:
            store.empty.add_point(len(store.lm))
            store.lm.append(None)

    def _interpolate(self, start, end, store):
        interpol_diff = store.lm[end + 1] - store.lm[start - 1]
        interpol_length = end - start + 1
        interpol_diff = interpol_diff / interpol_length
        for i in range(interpol_length):
            yield store.lm[start - 1] + interpol_diff * (i + 1)
        yield None

    def get_landmarks(self, continuous=False, return_frame_number=False):
        """
        Generator function that processes and returns landmarks frame by frame, handling pose interpolations
        and missing data conditions. It also processes hand landmarks for both left and right hands while
        optionally computing acceleration data.

        Parameters:
        continuous: bool, optional
            If True, the generator continues to run even after processing all frames by introducing a brief
            delay. Defaults to False.

        compute_accel: bool, optional
            If True, computes acceleration for the processed hand landmarks. Defaults to False.

        Yields:
        tuple
            A tuple containing:
            - pose_frame: The interpolated or processed pose frame data for the current frame.
            - left_frame: Processed left-hand landmarks for the current frame.
            - right_frame: Processed right-hand landmarks for the current frame.
            - jumped: A boolean indicating whether the generator jumped to a new frame due to missing data.
        """
        current_frame = 0
        jumped = False  # Track if we just jumped over a gap

        while True:
            # Check if we've reached the end of available frames
            if current_frame >= len(self.pose.lm):
                if continuous:
                    time.sleep(0.001)
                    continue
                else:
                    break

            pose_frame, jumped_position, should_wait = self._get_interpolated_frame(self.pose, current_frame, self.max_frames_interpolation)
            if jumped_position is not None:
                current_frame = jumped_position
                jumped = True
                continue
            elif should_wait:
                if not continuous: # There is no point in waiting, no new frames will be passed
                    break
                else: # We should wait until I can either interpolate or find a frame to jump to
                    time.sleep(0.001)
                    continue

            # We have a valid pose_frame - update the array
            self.pose.lm[current_frame] = pose_frame

            # Now process the face
            face_frame, _, _ = self._get_interpolated_frame(self.face, current_frame, self.max_face_frames_interpolation)
            if face_frame is None:
                face_frame = np.zeros(478)

            # Process hands
            left_frame = self._process_hand(self.left, current_frame, pose_frame, 13, 15)
            right_frame = self._process_hand(self.right, current_frame, pose_frame, 14, 16)

            # Yield results (pose, left_hand, right_hand, face_frame, jumped)
            if continuous:
                yield pose_frame, left_frame, right_frame, face_frame, jumped
            elif return_frame_number:
                yield pose_frame, left_frame, right_frame, face_frame, current_frame
            else:
                yield pose_frame, left_frame, right_frame, face_frame

            current_frame += 1
            jumped = False

    def _get_interpolated_frame(self, store, current_frame, max_interpolation):
        """
        :param store: The landmarks store
        :param current_frame: The current_frame
        :return: A tuple with:
        - The interpolated/fetched frame or None if it cant be interpolated and it should jump/return "no_frame"
        - Jumped frame. None if it shouldnt jump, the frame number if it should jump
        - Should wait. True if it should busy-wait/break, False if it shouldnt
        """
        # If currently interpolating, get next interpolated frame
        if store.interpolator is not None:
            frame = next(store.interpolator)
            if frame is None:
                store.interpolator = None
            else:
                return frame, None, False
        if store.lm[current_frame] is not None:
            return store.lm[current_frame], None, False
        else:
            # Current frame is None - check if we can/should interpolate
            start, end = store.empty.get_interval(current_frame)
            has_left_limit = start != 0
            has_right_limit = (end + 1) < len(store.lm)
            # Special case: if we're at the beginning (start == 0) and the current frame is None, we cant interpolate
            if not has_left_limit:
                if has_right_limit:
                    return None, end + 1, False
                else:
                    return None, None, True
            else:
                if has_right_limit and store.lm[end + 1] is not None: # If we can interpolate
                    interpol_length = end - start + 1
                    if interpol_length > max_interpolation:
                        # Gap too large - skip to next valid sequence
                        return None, end + 1, False
                    else:
                        # Start interpolation
                        store.interpolator = self._interpolate(start, end, store)
                        return next(store.interpolator), None, False
                else: # We dont have a right limit yet, we should wait/break
                   return None, None, True

    def _rodrigues(self, vec1, vec2):
        v1 = vec1 / np.linalg.norm(vec1)
        v2 = vec2 / np.linalg.norm(vec2)

        # Producto cruzado y producto punto
        cross = np.cross(v1, v2)
        dot = np.dot(v1, v2)

        # Caso: vectores casi iguales → no se necesita rotación
        if dot > 0.999999:
            return np.eye(3)

        # Caso: vectores opuestos → rotación de 180° alrededor de un eje perpendicular arbitrario
        if dot < -0.999999:
            # buscar un eje perpendicular a v1
            axis = np.cross(v1, np.array([1, 0, 0]))
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(v1, np.array([0, 1, 0]))
            axis /= np.linalg.norm(axis)

            # 180 degrees
            angle = np.pi
        else:
            # eje normal
            axis = cross / np.linalg.norm(cross)
            angle = np.arccos(dot)

        # Matriz de rotación (Rodrigues)
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])

        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        return R

    def _process_hand(self, hand: Hand, current_frame, pose_frame, elbow_num, wrist_num):
        # Ahora fijarme si puedo retornar la mano
        if hand.lm[current_frame] is not None:
            if hand.ratio is None:
                hand_vec_length = np.linalg.norm(hand.lm[current_frame][0] - hand.lm[current_frame][9])
                forearm_vec_length = np.linalg.norm(pose_frame[elbow_num] - pose_frame[wrist_num])
                hand.ratio = forearm_vec_length / hand_vec_length
            hand_frame = hand.lm[current_frame]
        else:
            # Ya checkee el limite de longitud de interpolacion antes. No voy a interpolar por ahora
            start, end = hand.empty.get_interval(current_frame)
            # Fijarme si tengo limite por izquierda
            if start != 0:
                hand_frame = hand.lm[start - 1]
            else:  # Usar la mano default
                hand_frame = Landmarks._neutral_hand

            # Empiezo moviendo la muñeca a 0, 0 asi lo roto usandola como eje de coordenadas
            wrist_pos = hand_frame[0].reshape(1, 3)
            hand_frame = hand_frame - wrist_pos  # Restarle wrist_pos a cada una de las columnas de left_frame

            # Rotar la mano para que sufra la misma rotacion que sufrio el antebrazo
            # Ver que vector y que angulo describen la rotacion de v_antebrazo. Esta dado por el codo y la muñeca(de la pose)
            v_forearm_new = pose_frame[elbow_num] - pose_frame[wrist_num]
            if start != 0 and self.pose[start - 1] is not None:
                v_forearm_old = self.pose[start - 1][elbow_num] - self.pose[start - 1][wrist_num]  # Origen de coordenadas en el codo
                # Ahora calcular el eje y el angulo
                R = self._rodrigues(v_forearm_old, v_forearm_new)
            else:
                # No se como estaba la mano originalmente rotada, asi que aplicarle una rotacion no tiene sentido
                # Quiero encontrar la rotacion que haga que v_forearm_norm = v_mano_norm
                # Para eso puedo usar Rodrigues. El eje de rotacion es v_forearm_norm x v_mano_norm, y calculo el angulo entre ellos
                v_forearm_norm = v_forearm_new / np.linalg.norm(v_forearm_new)
                v_wrist_norm = hand_frame[0] - hand_frame[9]
                v_wrist_norm = v_wrist_norm / np.linalg.norm(v_wrist_norm)
                R = self._rodrigues(v_wrist_norm, v_forearm_norm)

            # Aplicar la rotacion
            hand_frame = (R @ hand_frame.T).T

            # Escalar la mano para que tenga el ratio correcto
            forearm_size = np.linalg.norm(pose_frame[elbow_num] - pose_frame[wrist_num])
            hand_size = np.linalg.norm(hand_frame[0] - hand_frame[9])
            target_hand_size = forearm_size / (6.1 if (hand.ratio is None) else hand.ratio)
            hand_frame *= target_hand_size / hand_size

            # Ahora posicionar la mano en el lugar correcto. El wrist de la mano(0) en el wrist del pose(15)
            # mano_wrist_x + x = pose_wrist_x => x = pose_wrist_x - mano_wrist_x
            x_offset = pose_frame[wrist_num][0] - hand_frame[0][0]
            y_offset = pose_frame[wrist_num][1] - hand_frame[0][1]
            z_offset = pose_frame[wrist_num][2] - hand_frame[0][2]
            hand_frame[:, 0] += x_offset
            hand_frame[:, 1] += y_offset
            hand_frame[:, 2] += z_offset

        return hand_frame