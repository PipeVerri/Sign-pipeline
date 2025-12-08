import time

from src.utils.mediapipe.parse import mp_to_arr
import numpy as np
from src.utils.ds.segment_tree import SegmentTree
from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path

np.seterr(divide='raise', invalid='raise')

def nn_parser(pose, left, right):
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
    # Centrar todos los puntos del cuerpo sean hip-centric, y los de la mano(excepto la muñeca) sean wrist centric
    # Hip-centric
    hip_distance_v = pose[24] - pose[23]
    hip_center = pose[24] - (hip_distance_v / 2)
    pose = pose - hip_center
    left = left - hip_center
    right = right - hip_center
    # Wrist-centric
    left[1:] = left[1:] - left[0].reshape(1, 3)
    right[1:] = right[1:] - right[0].reshape(1, 3)

    # Eliminar los puntos mano y cara(no la voy a usar por ahora) de la pose
    mask = np.ones(pose.shape[0], dtype=bool)
    mask[15:23] = False
    mask[0:11] = False
    mask[25:] = False # Por si las dudas
    pose = pose[mask]
    return np.concatenate((pose, left, right), axis=0).flatten()


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

    def __init__(self, max_frames_interpolation=48, fps=12):
        self.pose = []
        self.empty_pose = SegmentTree()
        self.left = self.Hand()
        self.right = self.Hand()
        self.max_frames_interpolation = max_frames_interpolation
        self.fps = fps

        if Landmarks._neutral_hand is None:
            path = Path(__file__).resolve().with_name("neutral_hand.npy")
            Landmarks._neutral_hand = np.load(path)

    def add(self, pose, left, right):
        self._mediapipe_parser(pose, self.pose, self.empty_pose, pose=True)
        self._mediapipe_parser(left, self.left.lm, self.left.empty)
        self._mediapipe_parser(right, self.right.lm, self.right.empty)

    def _mediapipe_parser(self, lm, arr, s, pose=False):
        if lm:
            r_idx = 25 if pose else len(lm.landmark)
            arr.append(mp_to_arr(lm.landmark[:r_idx]))
        else:
            s.add_point(len(arr))
            arr.append(None)

    def _interpolate(self, start, end, arr):
        interpol_diff = arr[end + 1] - arr[start - 1]
        interpol_length = end - start + 1
        interpol_diff = interpol_diff / interpol_length
        for i in range(interpol_length):
            yield arr[start - 1] + interpol_diff * (i + 1)
        yield None

    def get_landmarks(self, continuous=False, compute_accel=False):
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
        """
        # Un generador que va retornando los landmarks a procesar
        current_frame = 0
        pose_interpolated = None

        while True:
            # Empezar interpolando los datos y sacando los rangos missing demasiado grandes
            # Empezar con el pose asi tengo algo de lo cual puedo agarrar las manos
            if current_frame == len(self.pose):
                if continuous:
                    time.sleep(0.001)
                    continue
                else:
                    break

            if pose_interpolated is not None: # Fijarme si estoy interpolando
                pose_frame = next(pose_interpolated)
                if pose_frame is None:
                    pose_interpolated = None

            # Fijarme si puedo retornar la pose asi nomas
            if (self.pose[current_frame] is not None) and (pose_interpolated is None):
                pose_frame = self.pose[current_frame]
            else:
                # Fijarme si tengo limite por derecha para interpolar
                start, end = self.empty_pose.get_interval(current_frame)
                if end != current_frame and (end + 1) < len(self.pose): # Si hay limite y puedo interpolar
                    interpol_length = end - start + 1
                    # Fijarme si tomarlo como 2 secuencias diferentes o no
                    if interpol_length > self.max_frames_interpolation:
                        current_frame = end + 1 # Saltar a la siguiente secuencia
                        pose_frame = self.pose[current_frame]
                    else:
                        pose_interpolated = self._interpolate(start, end, self.pose)
                        pose_frame = next(pose_interpolated)
                else: # Si no puedo interpolar, esperar a re-capturar la pose o que haya pasado demasiado tiempo
                    if current_frame - start >= self.max_frames_interpolation:
                        if continuous:
                            current_frame = len(self.pose)
                            continue
                        else:
                            break
                    else:
                        continue

            # Setearlo para saber el pose_frame generado en el i frame. No hay problema porque no lo cuenta dentro del SegmentTree
            self.pose[current_frame] = pose_frame

            # Ahora procesar las manos
            left_frame = self._process_hand(self.left, current_frame, pose_frame, 13, 15, compute_accel=compute_accel)
            right_frame = self._process_hand(self.right, current_frame, pose_frame, 14, 16, compute_accel=compute_accel)

            # Visualizarlas
            yield pose_frame, left_frame, right_frame
            current_frame += 1

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

    def _process_hand(self, hand: Hand, current_frame, pose_frame, elbow_num, wrist_num, compute_accel=False):
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
            if start != 0:
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

            # Ahora calcular la aceleracion respecto al wrist
            if compute_accel:
                hand.positions.append(hand_frame)
                if len(hand.positions) > 1:
                    hand.velocities.append((hand.positions[-1] - hand.positions[-2]) / self.fps)
                if len(hand.velocities) > 1:
                    accel = (hand.velocities[-1] - hand.velocities[-2]) / self.fps

                else:
                    accel = 0

            # Ahora posicionar la mano en el lugar correcto. El wrist de la mano(0) en el wrist del pose(15)
            # mano_wrist_x + x = pose_wrist_x => x = pose_wrist_x - mano_wrist_x
            x_offset = pose_frame[wrist_num][0] - hand_frame[0][0]
            y_offset = pose_frame[wrist_num][1] - hand_frame[0][1]
            z_offset = pose_frame[wrist_num][2] - hand_frame[0][2]
            hand_frame[:, 0] += x_offset
            hand_frame[:, 1] += y_offset
            hand_frame[:, 2] += z_offset

            return hand_frame