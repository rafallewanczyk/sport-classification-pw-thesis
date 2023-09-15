import inspect
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class SequenceDto:
    video: str
    frame_id: int
    bbox: np.array
    max_wh: np.array
    skeleton: np.array
    skeleton_normalized: np.array = field(init=False)
    skeleton_angles: np.array = field(init=False)
    visual_features: np.array
    visual_features_red: np.array
    frame_vis_features: np.array
    frame_vis_features_red: np.array
    score: float
    person_id: int
    frame_copied_from: int
    start_frame: int

    VIDEO: str = 'video'
    FRAME_ID: str = 'frame_id'
    MAX_WH: str = 'max_wh'
    BBOX: str = 'bbox'
    SKELETON: str = 'skeleton'
    SKELETON_NORMALIZED: str = 'skeleton_normalized'
    SKELETON_ANGLES: str = 'skeleton_angles'
    VISUAL_FEATURES: str = 'visual_features'
    VISUAL_FEATURES_RED: str = 'visual_features_red'
    FRAME_VIS_FEATURES: str = 'frame_vis_features'
    FRAME_VIS_FEATURES_RED: str = 'frame_vis_features_red'
    SCORE: str = 'score'
    PERSON_ID: str = 'person_id'
    FRAME_COPIED_FROM = 'frame_copied_from'
    START_FRAME: str = 'start_frame'

    BODY_PARTS = {
        'nose': 0,
        'right eye': 1,
        'left eye': 2,
        'right ear': 3,
        'left ear': 4,
        'right arm': 5,
        'left arm': 6,
        'right elbow': 7,
        'left elbow': 8,
        'right hand': 9,
        'left hand': 10,
        'right hip': 11,
        'left hip': 12,
        'right knee': 13,
        'left knee': 14,
        'right foot': 15,
        'left foot': 16,
    }

    def __post_init__(self):
        self._fix_bbox()
        self.skeleton_normalized = self._normalize_skeleton(self.skeleton, np.array([self.bbox[0], self.bbox[1]]))
        self.skeleton_angles = self._calculate_skeleton_angles(self.skeleton)

    def _fix_bbox(self):
        self.bbox = np.array([0 if self.bbox[0] < 0 else self.bbox[0],
                              0 if self.bbox[1] < 0 else self.bbox[1],
                              0 if self.bbox[2] < 0 else self.bbox[2],
                              0 if self.bbox[3] < 0 else self.bbox[3]])

    def _normalize_skeleton(self, skeleton: np.array, shift_point: np.array):
        to_normalize = skeleton.copy()
        to_normalize[::3] -= shift_point[0]
        to_normalize[1::3] -= shift_point[1]
        to_normalize[::3] /= self.max_wh[0]
        to_normalize[1::3] /= self.max_wh[1]
        return to_normalize

    @staticmethod
    def _angle_between_points(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        unit_vector_1 = v1 / np.linalg.norm(v1)
        unit_vector_2 = v2 / np.linalg.norm(v2)
        dot_product = np.dot(unit_vector_1, unit_vector_2)
        dot_product = max(-1, dot_product)
        dot_product = min(dot_product, 1)
        if dot_product > 1 or dot_product < -1:
            print(unit_vector_2, unit_vector_2, dot_product)
        angle = np.arccos(dot_product)

        return min(2 * np.pi - angle, angle) / (2 * np.pi)

    def _calculate_skeleton_angles(self, skeleton: np.array):
        skeleton = skeleton.reshape((17, 3))
        return np.array([
            self._angle_between_points(skeleton[self.BODY_PARTS['left foot']][:2],
                                       skeleton[self.BODY_PARTS['left knee']][:2],
                                       skeleton[self.BODY_PARTS['left hip']][:2]),
            self._angle_between_points(skeleton[self.BODY_PARTS['left knee']][:2],
                                       skeleton[self.BODY_PARTS['left hip']][:2],
                                       skeleton[self.BODY_PARTS['left arm']][:2]),
            self._angle_between_points(skeleton[self.BODY_PARTS['left knee']][:2],
                                       skeleton[self.BODY_PARTS['left hip']][:2],
                                       skeleton[self.BODY_PARTS['right hip']][:2]),
            self._angle_between_points(skeleton[self.BODY_PARTS['left hand']][:2],
                                       skeleton[self.BODY_PARTS['left elbow']][:2],
                                       skeleton[self.BODY_PARTS['left arm']][:2]),
            self._angle_between_points(skeleton[self.BODY_PARTS['left elbow']][:2],
                                       skeleton[self.BODY_PARTS['left arm']][:2],
                                       skeleton[self.BODY_PARTS['left ear']][:2]),
            self._angle_between_points(skeleton[self.BODY_PARTS['left elbow']][:2],
                                       skeleton[self.BODY_PARTS['left arm']][:2],
                                       skeleton[self.BODY_PARTS['right arm']][:2]),
            self._angle_between_points(skeleton[self.BODY_PARTS['right foot']][:2],
                                       skeleton[self.BODY_PARTS['right knee']][:2],
                                       skeleton[self.BODY_PARTS['right hip']][:2]),
            self._angle_between_points(skeleton[self.BODY_PARTS['right knee']][:2],
                                       skeleton[self.BODY_PARTS['right hip']][:2],
                                       skeleton[self.BODY_PARTS['right arm']][:2]),
            self._angle_between_points(skeleton[self.BODY_PARTS['right knee']][:2],
                                       skeleton[self.BODY_PARTS['right hip']][:2],
                                       skeleton[self.BODY_PARTS['left hip']][:2]),
            self._angle_between_points(skeleton[self.BODY_PARTS['right hand']][:2],
                                       skeleton[self.BODY_PARTS['right elbow']][:2],
                                       skeleton[self.BODY_PARTS['right arm']][:2]),
            self._angle_between_points(skeleton[self.BODY_PARTS['right elbow']][:2],
                                       skeleton[self.BODY_PARTS['right arm']][:2],
                                       skeleton[self.BODY_PARTS['right ear']][:2]),
            self._angle_between_points(skeleton[self.BODY_PARTS['right elbow']][:2],
                                       skeleton[self.BODY_PARTS['right arm']][:2],
                                       skeleton[self.BODY_PARTS['left arm']][:2]),
            (skeleton[:, 0::3].max() - skeleton[:, 0::3].min()) / 1080,
            self.bbox[0] / 1920,
            self.bbox[1] / 1080
        ])

    @classmethod
    def from_dict(cls, env):
        return cls(**{
            k: v for k, v in env.items()
            if k in inspect.signature(cls).parameters
        })

    @classmethod
    def get_cols(cls) -> List[str]:
        return [cls.VIDEO, cls.FRAME_ID, cls.BBOX, cls.MAX_WH, cls.SKELETON, cls.SKELETON_NORMALIZED,
                cls.SKELETON_ANGLES, cls.VISUAL_FEATURES, cls.VISUAL_FEATURES_RED, cls.FRAME_VIS_FEATURES,
                cls.FRAME_VIS_FEATURES_RED, cls.SCORE, cls.PERSON_ID, cls.FRAME_COPIED_FROM, cls.START_FRAME]
