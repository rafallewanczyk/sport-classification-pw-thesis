from dataclasses import dataclass

import numpy as np


@dataclass
class PoseDetection:
    video: str
    frame_id: int
    bbox: np.array
    skeleton: np.array
    score: float


@dataclass
class TrackedPoseDetection(PoseDetection):
    person_id: int
    unique_color: np.array

    @classmethod
    def from_pose_detection(cls, pose_detection: PoseDetection, person_id: int, unique_color: np.array):
        return cls(**pose_detection.__dict__, person_id=person_id, unique_color=unique_color)
