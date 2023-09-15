from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DetectionDto:
    video: str
    frame_id: int
    bbox: np.array
    skeleton: np.array
    score: float
    person_id: int
    unique_color: np.array
    visual_features: Optional[np.array] = None
    visual_features_red: Optional[np.array] = None
    frame_vis_features: Optional[np.array] = None
    frame_vis_features_red: Optional[np.array] = None

    VIDEO: str = 'video'
    FRAME_ID: str = 'frame_id'
    BBOX: str = 'bbox'
    SKELETON: str = 'skeleton'
    SCORE: str = 'score'
    PERSON_ID: str = 'person_id'
    UNIQUE_COLOR: str = 'unique_color'
    VISUAL_FEATURES: str = 'visual_features'
    VISUAL_FEATURES_RED: str = 'visual_features_red'
    FRAME_VIS_FEATURES: str = 'frame_vis_features'
    FRAME_VIS_FEATURES_RED: str = 'frame_vis_features_red'
