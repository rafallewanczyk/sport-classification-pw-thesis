# from abc import ABC, abstractmethod
# from copy import copy
# from typing import Tuple
#
# import cv2
# import numpy as np
# import pandas as pd
#
#
#
# class VideoSequenceService(ABC):
#     def __init__(self, sequence_length: int):
#         self.sequence_length = sequence_length
#
#     @staticmethod
#     def _fill_detections_holes(sequence):
#         first_non_none = None
#         for idx, el in enumerate(sequence):
#             if el is not None:
#                 first_non_none = el
#                 break
#         if first_non_none is None:
#             return None
#
#         current_non_ellipsis = first_non_none
#         replace_count = 0
#         for idx, el in enumerate(sequence):
#             if el is not None:
#                 current_non_ellipsis = el
#             else:
#                 sequence[idx] = copy(current_non_ellipsis)
#                 replace_count += 1
#         if replace_count / len(sequence) > 0.8:
#             return None
#         return sequence
#
#     def get_video_chunks(self, vid_len):
#         chunks = [(i, i + self.sequence_length - 1) for i in range(1, vid_len + 1, self.sequence_length)]
#         if chunks[-1][1] > vid_len:
#             del chunks[-1]
#         return chunks
#
#     @staticmethod
#     def load_sequence(video_capture: cv2.VideoCapture,  chunk: Tuple[int, int]):
#         read_video = [video_capture.read() for frame_idx in range(chunk[0], chunk[1] + 1)]
#         raw_video = np.array([frame[1] for frame in read_video])
#         error = np.array([frame[0] for frame in read_video])
#         assert np.all(error), 'there was error reading the video'
#         return raw_video
#
#
#     @abstractmethod
#     def process_sequence(self, video_capture: cv2.VideoCapture, detections: pd.DataFrame):
#         ...
