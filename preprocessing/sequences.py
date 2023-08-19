from typing import List, Optional

import numpy as np
import pandas as pd

from preprocessing.dto.detection_dto import DetectionDto
from preprocessing.dto.sequence_dto import SequenceDto
from scipy.signal import savgol_filter
from functools import partial
import math


class Sequences:
    def __init__(self, sequence_length: int, detections: pd.DataFrame):
        self.sequence_length = sequence_length
        self.detections = self.filter_detections(detections)
        self.detections.sort_values(DetectionDto.FRAME_ID, ascending=True, inplace=True)

    @staticmethod
    def filter_detections(detections_df):
        detections_df['avg_skeleton_accuracy'] = detections_df[DetectionDto.SKELETON].apply(
            lambda val: np.average(val[2::3]))
        filtered_detections = detections_df[detections_df['avg_skeleton_accuracy'] > 0.5].drop(
            ['avg_skeleton_accuracy'], axis=1)
        return filtered_detections

    def fill_detections_holes(self, sp_detections: pd.DataFrame, frames_chunk: np.array) -> pd.DataFrame:
        max_bbox = np.array(sp_detections[SequenceDto.BBOX].to_list()).max(axis=0)
        max_wh = np.array([max_bbox[2], max_bbox[3]])
        first_frame = sp_detections[DetectionDto.FRAME_ID].min()
        first_detection = DetectionDto(
            **sp_detections[sp_detections[DetectionDto.FRAME_ID] == first_frame].to_dict('records')[0]
        )
        missing_first_frames = list(frames_chunk).index(first_detection.frame_id)

        sequences = []
        filled_holes = 0
        for idx in range(missing_first_frames):
            sequences.append(SequenceDto(
                video=first_detection.video,
                frame_id=frames_chunk[idx],
                max_wh=max_wh,
                bbox=first_detection.bbox.astype(float),
                skeleton=first_detection.skeleton,
                visual_features=first_detection.visual_features,
                visual_features_pca=first_detection.visual_features_pca,
                frame_vis_features=first_detection.frame_vis_features,
                frame_vis_features_pca=first_detection.frame_vis_features_pca,
                score=first_detection.score,
                person_id=first_detection.person_id,
                frame_copied_from=first_detection.frame_id,
                start_frame=frames_chunk[0],
            ))
            filled_holes += 1
        for frame_id in frames_chunk[missing_first_frames:]:
            if not sp_detections[sp_detections[DetectionDto.FRAME_ID] == frame_id].empty:
                last_proper_detection = DetectionDto(
                    **sp_detections[sp_detections[DetectionDto.FRAME_ID] == frame_id].to_dict('records')[0]
                )
            else:
                available_detections = sp_detections[sp_detections[DetectionDto.FRAME_ID] <= frame_id]
                last_proper_frame = available_detections[DetectionDto.FRAME_ID].max()
                last_proper_detection = DetectionDto(
                    **sp_detections[sp_detections[DetectionDto.FRAME_ID] == last_proper_frame].to_dict('records')[0]
                )
                filled_holes += 1

            sequences.append(SequenceDto(
                video=last_proper_detection.video,
                frame_id=frame_id,
                bbox=last_proper_detection.bbox.astype(float),
                max_wh=max_wh,
                skeleton=last_proper_detection.skeleton,
                visual_features=last_proper_detection.visual_features,
                visual_features_pca=last_proper_detection.visual_features_pca,
                frame_vis_features=last_proper_detection.frame_vis_features,
                frame_vis_features_pca=last_proper_detection.frame_vis_features_pca,
                score=last_proper_detection.score,
                person_id=last_proper_detection.person_id,
                frame_copied_from=last_proper_detection.frame_id,
                start_frame=frames_chunk[0],
            ))
        if filled_holes / self.sequence_length > 0.8:
            return pd.DataFrame(columns=SequenceDto.get_cols())

        df = pd.DataFrame(sequences, columns=SequenceDto.get_cols())
        return df

    def get_frames_chunks(self, drop_num):
        unique_frames_ids = np.sort(self.detections[DetectionDto.FRAME_ID].unique())[drop_num:]
        sequence_multi_length = unique_frames_ids.shape[0] - (unique_frames_ids.shape[0] % self.sequence_length)
        frames_chunks = np.reshape(unique_frames_ids[:sequence_multi_length], (-1, self.sequence_length))
        return frames_chunks

    def process_frames_chunk(self, frames_chunk: np.array) -> Optional[pd.DataFrame]:
        detections = self.detections[self.detections[DetectionDto.FRAME_ID].isin(frames_chunk)]
        persons_ids = detections[DetectionDto.PERSON_ID].unique()

        frames_chunk_sequences = []
        for person_id in persons_ids:
            single_person_detections = detections[detections[DetectionDto.PERSON_ID] == person_id]
            filled_detections = self.fill_detections_holes(single_person_detections, frames_chunk)
            if not filled_detections.empty:
                frames_chunk_sequences.append(filled_detections)

        if frames_chunk_sequences:
            return pd.concat(frames_chunk_sequences, axis=0)
        return None

    def chunks_with_sequences(self, drop_num=0):
        frames_chunks = self.get_frames_chunks(drop_num)
        sequences = []
        for frames_chunk in frames_chunks:
            processed_frames = self.process_frames_chunk(frames_chunk)
            if processed_frames is not None and not processed_frames.empty:
                sequences.append(processed_frames)
        np.random.shuffle(sequences)
        return sequences

    def generate(self, klass=None, classes_distribution=None) -> pd.DataFrame:
        drops = [0, 32, 16, 8, 4, 2, 1]
        all_sequences = []
        seq_multiplier = 1
        if klass and classes_distribution:
            mid_klass = classes_distribution.most_common()[-1][1]
            seq_multiplier *= math.ceil(mid_klass / classes_distribution[klass])

        for drop in drops:
            all_sequences += self.chunks_with_sequences(drop)

            if not all_sequences:
                return pd.DataFrame(columns=SequenceDto.get_cols())
            if all_sequences and len(all_sequences) == 3 * seq_multiplier:
                return pd.concat(all_sequences, axis=0)
            if all_sequences and len(all_sequences) > 3 * seq_multiplier:
                return pd.concat(all_sequences[:3 * seq_multiplier], axis=0)

        return pd.concat(all_sequences, axis=0)
