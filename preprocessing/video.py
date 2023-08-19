import pickle
from itertools import chain
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from preprocessing.dto.sequence_dto import SequenceDto
from preprocessing.inference_service.external_model.mobilenet.mobilenet_service import MobilenetService
from preprocessing.sequences import Sequences


class Video:
    SEQUENCE_LENGTH = 64

    def __init__(self, video_path: Path, allow_processing_pipeline: bool = False):
        self.video_path = video_path
        self.klass = video_path.parent.stem
        self.allow_processing_pipeline = allow_processing_pipeline

        self.video_reshaped_path = self.get_reshaped_video_path(video_path)
        self.detections_path = self.get_detections_video_path(video_path)
        self.extended_detections_path = self.get_extended_detections_video_path(video_path)
        self.sequences_path = self.get_sequences_video_path(video_path)
        self.x_y_path = self.get_x_y_video_path(video_path)

    @classmethod
    def get_reshaped_video_path(cls, video_path):
        return cls._add_suffix_to_path(video_path, '_reshaped.mp4')

    @classmethod
    def get_detections_video_path(cls, video_path):
        return cls._add_suffix_to_path(video_path, '_detections.pkl')

    @classmethod
    def get_extended_detections_video_path(cls, video_path):
        return cls._add_suffix_to_path(video_path, '_detections_ext.pkl')

    @classmethod
    def get_sequences_video_path(cls, video_path):
        return cls._add_suffix_to_path(video_path, '_sequences.pkl')

    @classmethod
    def get_x_y_video_path(cls, video_path):
        return cls._add_suffix_to_path(video_path, '_x_y.pkl')

    @staticmethod
    def _add_suffix_to_path(path: Path, suffix: str):
        return path.parent.joinpath(path.stem + suffix)

    def get_as_array(self):
        import cv2
        cap = cv2.VideoCapture(self.video_reshaped_path.as_posix())
        video = [np.zeros((1080, 1920, 3))]
        while (cap.isOpened()):
            ret, img = cap.read()
            if not ret:
                break
            video.append(img)
        cap.release()
        return video

    def build_detections(self) -> pd.DataFrame:
        if not self.detections_path.exists() and self.allow_processing_pipeline:
            from preprocessing.inference_service.external_model.yolov7_pose_tracker.pose_track import run
            run(self.video_path.as_posix(), device='0')
        elif not self.detections_path.exists():
            raise FileNotFoundError("No detections file available")

        return pd.read_pickle(self.detections_path)

    def build_ext_detections(self)-> pd.DataFrame:
        return pd.read_pickle(self.extended_detections_path)

    def extend_detections(self, mobilenet_service: MobilenetService) -> pd.DataFrame:
        if not self.extended_detections_path.exists() and self.allow_processing_pipeline:
            detections = self.build_detections()
            detections_sp_ext = mobilenet_service.get_single_person_visual_features(self.get_as_array(), detections)
            detections_frame_ext = mobilenet_service.get_frame_visual_features(self.get_as_array(),detections_sp_ext )

            pd.to_pickle(detections_frame_ext , self.extended_detections_path)
        elif not self.extended_detections_path.exists():
            raise FileNotFoundError("No extended detections file available")

        return pd.read_pickle(self.extended_detections_path)

    def build_sequences(self, use_dump=True, classes_distribution=None) -> pd.DataFrame:
        detections = self.build_ext_detections()
        if detections.empty:
            return pd.DataFrame(columns=SequenceDto.get_cols())
        if not self.sequences_path.exists() or use_dump is False:
            sequences = Sequences(self.SEQUENCE_LENGTH, detections)
            sequences_df = sequences.generate(classes_distribution)
            sequences_df.to_pickle(self.sequences_path)
        elif not self.sequences_path.exists():
            raise FileNotFoundError("No sequences file available")
        return pd.read_pickle(self.sequences_path)

    def build_x_y(self, translate_class, features_types: List[str], max_person_number=6, use_dump=True) -> np.array:
        sequences = self.build_sequences()

        if self.x_y_path.exists() and use_dump:
            with self.x_y_path.open('rb') as file:
                return pickle.load(file)

        klass_translator = np.vectorize(translate_class)
        if sequences.empty:
            return np.array([]), np.array([])

        sequences = self._expand_sequences_df(sequences)

        x, y = [], []
        for feature in features_types:
            if feature in [SequenceDto.SKELETON_NORMALIZED, SequenceDto.SKELETON_ANGLES,
                           SequenceDto.VISUAL_FEATURES_PCA, SequenceDto.VISUAL_FEATURES]:
                x.append(self._single_person_features_to_x(sequences, feature, max_person_number))
            elif feature in [SequenceDto.FRAME_VIS_FEATURES, SequenceDto.FRAME_VIS_FEATURES_PCA]:
                x.append(self._global_features_to_x(sequences, feature))
            else:
                raise Exception(f'{feature} collection is not supported')

        assert any([val[0].shape[0] == x[0][0].shape[0] for val in x])

        x_inputs = list(chain(*x))
        y_array = klass_translator(np.array([self.klass] * x_inputs[0].shape[0]))[:, None]

        with self.x_y_path.open('wb') as file:
            pickle.dump([x_inputs, y_array], file)

        return x_inputs, y_array

    @staticmethod
    def _single_person_features_to_x(sequences, feature_name, max_person_number):
        x = []
        for group_idx in sequences['sequence_id'].unique():
            features = sequences[sequences['sequence_id'] == group_idx][
                [SequenceDto.PERSON_ID, feature_name, SequenceDto.MAX_WH]]
            features['area'] = features[SequenceDto.MAX_WH].apply(lambda val: val[0] * val[1])
            features_grouped = features.groupby([SequenceDto.PERSON_ID, 'area'])[
                feature_name].agg(
                lambda val: np.array(list(val))).reset_index().sort_values(by='area', ascending=False).reset_index()

            all_skeletons = (features_grouped[feature_name]
                             .agg(np.array).
                             to_list())[:max_person_number]
            all_skeletons += [np.zeros(all_skeletons[0].shape)] * (max_person_number - len(all_skeletons))
            x.append(np.array(all_skeletons))

        x = np.array(x)
        x_reshaped = list(map(lambda arr: arr.reshape(-1, Video.SEQUENCE_LENGTH, x[0].shape[-1]),
                              np.array_split(x, max_person_number, axis=1)))

        return x_reshaped

    @staticmethod
    def _global_features_to_x(sequences, feature_name):
        x = []
        for group_idx in sequences['sequence_id'].unique():
            features = sequences[sequences['sequence_id'] == group_idx]

            first_frame = features[SequenceDto.START_FRAME].min(axis=0)
            first_person = features[SequenceDto.PERSON_ID].min(axis=0)

            first_person_features = features[(features[SequenceDto.START_FRAME] == first_frame)
                                             & (features[SequenceDto.PERSON_ID] == first_person)][
                [SequenceDto.FRAME_ID, feature_name]]

            collected_features = np.array(first_person_features[feature_name].to_list())
            x.append(collected_features)
        return [np.array(x)]

    @staticmethod
    def _expand_sequences_df(sequences: pd.DataFrame):
        sequences['stem'] = sequences[SequenceDto.VIDEO].apply(lambda val: val.split('/')[-1])
        sequences = sequences.sort_values(by=['stem', SequenceDto.START_FRAME, SequenceDto.PERSON_ID],
                                          ascending=[True, True, True])
        sequences['sequence_id'] = sequences.groupby(
            by=[SequenceDto.VIDEO, SequenceDto.START_FRAME]).ngroup()
        skeleton_points = sorted([x for x in range(0, 51, 3)] + [x for x in range(1, 51, 3)])

        sequences[SequenceDto.SKELETON_NORMALIZED] = sequences[SequenceDto.SKELETON_NORMALIZED].apply(
            lambda val: val[skeleton_points])
        sequences[SequenceDto.SKELETON] = sequences[SequenceDto.SKELETON].apply(
            lambda val: val[skeleton_points])
        return sequences
