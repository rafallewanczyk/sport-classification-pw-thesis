import abc
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from collections import Counter

from preprocessing.video import Video


@dataclass
class Label:
    name: str


class AbcDataset(abc.ABC):

    def __init__(self, dataset_path: str = '/home/rafa/SVW', videos_extension: str = 'mp4',
                 stats_file: str = "SVW.csv", classes: Optional[List[Label]] = None):
        np.random.seed(1)

        self.dataset_path = Path(dataset_path)
        self.videos_extension = videos_extension
        self.videos = []
        self.classes = classes or self.get_all_classes()
        self.dataset_stats = pd.read_csv(self.dataset_path / stats_file)

        self.class_name_to_id, self.class_id_to_name = self._generate_classes_dict()

    @classmethod
    @abc.abstractmethod
    def get_all_classes(cls) -> List[Label]:
        ...

    def _generate_classes_dict(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        name_to_id = {}
        id_to_name = {}
        for idx, label in enumerate(self.classes):
            name_to_id[label.name] = idx
            id_to_name[idx] = label.name
        return name_to_id, id_to_name

    def translate_class(self, klass: Union[int, str]):
        if np.issubdtype(type(klass), int):
            return self.class_id_to_name[klass]
        if np.issubdtype(type(klass), str):
            return self.class_name_to_id[klass]

        raise TypeError(f"unknown class type {type(klass)}")

    def get_classes_names(self):
        return [klass.name for klass in self.classes]

    def _filter_classes(self, videos_paths):
        if self.classes is not None:
            return filter(lambda vid: any([klass in vid.as_posix() for klass in self.get_classes_names()]),
                          videos_paths)
        return videos_paths

    def _get_filtered_videos(self):
        paths = list(self._filter_classes(filter(lambda vid: 'reshaped' not in vid.stem,
                                                 self.dataset_path.rglob(f'*.{self.videos_extension}'))))
        np.random.shuffle(paths)
        return paths

    def preprocess_videos(self, mobilenet_service):
        loaded_videos_number = 0
        shuffled_paths = self._get_filtered_videos()
        np.random.shuffle(shuffled_paths)
        progress = tqdm(shuffled_paths)
        for video_path in progress:
            try:
                progress.set_postfix({'current video': video_path, 'loaded videos': loaded_videos_number})
                vid = Video(video_path, allow_processing_pipeline=True)
                if vid.extended_detections_path.exists():
                    continue
                vid.build_detections()
                vid.extend_detections(mobilenet_service)
                loaded_videos_number += 1
            except Exception as exc:
                continue
        return self

    @staticmethod
    def generate_features_single_video(path, classes_distribution):
        with suppress(FileNotFoundError):
            Video(path, False).build_sequences(use_dump=False, classes_distribution=classes_distribution)

    @staticmethod
    def generate_xy_single_video(path, translate_class, max_person_number, feature_types):
        with suppress(FileNotFoundError):
            Video(path, False).build_x_y(translate_class=translate_class,
                                         features_types=feature_types,
                                         max_person_number=max_person_number,
                                         use_dump=False)

    def generate_features(self):
        videos = self._get_filtered_videos()
        classes_distribution = Counter([vid.parent.stem for vid in videos])
        get_features_partial = partial(self.generate_features_single_video, classes_distribution=classes_distribution)
        process_map(get_features_partial, videos, max_workers=8, chunksize=1)

    def get_as_x_y(self, feature_types: List[str], max_person_number=6,
                   use_full_dump: bool = True, use_videos_dump: bool = True, split_id=1):

        x_y_path = self.dataset_path / f'training_data/{"_".join(feature_types)}_{split_id}_x_y.pkl'

        if use_full_dump and x_y_path.exists():
            return pd.read_pickle(x_y_path)

        loaded_x_y_number = 0
        videos_paths = list(self._filter_classes(self._get_filtered_videos()))

        if not use_videos_dump:
            xy_gen = partial(self.generate_xy_single_video, translate_class=self.translate_class,
                             max_person_number=max_person_number,
                             feature_types=feature_types)
            process_map(xy_gen, videos_paths, chunksize=1)

        progress = tqdm(videos_paths)

        histogram_train, histogram_test, histogram_val = {}, {}, {}
        x_train, y_train, x_test, y_test, x_val, y_val = [], [], [], [], [], []
        videos_train, videos_test, videos_val = [], [], []
        mode = ""
        for idx, video_path in enumerate(progress):
            if self.dataset_stats[self.dataset_stats.FileName == video_path.name][f'Train {split_id}?'].values[0]:
                x = x_train
                y = y_train
                videos = videos_train
                histogram = histogram_train
                mode = "train"
            else:
                x = x_test
                y = y_test
                videos = videos_test
                histogram = histogram_test
                mode = "test"
                if np.random.rand() < 0.3:
                    x = x_val
                    y = y_val
                    videos = videos_val
                    histogram = histogram_val
                    mode = "val"
            with suppress(FileNotFoundError):
                progress.set_postfix({'current video': video_path, 'loaded videos': loaded_x_y_number})
                vid = Video(video_path, allow_processing_pipeline=False)
                tmp_x, tmp_y = vid.build_x_y(self.translate_class, max_person_number=max_person_number,
                                             features_types=feature_types, use_dump=True)
                if len(tmp_x) > 0:
                    x.append(tmp_x)
                    y.append(tmp_y)
                    videos += [vid.video_path.as_posix()] * tmp_x[0].shape[0]
                    if vid.klass in histogram:
                        histogram[vid.klass] += np.array([tmp_x[0].shape[0], 1])
                    else:
                        histogram[vid.klass] = np.array([tmp_x[0].shape[0], 1])

                loaded_x_y_number += 1

        y_test_array = np.concatenate(y_test, axis=0).reshape((-1, 1))
        y_train_array = np.concatenate(y_train, axis=0).reshape((-1, 1))
        y_val_array = np.concatenate(y_val, axis=0).reshape((-1, 1))

        x_train_arrays = []
        for channel in tqdm(zip(*x_train)):
            x_train_arrays.append(np.concatenate(channel, axis=0))
        x_test_arrays = []
        for channel in tqdm(zip(*x_test)):
            x_test_arrays.append(np.concatenate(channel, axis=0))
        x_val_arrays = []
        for channel in tqdm(zip(*x_val)):
            x_val_arrays.append(np.concatenate(channel, axis=0))

        all_sets = [x_train_arrays, y_train_array, x_test_arrays, y_test_array, x_val_arrays, y_val_array, videos_train,
                    videos_test, videos_val, histogram_train, histogram_test, histogram_val]
        return all_sets
