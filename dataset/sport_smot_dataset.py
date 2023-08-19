# from dataclasses import dataclass
# from pathlib import Path
#
# import pandas as pd
#
#
# class SportSmotDataset:
#     CLASSES = ['basketball', 'football', 'volleyball']
#
#     @dataclass
#     class RawVideo:
#         base_path: Path
#         name_pattern: str
#         gt_file: Path
#         klass: str
#
#         def load_pd(self):
#             gt = pd.read_csv(self.gt_file,
#                              names=['frame', 'person_id', 'x1', 'y1', 'w', 'h', 'temp1', 'temp2', 'temp3'],
#                              index_col=False)
#             gt = gt.drop(['temp1', 'temp2', 'temp3'], axis=1)
#             return gt
#
#     def __init__(self, dataset_path):
#         self.dataset_path = Path(dataset_path)
#         self.raw_videos = self._load_raw_videos()
#
#     def _load_raw_videos(self):
#         raw_videos = []
#         for klass in self.CLASSES:
#             for path in (self.dataset_path / klass).iterdir():
#                 raw_videos.append(SportSmotDataset.RawVideo(
#                     base_path=path,
#                     name_pattern=(path / 'img1/').as_posix() + '/{frame:06d}.jpg',
#                     gt_file=path / 'gt' / 'gt.txt',
#                     klass=klass
#                 ))
#         return raw_videos
