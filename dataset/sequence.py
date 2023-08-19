# from dataclasses import dataclass
# from typing import List
#
# import numpy as np
# import pandas as pd
#
#
# class Sequence:
#     def __init__(self, gt: pd.DataFrame, name_pattern: str, klass: str, person_features: dict):
#         self.gt = gt
#         self.name_pattern = name_pattern
#         self.klass = klass
#         self.person_features = person_features
#
#
# class ClusteredSequence(Sequence):
#     def __init__(self, gt: pd.DataFrame, name_pattern: str, klass: str, person_features: dict,
#                  features_as_array: np.array, reduced_features: np.array, cluster_labels: np.array):
#         self.cluster_labels = cluster_labels
#         self.features_as_array = features_as_array
#         self.reduced_features = reduced_features
#         super().__init__(gt, name_pattern, klass, person_features)
#
#
# class SkeletonClusteredSequence(ClusteredSequence):
#
#     def __init__(self, clustered_seq: ClusteredSequence, skeleton_seq):
#         self.skeleton_seq = skeleton_seq
#         super().__init__(clustered_seq.gt, clustered_seq.name_pattern, clustered_seq.klass,
#                          clustered_seq.person_features, clustered_seq.features_as_array, clustered_seq.reduced_features,
#                          clustered_seq.cluster_labels)
#
#
# @dataclass
# class SkeletonSeq:
#     name_pattern: str
#     start_frame: int
#     end_frame: int
#     skeletons_seq: List
#     confidence_seq: List
#     classes_seq: List
#     bboxes_seq: List
#     scores_seq: List
#     person_ids: List
