'''
A Moduele which binds Yolov7 repo with Deepsort with modifications
'''

import os
from typing import List

import pandas as pd

from preprocessing.inference_service.external_model.yolov7_pose_tracker.detections_dto import PoseDetection, TrackedPoseDetection

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # comment out below line to enable tensorflow logging outputs
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto  # type: ignore

# deep sort imports
from preprocessing.inference_service.external_model.yolov7_pose_tracker.deep_sort import preprocessing, nn_matching
from preprocessing.inference_service.external_model.yolov7_pose_tracker.deep_sort.detection import Detection
from preprocessing.inference_service.external_model.yolov7_pose_tracker.deep_sort.tracker import Tracker

# import from helpers
from preprocessing.inference_service.external_model.yolov7_pose_tracker.tracking_helpers import create_box_encoder
from preprocessing.inference_service.external_model.yolov7_pose_tracker.detection_helpers import *

# load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True


class Deepsort:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''

    def __init__(self, reID_model_path: str, max_cosine_distance: float = 0.4, nn_budget: float = None,
                 nms_max_overlap: float = 1.0):
        '''
        args: 
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        '''
        self.nms_max_overlap = nms_max_overlap

        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance,
                                                           nn_budget)  # calculate cosine distance metric
        self.tracker = Tracker(metric)  # initialize tracker

    def track_video(self, frame: np.array, pose_detections: List[PoseDetection]):
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output: path to output video
            skip_frames: Skip every nth frame. After saving the video, it'll have very visuals experience due to skipped frames
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            count_objects: count objects being tracked on screen
            verbose: print details on the screen allowed values 0,1,2
        '''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
        bboxes = np.array([det.bbox for det in pose_detections])
        scores = np.array([det.score for det in pose_detections])

        # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
        features = self.encoder(frame,
                                bboxes)  # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
        detections = [Detection(bbox, score, 'person', feature, pose_detection) for bbox, score, feature, pose_detection
                      in
                      zip(bboxes, scores, features,
                          pose_detections)]  # [No of BB per frame] deep_sort.detection.Detection object

        cmap = plt.get_cmap('tab20b')  # initialize color map
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        self.tracker.predict()  # Call the tracker
        self.tracker.update(detections)  # updtate using Kalman Gain

        tracked_detections = []
        for track in self.tracker.tracks:  # update new findings AKA tracks
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            last_detection = track.detections[-1]
            color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
            color = [i * 255 for i in color]
            tracked_detections.append(
                TrackedPoseDetection.from_pose_detection(last_detection.pose_detection, track.track_id, color))

            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
            #               (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
            # cv2.putText(frame, class_name + " : " + str(track.track_id), (int(bbox[0]), int(bbox[1] - 11)), 0, 0.6,
            #             (255, 255, 255), 1, lineType=cv2.LINE_AA)
        return tracked_detections

        # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
