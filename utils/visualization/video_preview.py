import cv2
import numpy as np
from gluoncv import utils

from preprocessing.video import Video
from utils.color_generator import ColorGenerator


def preview_video(video: Video):
    cap = cv2.VideoCapture(video.video_path.as_posix())
    while (cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            break
        cv2.imshow('frame', img)
        cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


def preview_video_with_detections(video: Video):
    cap = cv2.VideoCapture(video.video_path.as_posix())
    color_generator = ColorGenerator()
    idx = 0
    while (cap.isOpened()):
        ret, img = cap.read()
        labels = video.get_detections_for_frame(idx)

        if not ret:
            print('breaking')
            break
        if not len(labels) == 0:

            bboxes = np.array([[label.x1, label.y1, label.x1 + label.w, label.y1 + label.h] for label in labels],
                              dtype=np.float64)
            scores = np.array([np.ones(bboxes.shape[0])[:, None]])
            classes = np.array([np.zeros(bboxes.shape[0])[:, None]])
            det_labels = np.array([label.person_id for label in labels])
            bboxes = np.array([bboxes[:]])
            colors = {label.person_id: color_generator.id_to_random_color(label.person_id, normalized=True) for label in
                      labels}
            pred_coords, confidence = np.zeros((det_labels.shape[0], 17, 2)), np.zeros((det_labels.shape[0], 17, 1))
            ax = utils.viz.cv_plot_keypoints(img, pred_coords, confidence,
                                             classes[:, ], bboxes[:, ], scores[:, ],
                                             box_thresh=0.5, keypoint_thresh=0.1,
                                             labels=det_labels, colors=colors)
        else:
            ax = img
        cv2.imshow('frame', ax)
        cv2.waitKey(0)
        idx += 1

    cap.release()
    cv2.destroyAllWindows()

