import tensorflow as tf
import pickle as pkl
import cv2
import numpy as np
import pandas as pd

from preprocessing.dto.detection_dto import DetectionDto


class MobilenetService:
    def __init__(self):
        self.mobilenet_model = tf.keras.applications.MobileNetV3Large(
            input_shape=(224, 224, 3),
            alpha=1.0,
            minimalistic=False,
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            classes=1000,
            pooling='avg',
            dropout_rate=0.2,
            include_preprocessing=True,
        )
        with open('/home/rafa/PycharmProjects/sdm2-ready-code/preprocessing/pca.pkl', 'rb') as file:
            self.pca = pkl.load(file)

    @staticmethod
    def get_roi(vid_as_arr, frame, bbox):
        roi = vid_as_arr[frame][int(bbox[1]): int(bbox[1] + bbox[3]), int(bbox[0]): int(bbox[0] + bbox[2])]
        if roi.size == 0:
            return np.zeros((224, 224, 3))

        roi = cv2.resize(roi, (224, 224))
        return np.array(roi)

    @staticmethod
    def get_frame_roi(frame):
        if frame.size == 0:
            return np.zeros((224, 224, 3))

        roi = cv2.resize(frame, (224, 224))
        return np.array(roi)

    def get_single_person_visual_features(self, vid_as_array, original_df):
        df_splitted = np.array_split(original_df, np.ceil(original_df.shape[0]/1000))
        processed_dfs = []
        for df in df_splitted:
            df['visual_features'] = df.apply(lambda row: self.get_roi(vid_as_array, row.frame_id, row.bbox), axis=1)

            rois = np.array(df['visual_features'].to_list())
            preds = self.mobilenet_model.predict(rois, verbose=False)
            preds_list = [val for val in preds]
            pca_preds = self.pca.transform(preds_list)
            pca_preds_list = [val for val in pca_preds]
            df[DetectionDto.VISUAL_FEATURES] = preds_list
            df[DetectionDto.VISUAL_FEATURES_PCA] = pca_preds_list
            processed_dfs.append(df)

        return pd.concat(processed_dfs, axis=0)

    def get_frame_visual_features(self, vid_as_array, df):

        frames = df[DetectionDto.FRAME_ID].unique()
        filtered_vid = [vid_as_array[idx-1] for idx in frames]
        resized = np.array([self.get_frame_roi(single_frame) for single_frame in filtered_vid])

        resized_splitted = np.array_split(resized, np.ceil(resized.shape[0]/1000))
        preds_list = []
        preds_red_list = []
        for chunk in resized_splitted:
            preds = self.mobilenet_model.predict(chunk, verbose=False)
            preds_red = self.pca.transform(preds)
            preds_list += [val for val in preds]
            preds_red_list += [val for val in preds_red]
        preds_df = pd.DataFrame(
            {'right_frame_id': frames, 'frame_vis_features': preds_list, 'frame_vis_features_pca': preds_red_list})
        ext_df = df.merge(preds_df, 'inner', left_on='frame_id', right_on='right_frame_id')
        ext_df = ext_df.drop('right_frame_id', axis=1)
        return ext_df


class TooMuchDataException(Exception):
    pass
