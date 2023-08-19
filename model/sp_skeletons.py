from model.abc_model import ABCModel
from model.sp import Sp
from model.skeletons import Skeletons
from tensorflow.keras.layers import Input, LSTM, concatenate, Dense, Dropout, Reshape, MaxPooling2D, BatchNormalization, \
    Concatenate, GlobalAveragePooling2D

from preprocessing.dto.sequence_dto import SequenceDto

class SpSkeletons(ABCModel):
    DENSE_NEURONS = 256

    def __init__(self):
        super().__init__()

    def get_model_name(self):
        return 'sp_skeletons'

    @staticmethod
    def get_required_features_names():
        return [SequenceDto.VISUAL_FEATURES_PCA, SequenceDto.SKELETON_NORMALIZED]

    @classmethod
    def get_inputs(cls):
        return Sp.get_inputs() + Skeletons.get_inputs()

    @classmethod
    def get_body(cls, inputs):
        sp_hierarchical = Sp.get_body(inputs[:cls.PERSON_LIMIT])
        skeletons = Skeletons.get_body(inputs[cls.PERSON_LIMIT:])

        return [Sp.get_head(sp_hierarchical), Skeletons.get_head(skeletons)]

    @classmethod
    def get_head(cls, body):
        classifier_head = Concatenate()(body)
        classifier_head = Dense(cls.DENSE_NEURONS, 'relu')(classifier_head)
        classifier_head = Dense(cls.CLASSES_NUMBER, 'softmax')(classifier_head)
        return classifier_head

# m = VisFeaturesWithFrame(5)
# m.get_model_body()
