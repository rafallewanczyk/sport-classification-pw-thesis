from model.abc_model import ABCModel
from model.concated import Concated
from model.sp import Sp
from model.frame import Frame
from tensorflow.keras.layers import Input, LSTM, concatenate, Dense, Dropout, Reshape, MaxPooling2D, BatchNormalization, \
    Concatenate, GlobalAveragePooling2D

from preprocessing.dto.sequence_dto import SequenceDto


class SpConcatedFrame(ABCModel):
    DENSE_NEURONS = 64

    def __init__(self):
        super().__init__()

    def get_model_name(self):
        return 'sp_concated_frame'

    @staticmethod
    def get_required_features_names():
        return (Sp.get_required_features_names() +
                Concated.get_required_features_names() +
                Frame.get_required_features_names())

    @classmethod
    def get_inputs(cls):
        return Sp.get_inputs() + Concated.get_inputs() + Frame.get_inputs()

    @classmethod
    def get_body(cls, inputs):
        sp_hierarchical = Sp.get_body(inputs[:cls.PERSON_LIMIT])
        concated = Concated.get_body(inputs[cls.PERSON_LIMIT: -1])
        frame = Frame.get_body(inputs[-1:])

        return [sp_hierarchical, concated, frame]

    @classmethod
    def get_head(cls, body):
        classifier_head = Concatenate()(body)
        classifier_head = Dense(cls.DENSE_NEURONS, 'relu')(classifier_head)
        classifier_head = Dense(cls.CLASSES_NUMBER, 'softmax')(classifier_head)
        return classifier_head

# m = VisFeaturesWithFrame(5)
# m.get_model_body()
