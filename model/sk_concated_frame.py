from model.abc_model import ABCModel
from model.concated import Concated
from model.frame import Frame
from tensorflow.keras.layers import Input, LSTM, concatenate, Dense, Dropout, Reshape, MaxPooling2D, BatchNormalization, \
    Concatenate, GlobalAveragePooling2D

from model.skconcated import Skconcated
from preprocessing.dto.sequence_dto import SequenceDto


class SkConcatedFrame(ABCModel):
    #78
    DENSE_NEURONS = 128

    def __init__(self):
        super().__init__()

    def get_model_name(self):
        return 'sk_concated_frame'

    @staticmethod
    def get_required_features_names():
        return Skconcated.get_required_features_names() + Concated.get_required_features_names() + Frame.get_required_features_names()

    @classmethod
    def get_inputs(cls):
        return Skconcated.get_inputs() + Concated.get_inputs() + Frame.get_inputs()

    @classmethod
    def get_body(cls, inputs):
        skconcated = Skconcated.get_body(inputs[:cls.PERSON_LIMIT])
        concated = Concated.get_body(inputs[cls.PERSON_LIMIT:-1])
        frame_linear = Frame.get_body(inputs[-1:])

        # return [Concated.get_head(concated), Frame.get_head(frame_linear)]
        return [skconcated, concated, frame_linear]

    @classmethod
    def get_head(cls, body):
        classifier_head = Concatenate()(body)
        classifier_head = Dense(cls.DENSE_NEURONS, 'relu')(classifier_head)
        classifier_head = Dropout(0.2)(classifier_head)
        classifier_head = Dense(cls.CLASSES_NUMBER, 'softmax')(classifier_head)
        return classifier_head

# m = VisFeaturesWithFrame(5)
# m.get_model_body()
