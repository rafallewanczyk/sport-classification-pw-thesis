from model.abc_model import ABCModel
from tensorflow.keras.layers import Input, LSTM, concatenate, Dense, Dropout, Reshape, MaxPooling2D, BatchNormalization, \
    Concatenate, Bidirectional

from preprocessing.dto.sequence_dto import SequenceDto


class Skconcated(ABCModel):
    # 35 %
    SK_POINTS = 34
    PARALLEL_LSTM_CHANNELS = 32
    JOINED_LSTM_CHANNELS = 64
    DENSE_NEURONS = 96

    def __init__(self):
        super().__init__()

    def get_model_name(self):
        return 'skconcated'

    @classmethod
    def get_inputs(cls):
        return [Input(shape=(cls.TIME_STEPS, cls.SK_POINTS)) for _ in range(cls.PERSON_LIMIT)]

    @staticmethod
    def get_required_features_names():
        return [SequenceDto.SKELETON_NORMALIZED]

    @classmethod
    def get_body(cls, inputs):
        sp_hierarchical = [BatchNormalization()(inp) for inp in inputs]

        bi = Bidirectional(
            LSTM(cls.PARALLEL_LSTM_CHANNELS, input_shape=(cls.TIME_STEPS, cls.SK_POINTS),
                 go_backwards=True),
            backward_layer=LSTM(cls.PARALLEL_LSTM_CHANNELS)
        )
        branches = []
        for inp in sp_hierarchical:
            branch = bi(inp)
            branches.append(branch)

        sp_hierarchical = Concatenate()(branches)
        sp_hierarchical = Dense(cls.DENSE_NEURONS, 'relu')(sp_hierarchical)
        return sp_hierarchical

    @classmethod
    def get_head(cls, body):
        return Dense(cls.CLASSES_NUMBER, 'softmax')(body)
