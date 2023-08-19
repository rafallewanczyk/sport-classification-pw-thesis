from model.abc_model import ABCModel
from tensorflow.keras.layers import Input, LSTM, concatenate, Dense, Dropout, Reshape, MaxPooling2D, BatchNormalization, \
    Concatenate, Bidirectional

from preprocessing.dto.sequence_dto import SequenceDto


class Sp(ABCModel):
    # 54%
    SP_VISUAL_FEATURES_CHANNELS = 256
    PARALLEL_LSTM_CHANNELS = 128
    JOINED_LSTM_CHANNELS = 64
    DENSE_NEURONS = 128

    def __init__(self):
        super().__init__()

    def get_model_name(self):
        return 'sp'

    @staticmethod
    def get_required_features_names():
        return [SequenceDto.VISUAL_FEATURES_PCA]

    @classmethod
    def get_inputs(cls):
        return [Input(shape=(cls.TIME_STEPS, cls.SP_VISUAL_FEATURES_CHANNELS)) for _ in
                range(cls.PERSON_LIMIT)]

    @classmethod
    def get_body(cls, inputs):
        sp_hierarchical = [BatchNormalization()(inp) for inp in inputs]

        branches = []
        a = LSTM(cls.PARALLEL_LSTM_CHANNELS, input_shape=(cls.TIME_STEPS, cls.SP_VISUAL_FEATURES_CHANNELS),
                 return_sequences=True,
                 stateful=False)
        b = Dropout(0.2)
        c = LSTM(cls.PARALLEL_LSTM_CHANNELS, return_sequences=True, stateful=False)
        d = Dropout(0.2)
        for inp in sp_hierarchical:
            branch = a(inp)
            branch = b(branch)
            branch = c(branch)
            branch = d(branch)
            branches.append(branch)

        sp_hierarchical = Concatenate()(branches)
        sp_hierarchical = Reshape((cls.TIME_STEPS, cls.PERSON_LIMIT, cls.PARALLEL_LSTM_CHANNELS))(sp_hierarchical)
        sp_hierarchical = MaxPooling2D(pool_size=(1, cls.PERSON_LIMIT), strides=(1, 1), padding='valid')(
            sp_hierarchical)
        sp_hierarchical = Reshape((cls.TIME_STEPS, cls.PARALLEL_LSTM_CHANNELS))(sp_hierarchical)
        sp_hierarchical = Bidirectional(
            LSTM(cls.JOINED_LSTM_CHANNELS, go_backwards=True),
            backward_layer=LSTM(cls.JOINED_LSTM_CHANNELS)
        )(sp_hierarchical)
        sp_hierarchical = Dropout(0.2)(sp_hierarchical)
        sp_hierarchical = Dense(cls.DENSE_NEURONS, 'relu')(sp_hierarchical)
        return sp_hierarchical

    @classmethod
    def get_head(cls, body):
        return Dense(cls.CLASSES_NUMBER, 'softmax')(body)
