from model.abc_model import ABCModel
from tensorflow.keras.layers import Input, LSTM, concatenate, Dense, Dropout, Reshape, MaxPooling2D, BatchNormalization, \
    Concatenate, Bidirectional

from preprocessing.dto.sequence_dto import SequenceDto


class Frame(ABCModel):
    # acc 73%
    LSTM_CHANNELS = 128
    FRAME_VISUAL_FEATURES_CHANNELS = 256
    DENSE_NEURONS = 96

    def __init__(self):
        super().__init__()

    def get_model_name(self):
        return 'frame'

    @staticmethod
    def get_required_features_names():
        return [SequenceDto.FRAME_VIS_FEATURES_RED]

    @classmethod
    def get_inputs(cls):
        return [Input(shape=(cls.TIME_STEPS, cls.FRAME_VISUAL_FEATURES_CHANNELS))]

    @classmethod
    def get_body(cls, inputs):
        start = BatchNormalization()(inputs[0])
        body = Bidirectional(LSTM(cls.LSTM_CHANNELS, go_backwards=True,
                                  input_shape=(cls.TIME_STEPS, cls.FRAME_VISUAL_FEATURES_CHANNELS)),
                             backward_layer=LSTM(cls.LSTM_CHANNELS, return_sequences=False),
                             )(start)
        # body = LSTM(cls.LSTM_CHANNELS, return_sequences=True)(start)
        # body = LSTM(cls.LSTM_CHANNELS )(body)
        body = Dropout(0.2)(body)
        body = Dense(cls.DENSE_NEURONS, 'relu')(body)
        return body

    @classmethod
    def get_head(cls, body):
        body = Dense(128, 'relu')(body)
        return Dense(cls.CLASSES_NUMBER, 'softmax')(body)

# m = Frame(12)
# m.compile_model()
