from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Reshape, MaxPooling2D, BatchNormalization, \
    Concatenate

from model.abc_model import ABCModel
from preprocessing.dto.sequence_dto import SequenceDto


class Skeletons(ABCModel):
    # 31 %
    SP_SKELETON_POINTS = 34
    PARALLEL_LSTM_CHANNELS = 96
    JOINED_LSTM_CHANNELS = 64
    DENSE_NEURONS = 64

    def __init__(self, classes_number=4):
        self.classes_number = classes_number
        super().__init__()

    def get_model_name(self):
        return 'skeletons'

    @staticmethod
    def get_required_features_names():
        return [SequenceDto.SKELETON_NORMALIZED]

    @classmethod
    def get_inputs(cls):
        return [Input(shape=(cls.TIME_STEPS, cls.SP_SKELETON_POINTS)) for _ in range(cls.PERSON_LIMIT)]

    @classmethod
    def get_body(cls, inputs):
        skeletons = [BatchNormalization()(inp) for inp in inputs]

        branches = []
        a = LSTM(cls.PARALLEL_LSTM_CHANNELS, input_shape=(cls.TIME_STEPS, cls.SP_SKELETON_POINTS),
                 return_sequences=True,
                 stateful=False)
        b = Dropout(0.2)
        c = LSTM(cls.PARALLEL_LSTM_CHANNELS, return_sequences=True, stateful=False)

        for inp in skeletons:
            branch = a(inp)
            branch = b(branch)
            branch = c(branch)
            branches.append(branch)

        skeletons = Concatenate()(branches)
        skeletons = Reshape((cls.TIME_STEPS, cls.PERSON_LIMIT, cls.PARALLEL_LSTM_CHANNELS))(skeletons)
        skeletons = MaxPooling2D(pool_size=(1, cls.PERSON_LIMIT), strides=(1, 1), padding='valid')(skeletons)
        skeletons = Reshape((cls.TIME_STEPS, cls.PARALLEL_LSTM_CHANNELS))(skeletons)
        skeletons = LSTM(cls.JOINED_LSTM_CHANNELS, input_shape=(cls.TIME_STEPS, cls.PARALLEL_LSTM_CHANNELS),
                         return_sequences=True,
                         stateful=False)(skeletons)
        skeletons = Dropout(0.2)(skeletons)
        skeletons = LSTM(cls.JOINED_LSTM_CHANNELS, return_sequences=True, stateful=False)(skeletons)
        skeletons = Dropout(0.2)(skeletons)
        skeletons = LSTM(cls.JOINED_LSTM_CHANNELS, return_sequences=False, stateful=False)(skeletons)
        skeletons = Dense(cls.DENSE_NEURONS)(skeletons)
        return skeletons

    @classmethod
    def get_head(cls, body):
        skeletons = Dense(cls.CLASSES_NUMBER, 'softmax')(body)
        return skeletons
