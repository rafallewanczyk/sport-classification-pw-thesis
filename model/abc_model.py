from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

from preprocessing.video import Video


class ABCModel(ABC):
    PERSON_LIMIT = 6
    CLASSES_NUMBER = 27
    TIME_STEPS = Video.SEQUENCE_LENGTH

    def __init__(self):
        tf.keras.utils.set_random_seed(1)
        tf.get_logger().setLevel('ERROR')
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        self._model = self.compile_model()

    @staticmethod
    @abstractmethod
    def get_required_features_names():
        ...

    @staticmethod
    @abstractmethod
    def get_model_name():
        ...

    @classmethod
    @abstractmethod
    def get_inputs(cls):
        ...

    @classmethod
    @abstractmethod
    def get_body(cls, inputs):
        ...

    @classmethod
    @abstractmethod
    def get_head(cls, body):
        ...

    def get_model_path(self):
        return 'weights/' + self.get_model_name() + '.keras'

    def compile_model(self):
        inputs = self.get_inputs()
        body = self.get_body(inputs)
        head = self.get_head(body)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model = Model(inputs=inputs, outputs=head)
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()
        return model

    def train(self, epochs, x_train, y_train, x_test, y_test, **kwargs):
        assert self._model is not None, 'model is not compiled'
        history = self._model.fit(x=x_train, y=y_train, epochs=epochs, verbose=True, validation_data=(x_test, y_test),
                                  callbacks=[EarlyStopping(monitor='val_loss', patience=3)], **kwargs)

        self._model.save(self.get_model_path())
        return history

    def predict(self, x):
        predictions = self._model.predict(x)
        return predictions

    def evaluate_weighted_avg(self, x, videos_y, class_translator):
        keras.backend.clear_session()
        self._model = keras.models.load_model(self.get_model_path())

        predictions = self.predict(x)
        df = pd.DataFrame({'pred': list(predictions), 'gt': videos_y})
        grouped = df.groupby('gt')['pred'].apply(pd.Series.sum).to_frame('mode').reset_index(inplace=False)[
            ['gt', 'mode']]
        grouped['gt'] = grouped['gt'].apply(lambda val: val.split('/')[-2])
        grouped['mode'] = grouped['mode'].apply(lambda preds: np.argmax(preds, axis=0))
        grouped['mode'] = grouped['mode'].apply(class_translator)
        grouped['is_eq'] = grouped['gt'] == grouped['mode']
        return grouped['is_eq'].mean(), grouped['mode'].to_numpy(), grouped['gt'].to_numpy()

    def evaluate_voting(self, x, videos_y, class_translator):
        keras.backend.clear_session()
        self._model = keras.models.load_model(self.get_model_path())

        predictions = self.predict(x)
        maxes = [np.argmax(preds, 0) for preds in predictions]
        df = pd.DataFrame({'pred': list(maxes), 'gt': videos_y})
        grouped = df.groupby('gt')['pred'].apply(pd.Series.mode).to_frame('mode').reset_index(inplace=False)[
            ['gt', 'mode']]
        grouped['gt'] = grouped['gt'].apply(lambda val: val.split('/')[-2])
        grouped['mode'] = grouped['mode'].apply(class_translator)
        grouped['is_eq'] = grouped['gt'] == grouped['mode']
        return grouped['is_eq'].mean(), grouped['mode'].to_numpy(), grouped['gt'].to_numpy()

    @staticmethod
    def plot_history(history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.plot(epochs, loss, 'ro', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()

        epochs = range(1, len(acc) + 1)

        fig = plt.figure(figsize=(8, 8))
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'ro', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
