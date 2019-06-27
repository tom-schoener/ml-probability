from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import os
import pathlib
import math

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import commons as cm

tfk = tf.keras
tfkl = tfk.layers
tfd = tfp.distributions
tfpl = tfp.layers

print("Tensorflow Version: %s" % tf.__version__)
print("Tensorflow Probability Version: %s" % tfp.__version__)

if tf.test.gpu_device_name() != '/device:GPU:0':
    print('GPU device not found. Using CPU')
else:
    print('Found GPU: {}'.format(tf.test.gpu_device_name()))


class Model(metaclass=ABCMeta):
    """
    Base model class.
    """

    def __init__(self, model_setup, validation_split=0.05, models_dir="./models/", history_dir="./history/", embedding_dim=50):
        self.setup = model_setup
        self.models_dir = models_dir
        self.history_dir = history_dir
        self.validation_split = validation_split
        self.N = self.calc_N(len(self.setup["train"][0]), validation_split)

        pathlib.Path(history_dir).mkdir(
            parents=True, exist_ok=True)
        pathlib.Path(models_dir).mkdir(
            parents=True, exist_ok=True)

    def load_history(self, N=None):
        (history_df, last_epoch) = cm.load_history_from_file(
            self.get_history_save_file())
        return (history_df, last_epoch)

    def load_model(self):
        try:
            if not self.save_weights_only():
                model = tfk.models.load_model(self.get_model_save_file())
            else:
                # https://github.com/tensorflow/probability/issues/325
                # model = tfk.models.load_model(model_save_file)
                model = self.keras_model()
                model.load_weights(self.get_model_save_file())
            print("using saved model")
        except IOError:
            model = self.keras_model()
            print("model has not been trained (IOError)")

        return model

    def fit(self, epochs, batch_size):
        (x_train, x_train_padded, y_train) = self.setup["train"]

        model = self.load_model()
        last_epoch = self.load_history()[1]

        model.fit(x_train_padded,
                  y_train,
                  validation_split=self.validation_split,
                  initial_epoch=last_epoch,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=cm.get_keras_callbacks(
                      model_save_file=self.get_model_save_file(),
                      history_save_file=self.get_history_save_file(),
                      weights_only=self.save_weights_only()))

    def keras_summary(self):
        return self.keras_model().summary()

    def plot_history_metrics(self, metrics=["acc", "loss", "recall", "precision", "kl"]):
        for metric in metrics:
            cm.plot_metric(metric, history_df)
            plt.show()

    @staticmethod
    def get_history_save_file_static(history_dir, model_id):
        return os.path.abspath(os.path.join(history_dir, model_id + ".csv"))

    def get_history_save_file(self):
        return self.get_history_save_file_static(self.history_dir, self.model_id())

    def get_model_save_file(self):
        return os.path.abspath(os.path.join(self.models_dir, self.model_id() + ".h5"))

    @property
    def embedding_layer(self):
        return self.setup["embedding_layer"]

    @property
    def embedding_input_dim(self):
        return self.setup["embedding_input_dim"]

    @property
    def word_index(self):
        return self.setup["word_index"]

    @abstractmethod
    def keras_model(self):
        pass

    @abstractmethod
    def model_id(self, N=None):
        pass

    @abstractmethod
    def readable_name(self):
        pass

    def is_variational(self):
        return False

    def save_weights_only(self):
        return False

    @staticmethod
    def calc_N(training_size, validation_split):
        return math.floor(training_size * (1.0 - validation_split))


class DefaultDenseModel(Model):
    def __init__(self, neurons_hidden_layers, *args, **kwargs):
        self.neurons_hidden_layers = neurons_hidden_layers
        super(DefaultDenseModel, self).__init__(*args, **kwargs)

    def keras_model(self):
        model = tfk.Sequential(name=self.model_id())
        model.add(self.embedding_layer)
        for neurons in self.neurons_hidden_layers:
            model.add(tfkl.Dense(
                neurons, activation='relu', kernel_regularizer=tfk.regularizers.l2(0.01)))
        model.add(tfkl.Flatten())
        model.add(tfkl.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=cm.metrics)

        return model

    @staticmethod
    def model_id_static(N, neurons_hidden_layers):
        return "default_dense_%d_%s" % (N, "_".join(map(str, neurons_hidden_layers)))

    def model_id(self, N=None):
        if N is None:
            N = self.N
        return self.model_id_static(N, self.neurons_hidden_layers)

    def save_weights_only(self):
        return False

    def readable_name(self):
        return "Dense NN"


class DefaultConvModel(Model):
    def __init__(self, *args, **kwargs):
        super(DefaultConvModel, self).__init__(*args, **kwargs)

    def keras_model(self):
        model = tfk.Sequential([
            self.embedding_layer,
            tfkl.Dropout(0.25),
            tfkl.Conv1D(64, 15, activation="relu"),
            tfkl.Dropout(0.25),
            tfkl.Conv1D(64, 5, activation="relu"),
            tfkl.MaxPooling1D(10),
            tfkl.Dropout(0.25),
            tfkl.Flatten(),
            tfkl.Dropout(0.25),
            tfkl.Dense(64, activation='relu'),
            tfkl.Dropout(0.25),
            tfkl.Dense(1, activation='sigmoid')
        ], name=self.model_id())
        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=cm.metrics)
        return model

    @staticmethod
    def model_id_static(N):
        return "default_conv_%d" % N

    def model_id(self, N=None):
        if N is None:
            N = self.N
        return self.model_id_static(N)

    def save_weights_only(self):
        return False

    def readable_name(self):
        return "Conv NN"


class McDropoutModel(Model):
    def __init__(self, neurons_hidden_layers, tau=1.0, lengthscale=1e-2, dropout=0.5, *args, **kwargs):
        self.neurons_hidden_layers = neurons_hidden_layers
        self.tau = tau
        self.lengthscale = lengthscale
        self.dropout = dropout
        super(McDropoutModel, self).__init__(*args, **kwargs)

    def keras_model(self):
        reg = self.lengthscale**2 * \
            (1 - self.dropout) / (2. * self.N * self.tau)

        # TODO: test input_shape
        inputs = tfk.Input(
            shape=(self.embedding_input_dim,), dtype='int32')

        inter = self.embedding_layer(inputs)

        for neurons in self.neurons_hidden_layers:
            inter = tfkl.Dropout(self.dropout)(inter, training=True)
            inter = tfkl.Dense(neurons, activation='relu',
                               kernel_regularizer=tfk.regularizers.l2(reg))(inter)

        inter = tfkl.Dropout(self.dropout)(inter, training=True)
        inter = tfkl.Flatten()(inter)
        outputs = tfkl.Dense(1,
                             kernel_regularizer=tfk.regularizers.l2(reg),
                             activation="sigmoid")(inter)
        model = tfk.Model(inputs, outputs, name=self.model_id())

        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=cm.metrics)
        return model

    @staticmethod
    def model_id_static(N, neurons_hidden_layers):
        return "mc_dropout_%d_%s" % (N, "_".join(map(str, neurons_hidden_layers)))

    def model_id(self, N=None):
        if N is None:
            N = self.N
        return self.model_id_static(N, self.neurons_hidden_layers)

    def save_weights_only(self):
        return False

    def is_variational(self):
        return True

    def readable_name(self):
        return "MC Dropout"


class BayesByBackpropModel(Model):
    def __init__(self, neurons_hidden_layers, variational_layer=tfpl.DenseReparameterization, *args, **kwargs):
        self.neurons_hidden_layers = neurons_hidden_layers
        self.variational_layer = variational_layer
        super(BayesByBackpropModel, self).__init__(*args, **kwargs)

    def keras_model(self):
        def kernel_divergence_fn(q, p, _):
            return tfd.kl_divergence(q, p) / tf.cast(self.N, tf.float32)

        model = tfk.Sequential(name=self.model_id())

        model.add(self.embedding_layer)
        for neurons in self.neurons_hidden_layers:
            model.add(self.variational_layer(neurons,
                                             activation='relu',
                                             kernel_divergence_fn=kernel_divergence_fn))
        model.add(tfkl.Flatten())
        model.add(self.variational_layer(1, activation="sigmoid",
                                         kernel_divergence_fn=kernel_divergence_fn))

        model.compile(optimizer=tfk.optimizers.Adam(0.001),
                      loss=tfk.losses.binary_crossentropy,
                      metrics=cm.metrics)

        return model

    @staticmethod
    def model_id_static(N, neurons_hidden_layers, variational_layer):
        return "bayes_by_backprop_%s_%d_%s" % (variational_layer.__name__, N, "_".join(map(str, neurons_hidden_layers)))

    def model_id(self, N=None):
        if N is None:
            N = self.N
        return self.model_id_static(N, self.neurons_hidden_layers, self.variational_layer)

    def save_weights_only(self):
        return True

    def is_variational(self):
        return True

    def readable_name(self):
        return "Bayes By Backprop (%s)" % self.variational_layer.__name__.replace("Dense", "")
