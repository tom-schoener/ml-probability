from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import os

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

    def __init__(self, glove_dir, models_dir="./models/", history_dir="./history/", embedding_dim=50):
        self.models_dir = models_dir
        self.history_dir = history_dir

        cm.one_time_setup()
        self.setup = cm.setup(glove_dir, embedding_dim=embedding_dim)

    def load_history(self):
        (history_df, last_epoch) = cm.load_history_from_file(
            self.get_history_save_file())
        return (history_df, last_epoch)

    def load_model(self, weight_only=False):
        try:
            if not weight_only:
                model = tfk.models.load_model(self.get_model_save_file())
            else:
                # https://github.com/tensorflow/probability/issues/325
                # model = tfk.models.load_model(model_save_file)
                model = self.keras_model()
                model.load_weights(model_save_file)
            print("using saved model")
        except IOError:
            model = self.keras_model()
            print("model has not been trained")

        return model

    # TODO
    def fit(self, epochs, batch_size, validation_split):
        (x_train, x_train_padded, y_train) = self.setup["train"]

        model = self.load_model()
        last_epoch = self.load_history()[1]

        model.fit(x_train_padded,
                  y_train,
                  validation_split=validation_split,
                  initial_epoch=last_epoch,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=cm.get_keras_callbacks(
                      model_save_file=self.get_model_save_file(),
                      history_save_file=self.get_history_save_file()))

    def get_history_save_file(self):
        return os.path.abspath(os.path.join(self.history_dir, self.get_model_id() + ".csv"))

    def get_model_save_file(self):
        return os.path.abspath(os.path.join(self.models_dir, self.get_model_id() + ".h5"))

    @abstractmethod
    def keras_model(self):
        pass

    @abstractmethod
    def get_model_id(self):
        pass


class DefaultDenseModel(Model):
    def __init__(self, neurons_hidden_layers, glove_dir, *args, **kwargs):
        super(DefaultDenseModel, self).__init__(*args, **kwargs)
        self.neurons_hidden_layers = neurons_hidden_layers

    def keras_model(self):
        model = tfk.Sequential(name=self.model_name)
        model.add(embedding_layer)
        for neurons in range(self.neurons_hidden_layers):
            model.add(tfkl.Dense(
                neurons, activation='relu', kernel_regularizer=tfk.regularizers.l2(0.01)))
        model.add(tfkl.Flatten())
        model.add(tfkl.Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=cm.metrics)

        return model


class DefaultConvModel(Model):
    def __init__(self):
        pass


class McDropoutModel(Model):
    def __init__(self):
        pass


class BayesByBackpropModel(Model):
    def __init__(self):
        pass


# TODO: scale N by (1 - validation_split)
def create_default_conv_model(self, embedding_layer, model_name="conv"):
    model = tfk.Sequential([
        embedding_layer,
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
    ], name=model_name)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=cm.metrics)
    return model


def create_mc_dropout_model(self, embedding_layer, N, neurons_hidden_layers=[64, 64, 32],  tau=1.0, lengthscale=1e-2, dropout=0.5, model_name="mc_dropout"):
    reg = lengthscale**2 * (1 - dropout) / (2. * N * tau)

    # TODO: test input_shape
    inputs = tfk.Input(
        shape=(embedding_layer.input_shape[1],), dtype='int32')

    inter = embedding_layer(inputs)

    for neurons in range(neurons_hidden_layers):
        inter = tfkl.Dropout(dropout)(inter, training=True)
        inter = tfkl.Dense(neurons, activation='relu',
                           kernel_regularizer=tfk.regularizers.l2(reg))(inter)

    inter = tfkl.Dropout(dropout)(inter, training=True)
    inter = tfkl.Flatten()(inter)
    outputs = tfkl.Dense(1,
                         kernel_regularizer=tfk.regularizers.l2(reg),
                         activation="sigmoid")(inter)
    model = tfk.Model(inputs, outputs, name=model_name)

    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=cm.metrics)
    return model


def create_bayes_by_backprop_model(self, embedding_layer, N, neurons_hidden_layers=[64, 64, 32], variational_layer=tfpl.DenseReparameterization, model_name="bayes_by_backprop"):
    def kernel_divergence_fn(q, p, _):
        return tfd.kl_divergence(q, p) / tf.cast(N, tf.float32)

    model = tfk.Sequential(name=model_name)

    model.add(embedding_layer)
    for neurons in range(neurons_hidden_layers):
        model.add(variational_layer(neurons,
                                    activation='relu',
                                    kernel_divergence_fn=kernel_divergence_fn))
    model.add(tfkl.Flatten())
    model.add(variational_layer(1, activation="sigmoid",
                                kernel_divergence_fn=kernel_divergence_fn))

    model.compile(optimizer=tfk.optimizers.Adam(0.001),
                  loss=tfk.losses.binary_crossentropy,
                  metrics=cm.metrics)

    return model
