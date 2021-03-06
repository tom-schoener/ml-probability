import os
import re
import random
import math

import pandas as pd
import matplotlib.pyplot as plt
import nltk
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.datasets import imdb
keras = tf.keras
tfk = tf.keras

tfkl = tfk.layers
tfd = tfp.distributions
tfpl = tfp.layers


def one_time_setup():
    import nltk
    nltk.download('wordnet')


def mount_google_drive():
    from google.colab import drive
    drive.mount('/content/drive')


def setup(glove_dir, embedding_dim=50, trainingset_proportion=0.5, words_per_sentence=1000):
    """
    Utility setup method loading all required datasets and word indecies.

    Keyword arguments:
    glove_dir               -- GloVe directory as an absolute path
    embedding_dim           -- dimension of the GloVe word embedding. (default: 50)
    trainingset_proportion  -- training dataset proportion. Must be between 0 and 0.95.
    words_per_sentence      -- max length of words in a review to be included in the returned datasets. All
                               other sequences will have this length (padded with 0).
    """

    dataset_props = [trainingset_proportion,
                     1 - 0.05 - trainingset_proportion,
                     0.05]
    (x_train, y_train), (x_test, y_test), (x_validation, y_validation) = load_imdb(
        dataset_proportions=dataset_props,
        maxlen=words_per_sentence)

    word_index = WordIndex(embedding_dim=embedding_dim)
    embedding_index = load_glove_embedding(glove_dir, embedding_dim)
    (embedding_matrix, unknown_words) = word_index.match_glove(
        embedding_index=embedding_index)

    # pad input vectors
    x_train_padded = pad_input(x_train, words_per_sentence)
    x_test_padded = pad_input(x_test, words_per_sentence)
    x_validation_padded = pad_input(x_validation, words_per_sentence)

    embedding_layer = word_index.as_embedding_layer(
        words_per_sentence, embedding_matrix)

    return {
        "train": (x_train, x_train_padded, y_train),
        "test": (x_test, x_test_padded, y_test),
        "validation": (x_validation, x_validation_padded, y_validation),
        "word_index": word_index,
        "embedding_layer": embedding_layer,
        "embedding_input_dim": words_per_sentence
    }


def load_history_from_file(history_save_file):
    """
    Loads the preserved history from the saved csv file
    """
    try:
        history_df = pd.read_csv(history_save_file, sep=";", index_col="epoch")
        last_epoch = history_df.index[-1] + 1
        print("Loaded history successfully. Last epoch: %i" % last_epoch)
        return (history_df, last_epoch)
    except pd.errors.EmptyDataError:
        print("Cannot load history file (EmptyDataError)")
    except OSError:
        print("Cannot load history file (OSError)")
    except FileNotFoundError:
        print("No saved history file")

    return (None, 0)


def load_imdb(dataset_proportions=(0.5, 0.45, 0.05), maxlen=None):
    """
    Loads the IMDB dataset and stores it locally for later use. Returns a triple with
    (x_train, y_train), (x_test, y_test), (x_validation, y_validation)
    https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification

    Keyword arguments:
    dataset_proportions     -- dataset proportions for train, test and validation sizes. Must add up to 1.
    maxlen                  -- max length of words in a review to be included in the returned datasets
    """
    (x_train, y_train), (x_test, y_test) = imdb.load_data(index_from=3,
                                                          seed=math.floor(
                                                              dataset_proportions[0] * 100),
                                                          maxlen=maxlen)

    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    dataset_proportions_cumsum = np.cumsum(dataset_proportions)
    datasets = []
    for start_prop, end_prop in zip([0, *dataset_proportions_cumsum[:-1]], dataset_proportions_cumsum):
        start, end = (math.floor(len(x) * start_prop),
                      math.floor(len(x) * end_prop))
        datasets += [(x[start:end], y[start:end])]

    return datasets


def load_imdb_word_index():
    """
    Loads the imbd word index. It uses the following special words at the beginning:
    0: <PAD>
    1: <START>
    2: <UNK>
    """
    word_index = imdb.get_word_index()

    for word, index in word_index.items():
        word_index[word] = index + 3

    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    return word_index


def load_glove_embedding(glove_dir, embedding_dim=50, verbose=True):
    """
    Loads the pretrained GloVe word embedding and returns it as a
    dictonary.

    Keyword arguments:
    glove_dir       -- GloVe directory as an absolute path
    embedding_dim   -- dimension of the GloVe word embedding. (default: 50)
    verbose         -- if False it won't log additional information (default: True)
    """

    embedding_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.%sd.txt' %
                          embedding_dim), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs
    f.close()

    if (verbose):
        print('Found %s word vectors.' % len(embedding_index))

    return embedding_index


def word_net_lemmatizer_fn():
    """
    Returns a function which lemmatizes a given word using the english language.

    Note: Requires the wordnet corpus. To download the corpus execute the following
    python script: nltk.download()
    """

    lemmatizer = nltk.stem.WordNetLemmatizer()

    def fn(word):
        word = re.sub(r'\W', '', word)
        return lemmatizer.lemmatize(word)
    return fn


class WordIndex:
    """
    Utility class to handle the imdb word index.
    """

    def __init__(self, word_index=load_imdb_word_index(), embedding_dim=50):
        self.index = word_index
        self.index_inverse = {v: k for k, v in self.index.items()}
        self.embedding_dim = 50

    def vec2sentence(self, vec):
        """
        Converts a word vector representation into its sentence counterpart.

        Keyword arguments:
        vec -- word vector
        """
        sentence = map(lambda i: self.index_inverse[i], vec)
        return " ".join(sentence)

    def sentence2vec(self, sentence):
        """
        Converts a sentence into its word vector representation.

        Keyword arguments:
        sentence -- english sentence
        """
        sentence_arr = sentence.split(" ")

        def convert(word):
            if word in self.index:
                return self.index[word.lower()]
            else:
                return self.index["<UNK>"]
        return list(map(convert, sentence_arr))

    def match_glove(self, embedding_index, normalize_word_fn=word_net_lemmatizer_fn(), verbose=True):
        """
        Matches the word index with the word_embedding index.
        """

        unknown_words = []
        embedding_matrix = np.zeros((len(self.index) + 1, self.embedding_dim))
        for word, i in self.index.items():
            embedding_vector = embedding_index.get(word)

            if embedding_vector is None:
                embedding_vector = embedding_index.get(normalize_word_fn(word))

            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                unknown_words += [word]

        if (verbose):
            print("{}/{} unknown words".format(len(unknown_words), len(self.index)))

        return (embedding_matrix, unknown_words)

    def as_embedding_layer(self, words_per_sentence, embedding_matrix):
        """
        Creates a keras embedding layers based on the word index. The weights are not trainable.
        """
        return tfkl.Embedding(len(self.index) + 1,
                              self.embedding_dim,
                              weights=[embedding_matrix],
                              input_length=words_per_sentence,
                              trainable=False)


def get_max_length(x_train, x_test=[]):
    """
    Returns the max length of the training and test data.

    Keyword arguments:
    x_train -- training features
    x_test  -- testing features (default: [])
    """
    return max(map(lambda vec: len(vec), x_train + x_test))


def pad_input(vec, max_length):
    """
    Pads the vec by appending zeros until a length of max_length is reached.

    Keyword arguments:
    vec         -- word vector representation
    max_length  -- max number of words in a sentence
    """
    return keras.preprocessing.sequence.pad_sequences(vec, maxlen=max_length, padding='post')


class Rating():
    """
    Provides a quick and easy way to validate the model with custom sentences.
    """

    def __init__(self, word_index, model):
        self.word_index = word_index
        self.model = model
        self.max_length = model.input_shape[1]

    def of(self, sentences):
        encoded_custom_sentences = pad_input(
            list(map(self.word_index.sentence2vec, sentences)), self.max_length)
        custom_result = self.model.predict(encoded_custom_sentences)

        result = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            prediction = custom_result[i][0]
            rating = int(round(prediction * 10))
            result += [(sentence, prediction, rating)]
        return result

    def print(self, ratings):
        """
        Prints the ratings using the star symbol

        Keyword arguments:
        ratings -- ratings generated by the of method
        """
        for (sentence, prediction, rating) in ratings:
            print("%s (%.2f%%)\n%s\n" %
                  (rating * "⭐", prediction * 100, sentence))


def plot_confidence(means, stddevs, true_ys):
    """
    Plots the confidence using errorbars.

    Keyword arguments:
    means    -- expectations
    stddevs  -- samples standard deviations
    true_ys  -- true y values
    """
    x = np.arange(0, len(means), 1)
    y = means
    yerr = np.array(stddevs) * 2

    fig, ax = plt.subplots()
    plt.xlabel("movie")
    plt.ylabel("prediction")

    ax.hlines(y=[0.4, 0.5, 0.6], xmin=0, xmax=len(x) - 1, linewidth=1,
              linestyle=":", color=["gray", "black", "gray"])
    ax.hlines(y=[0, 1], xmin=0, xmax=len(x) - 1, linewidth=1,
              linestyle="--", color=["black", "black"])
    ax.errorbar(x, y, yerr=yerr, fmt="o", elinewidth=1, color="black")
    ax.errorbar(x, true_ys, fmt="x", color="r")



def plot_metric(name, history_df):
    """
    Plots the metric.

    Keyword arguments:
    name       -- name of the metric
    history_df -- history dataframe
    """
    plt.plot(history_df[name])
    plt.plot(history_df["val_%s" % name])
    plt.title('Model %s' % name)
    plt.ylabel(name)
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.grid(True)


def get_keras_callbacks(model_save_file, history_save_file, weights_only=False):
    """
    Returns a list of keras callbacks to save the model state and history.

    Keyword arguments:
    model_save_file    -- absolute model file path
    history_save_file  -- absolute history file path
    weights_only       -- set to True to save only the weights
    """
    return [
        tfk.callbacks.CSVLogger(history_save_file, append=True, separator=';'),
        tfk.callbacks.ModelCheckpoint(model_save_file,
                                      monitor='val_loss',
                                      verbose=0,
                                      save_best_only=True,
                                      save_weights_only=weights_only,
                                      mode='auto')
    ]

metrics = [
    "acc",
    tfk.metrics.TrueNegatives(name="true_negatives"),
    tfk.metrics.FalseNegatives(name="false_negatives"),
    tfk.metrics.TruePositives(name="true_positives"),
    tfk.metrics.FalsePositives(name="false_positives"),
    tfk.metrics.Precision(name="precision"),
    tfk.metrics.Recall(name="recall"),
    tfk.metrics.KLDivergence(name="kl"),
]
