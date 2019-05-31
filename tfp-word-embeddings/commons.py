import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
keras = tf.keras


def load_imdb():
    """
    IMDB Dataset
    https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification
    """
    INDEX_FROM = 0

    (x_train, y_train), (x_test, y_test) = imdb.load_data(index_from=INDEX_FROM)

    return ((x_train, y_train), (x_test, y_test))


def load_imdb_word_index():
    word_index = imdb.get_word_index()
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    return word_index


def load_glove_embedding(glove_dir, embedding_dim=50, verbose=True):
    """
    Pretrained Word Embedding
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


class WordIndex:
    def __init__(self, word_index=load_imdb_word_index()):
        self.index = word_index
        self.index_inverse = {v: k for k, v in self.index.items()}

    def vec2sentence(self, vec):
        sentence = map(lambda i: self.index_inverse[i], vec)
        return " ".join(sentence)

    def sentence2vec(self, sentence):
        sentence_arr = sentence.split(" ")
        return list(map(lambda word: self.index[word.lower()], sentence_arr))

    def match_glove(self, embedding_index, embedding_dim, normalize_word_fn=lambda word: word, verbose=True):
        """
        match the word index with the word_embedding index
        """

        unknown_words = []
        embedding_matrix = np.zeros((len(self.index) + 1, embedding_dim))
        for word, i in self.index.items():
            embedding_vector = embedding_index.get(normalize_word_fn(word))
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
            else:
                unknown_words += [word]

        if (verbose):
            print("{}/{} unknown words".format(len(unknown_words), len(self.index)))

        return (embedding_matrix, unknown_words)


def get_max_length(x_train, x_test):
    """
    Max Input Length
    """
    return max(map(lambda vec: len(vec), x_train + x_test))


def pad_input(vec, maxlen):
    return keras.preprocessing.sequence.pad_sequences(vec, maxlen=maxlen, padding='post')


class Rating():
    def __init__(self, word_index, model):
        self.word_index = word_index
        self.model = model

    def of(self, sentences, max_length):
        encoded_custom_sentences = pad_input(
            list(map(self.word_index.sentence2vec, sentences)), max_length)
        custom_result = self.model.predict(encoded_custom_sentences)

        result = []
        for i in range(len(sentences)):
            sentence = sentences[i]
            prediction = custom_result[i][0]
            rating = int(round(prediction * 10))
            result += [(sentence, prediction, rating)]
        return result

    def print(self, ratings):
        for (sentence, prediction, rating) in ratings:
            print("%s (%.2f%%)\n%s\n" %
                  (rating * "⭐", prediction * 100, sentence))
