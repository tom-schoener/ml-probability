import os
import re

import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
keras = tf.keras


def load_imdb():
    """
    Loads the IMDB dataset and stores it locally for later use.
    https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification
    """
    INDEX_FROM = 0

    (x_train, y_train), (x_test, y_test) = imdb.load_data(index_from=INDEX_FROM)
    return ((x_train, y_train), (x_test, y_test))


def load_imdb_word_index():
    """
    Loads the imbd word index. It uses the following special words at the beginning: 
    0: <PAD> 
    1: <START>
    2: <UNK>
    """
    word_index = imdb.get_word_index()
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

    def __init__(self, word_index=load_imdb_word_index()):
        self.index = word_index
        self.index_inverse = {v: k for k, v in self.index.items()}

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

    def match_glove(self, embedding_index, embedding_dim, normalize_word_fn=word_net_lemmatizer_fn(), verbose=True):
        """
        Matches the word index with the word_embedding index.
        """

        unknown_words = []
        embedding_matrix = np.zeros((len(self.index) + 1, embedding_dim))
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
    vec -- word vector representation
    max_length -- max number of words in a sentence
    """
    return keras.preprocessing.sequence.pad_sequences(vec, maxlen=max_length, padding='post')


class Rating():
    """
    Enables a quick and easy way to validate the model with custom sentences.
    """

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
