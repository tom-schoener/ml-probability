import random
import numpy
from tensorflow import keras

good_reviews = [
    "good movie",
    "very good movie",
    "not a bad movie",
    "not bad"
]

bad_reviews = [
    "bad movie",
    "very bad movie",
    "not a good movie",
    "not good",
    "not very bad",
    "very very bad"
]


def load_simple_data(good_reviews=good_reviews, bad_reviews=bad_reviews):
    data = list(zip(
        good_reviews + bad_reviews,
        list(map(lambda _: 1, good_reviews)) +
        list(map(lambda _: 0, bad_reviews))
    ))
    random.shuffle(data)

    tokenizer = keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(good_reviews + bad_reviews)

    # print(tokenizer.word_counts)
    # print(tokenizer.document_count)
    # print(tokenizer.word_index)
    # print(tokenizer.index_word)
    # print(tokenizer.word_docs)
    # print(data)
    # print(tokenizer.texts_to_matrix(["very very good"]))

    print(tokenizer.texts_to_sequences(["very very good"]))

    return (tokenizer, data)
