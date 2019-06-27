# Bayesian Neural Networks for NLP

## Environment

All Jupyter Notebooks use Tensoflow 2.0 Beta, but most of the still work with Tensorflow >=1.13.
The Anavonda enviroment can be created using the following commands:

```sh
conda env create -n tf2
conda activate tf2
pip install tensorflow==2.0.0-beta1 tensorflow-gpu==2.0.0-beta1 tfp-nightly matplotlib nltk pandas
conda install jupyter
```

## GloVe Word Embedding

The GloVe word embedding can downloaded here: https://nlp.stanford.edu/projects/glove/
Here, the Wikipedia crawl `glove.6B.zip` is used.

## Pretrained models and training history

All models are trained for 50 epochs and for different dataset sizes. You can download the trained models and history data from:
https://drive.google.com/drive/folders/12ToPTgfyZaL8yteXZlmgKLqT1Wjck3G8?usp=sharing

## Python Notebooks

| Name                | Description                              | Path                                                                |
| ------------------- | ---------------------------------------- | ------------------------------------------------------------------- |
| Deep NN             | Deep nn with l2 regularization           | [./nn/word_embedding_nn_0.ipynb](./nn/word_embedding_nn_0.ipynb)    |
| CNN                 | CNN with Dropout                         | [./nn/word_embedding_nn_1.ipynb](./nn/word_embedding_nn_1.ipynb)    |
| MC Dropout          | Using Dropout at prediction time         | [./bnn/mc_dropout.ipynb](./bnn/mc_dropout.ipynb)                    |
| Bayes by Backprop   | Flipout and Reparameterization estimator | [./bnn/bayes_by_backprop.ipynb](./bnn/bayes_by_backprop.ipynb)      |
| Distribution Lambda | Tests with TFPs variational layers       | [./bnn/distribution_lambda.ipynb](./bnn/distribution_lambda.ipynbb) |
