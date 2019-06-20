# Environment

All Jupyter Notebooks use Tensoflow 2.0 Beta. Some may still work with Tensorflow >1.13.
The Anavonda enviroment can be created using the following commands:

```sh
conda env create -n tf2
conda activate tf2
pip install tensorflow==2.0.0-beta1 tensorflow-gpu==2.0.0-beta1 tfp-nightly matplotlib nltk pandas
conda install jupyter
```

# Python Notebooks

| Name                | Description                    | Path                                                                |
|---------------------|--------------------------------|---------------------------------------------------------------------|
| Deep nn             | Deep nn with l2 regularization | [./nn/word_embedding_nn_0.ipynb](./nn/word_embedding_nn_0.ipynb)    |
| CNN                 | CNN with Dropout               | [./nn/word_embedding_nn_1.ipynb](./nn/word_embedding_nn_1.ipynb)    |
| MC Dropout          |                                | [./bnn/mc_dropout.ipynb](./bnn/mc_dropout.ipynb)                    |
| Bayes by Backprop   |                                | [./bnn/bayes_by_backprop.ipynb](./bnn/bayes_by_backprop.ipynb)      |
| Distribution Lambda |                                | [./bnn/distribution_lambda.ipynb](./bnn/distribution_lambda.ipynbb) |
