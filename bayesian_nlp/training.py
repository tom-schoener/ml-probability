import tensorflow_probability as tfp

from models import DefaultDenseModel, DefaultConvModel, McDropoutModel, BayesByBackpropModel, BayesianConvModel
import commons


def create_models(setup, models_dir, history_dir):
    """
    Creates a list of trainable models.
    """

    return [
        DefaultDenseModel(model_setup=setup,
                          models_dir=models_dir,
                          history_dir=history_dir,
                          neurons_hidden_layers=[64, 64, 32]),
        DefaultConvModel(model_setup=setup,
                         models_dir=models_dir,
                         history_dir=history_dir),
        McDropoutModel(model_setup=setup,
                       models_dir=models_dir,
                       history_dir=history_dir,
                       neurons_hidden_layers=[64, 64, 32],
                       tau=1.0,
                       lengthscale=1e-2,
                       dropout=0.5),
        BayesByBackpropModel(model_setup=setup,
                             models_dir=models_dir,
                             history_dir=history_dir,
                             variational_layer=tfp.layers.DenseFlipout,
                             neurons_hidden_layers=[64, 64, 32]),
        BayesByBackpropModel(model_setup=setup,
                             models_dir=models_dir,
                             history_dir=history_dir,
                             variational_layer=tfp.layers.DenseReparameterization,
                             neurons_hidden_layers=[64, 64, 32]),
        BayesianConvModel(model_setup=setup,
                          models_dir=models_dir,
                          history_dir=history_dir,)
    ]


def train_models(models_dir, history_dir, glove_dir, epochs=50, dataset_sizes=[0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.005], models_provider=create_models):
    """
    Trains the models listed in create_models. The progess including the resulting weights of the model
    and the training history will be preserved at the specified location.
    """

    commons.one_time_setup()

    embedding_dim = 50
    batch_size = 128

    for dataset_size in dataset_sizes:
        setup = commons.setup(glove_dir=glove_dir,
                              embedding_dim=embedding_dim,
                              trainingset_proportion=dataset_size)
        models = models_provider(
            setup, models_dir=models_dir, history_dir=history_dir)

        for model in models:
            print("\n======\nTraining model %s\n======\n" % model.model_id())

            model.fit(epochs=epochs, batch_size=batch_size)
