from models import DefaultDenseModel, DefaultConvModel, McDropoutModel, BayesByBackpropModel
import commons


# commons.one_time_setup()

glove_dir = "D:/google drive/haw/master/mastertheisis/hauptprojekt/glove"
embedding_dim = 50
dataset_size = 1.0
epochs = 0
batch_size = 128
words_per_sentence = 2800

setup = commons.setup(glove_dir=glove_dir,
                      embedding_dim=embedding_dim,
                      dataset_size=dataset_size,
                      words_per_sentence=words_per_sentence)


models = [
    DefaultDenseModel(model_setup=setup,
                      neurons_hidden_layers=[64, 64, 32]),
    DefaultConvModel(model_setup=setup),
    McDropoutModel(model_setup=setup,
                   neurons_hidden_layers=[64, 64, 32],
                   tau=1.0,
                   lengthscale=1e-2,
                   dropout=0.5),
    BayesByBackpropModel(model_setup=setup,
                         neurons_hidden_layers=[64, 64, 32])
]

weights_only_list = [False, False, False, True]

for model, weights_only in zip(models, weights_only_list):
    print("====\nTraining model %s\n====" % model.model_id())
    for i in range(2):
        model.fit(epochs=epochs, batch_size=batch_size,
                  weights_only=weights_only)
