# Planung: Einarbeitung Tensorflow Propability

## Grundlagen

Einige grundlegende (mathematische) Konzepte für die Programmierung mit Tensorflow Probability sollten vorher wiederholt werden. Die wichtigsten sind unten aufgelistet. Die Liste ist nicht unbedingt vollständig.

### Einarbeitung/ Wiederholung Stochastik und Statistik

- Uni- and Multivariate https://learnche.org/pid/
- Varianz (Kovarianz)
- Binomial distribution (Bernoulli distribution)
- Markov Chain Monte Carlo (MCMC)
- Variational Inference (VI)
- Bayesian Linear Regression
- Statistik https://sites.google.com/site/fundamentalstatistics/

### Einrichtung des Workspaces

- Lokal: Python mit Tensorflow, Pandas, Numpy usw. mit Anaconda
- Remote: Colaboratory Notebooks von Google https://colab.research.google.com/notebooks/

## Tensorflow Probability

Für die Einarbeitung in Tensorflow Probability eignet sich, neben den von Google bereitgestellten Tutorials, der Datensatz "Air Quality" sehr gut. Er bietet genügend Attribute/ Features und es können leicht synthetische Features erzeugt werden. Zudem ist jedes Feature unter https://archive.ics.uci.edu/ml/datasets/Air+quality kurz beschrieben. Ein sinnvolles Ziel wäre für mich, möglichst jedes Tutorial einmal durchzuarbeiten, mich mit der Dokumentation von Tensorflow Probability vertraut zu machen und nebenbei mit dem selbstausgewählten Datensatz zu experimentieren. Für den Datensatz ließe sich dann noch eine Aufgabe definieren, auf welche ich zum Ende des Grundprojekts hinarbeite. Beispielsweise: „Eine erhöhte Luftfeuchtigkeit führt zu einer besseren Luftqualität“.

### Tutorials zum Einarbeiten

- Allgemeine Tensorflow Tutorials https://github.com/aymericdamien/TensorFlow-Examples
- Tensorflow Probability Tutorials https://www.tensorflow.org/probability/overview

### Datasets

Datasets aus https://archive.ics.uci.edu/ml/

| Name                           | Quelle                                                               | Attribute | Einträge | Typ       |
| ------------------------------ | -------------------------------------------------------------------- | --------- | -------- | --------- |
| Twitter Health news            | https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter       | 6         | 58000    | Text      |
| Sentiment Labelled Sentences   | https://archive.ics.uci.edu/ml/datasets/Sentiment+Labelled+Sentences | 2         | 3000     | Text      |
| Communities and Crime Data Set | https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime        | 128       | 1994     | Numerisch |
| Air Quality                    | https://archive.ics.uci.edu/ml/datasets/Air+quality                  | 15        | 9358     | Numerisch |

## Grobe Zeiteinteilung

- Grundlagen: 3-4 Wochen
- Tensorflow Probability: 8 Wochen (mit Überschneidung)
  - davon Tutorials: 4 Wochen
  - Tests mit dem ausgewählten Dataset: 6 Wochen

Tom Schöner, Informatik Master
