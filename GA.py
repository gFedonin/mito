import pandas as pd
import numpy as np
from genetic_selection import GeneticSelectionCV

from sklearn.linear_model import LogisticRegression

if __name__ == '__main__':
    np.set_printoptions(linewidth=150)

    scores = pd.read_csv("./data/scores_no_sco_norm.csv", sep=';')

    print(scores.columns.values)
    print(np.corrcoef(scores, rowvar=False))

    scores = pd.read_csv("./data/scores_no_sco_ext.csv", sep=';')

    y = scores['interact']
    del scores['interact']
    # X = preprocessing.scale(scores)
    X = scores.values

    lr = LogisticRegression(C=1, penalty="l2", dual=False)


    selector = GeneticSelectionCV(lr,
                                  cv=10,
                                  verbose=1,
                                  scoring="accuracy",
                                  n_population=100,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=10000,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  caching=True,
                                  n_jobs=-1)
    selector = selector.fit(X, y)

    print(scores.columns[selector.support_].values)
