import pandas as pd
import numpy as np
from numpy.random.mtrand import RandomState
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

if __name__ == '__main__':
    np.set_printoptions(linewidth=150)

    scores = pd.read_csv("./data/scores_no_sco_norm.csv", sep=';')

    print(scores.columns.values)
    print(np.corrcoef(scores, rowvar=False))

    scores = pd.read_csv("./data/scores_no_sco_ext.csv", sep=';')

    y = scores['interact']
    del scores['interact']
    X = scores.values




    lsvc = LinearSVC(C=100, penalty="l1", dual=False)
    lsvc.fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    print(scores.columns.values[model.get_support()])

    X_new = model.transform(X)
    lsvc = LinearSVC(C=100, penalty="l2", dual=False)
    lsvc.fit(X_new, y)
    print(lsvc.score())
    print(lsvc.coef_)

