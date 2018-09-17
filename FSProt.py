# import distributed.joblib
from collections import OrderedDict

from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import Data as d

mode = 'all'
method = 'svm'
top = 1.5
forward = False
balanced = True
floating = False
scoring = 'roc_auc'#'accuracy'#
cv = 10
C = 1
min_dist = 6


def get_classifier():
    if balanced:
        if method == 'lr':
            return LogisticRegression(C=C, penalty="l2", dual=False, class_weight='balanced')
        elif method == 'svm':
            return LinearSVC(C=C, dual=False, class_weight='balanced')
        elif method == 'tree':
            return DecisionTreeClassifier(min_samples_leaf=100, class_weight='balanced')
        else:
            raise ValueError('Undefined lr type: {}'.format(method))
    else:
        if method == 'lr':
            return LogisticRegression(C=C, penalty="l2", dual=False)
        elif method == 'svm':
            return LinearSVC(C=C, dual=False)
        elif method == 'tree':
            return DecisionTreeClassifier(min_samples_leaf=100)
        else:
            raise ValueError('Undefined lr type: {}'.format(method))
        # return MLPClassifier(alpha=1e-5, hidden_layer_sizes=(10, ))


def select_on_split(X, y):
    sfs = SFS(get_classifier(),
               k_features=(1, feature_num),
               forward=forward,
               floating=floating,
               # verbose=2,
               scoring=scoring,
               cv=cv, n_jobs=-1)
    sfs = sfs.fit(X.values, y)
    return sfs.subsets_


def compute_selection_stats(X, y, f_num):
    features = {}
    accuracy = {}
    for prot_name in d.prot_names:
        X_prot = X[X['prot_name'] == prot_name]
        y_prot = y[X['prot_name'] == prot_name]
        del X_prot['prot_name']
        subsets = select_on_split(X_prot, y_prot)
        feature_set = set()
        acc = np.zeros(f_num)
        sel_f = np.zeros(f_num, dtype=int)
        od = OrderedDict(sorted(subsets.items()))
        for key, value in od.items():
            for feature_id in value['feature_idx']:
                if feature_id not in feature_set:
                    sel_f[key - 1] = int(feature_id)
                    feature_set.add(feature_id)
                    break
            acc[key-1] = value['avg_score']
        accuracy[prot_name] = acc
        features[prot_name] = sel_f
    return features, accuracy

if __name__ == '__main__':

    plt.switch_backend('agg')
    np.set_printoptions(linewidth=150)

    scores = d.read_data(mode=mode, min_dist=min_dist, norm=True, inverse_pval=False, keep_prot_name=True)

    # print(scores.columns.values)
    # print(np.corrcoef(scores, rowvar=False))

    mode += '_min_dist_' + str(min_dist) + '_' + scoring

    y = scores['interact']
    del scores['interact']
    X = scores


    feature_num = len(scores.columns) - 1
    # feature_num = min(20, len(scores.columns))

    features, accuracy = compute_selection_stats(X, y, feature_num)

    fig, ax = plt.subplots(figsize=(20, 10))

    del X['prot_name']

    for prot_name in d.prot_names:
        plt.title('Cross Validation ' + scoring + ' for ' + prot_name)
        x = range(feature_num)
        # print(features[prot_name])
        # print(X.columns[features[prot_name]])
        plt.xticks(x, X.columns[features[prot_name]])
        plt.plot(x, accuracy[prot_name])
        plt.ylabel(scoring)
        plt.xlabel('Selected features')
        name = './ProtPlots/' + prot_name
        if(forward):
            name += '_FFS'
        else:
            name += '_BFS'
        name += '_' + method + '_' + str(C) + '_' + mode
        if balanced:
            name += '_balanced'
        plt.savefig(name + ".png")
        plt.clf()


