# import distributed.joblib
from sklearn.externals.joblib import Parallel, delayed#, parallel_backend
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt
# from dask.distributed import Client
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import Data as d

mode = 'top'
method = 'svm'
top = 10
forward = True
balanced = True
floating = False
cv = 10
C = 1
min_dist = 2


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


def select_on_split(X, y, n_jobs):
    sfs = SFS(get_classifier(),
               k_features=(1, feature_num),
               forward=forward,
               floating=floating,
               # verbose=2,
               scoring='roc_auc',
               cv=cv, n_jobs=n_jobs)
    sfs.fit(X, y)
    return sfs.subsets_


def compute_selection_stats(X, y, f_num, cv=10):
    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)
    if cv < f_num/4:
        work = []
        for train_index, test_index in kf.split(X):
            work.append(select_on_split(X[train_index], y[train_index], n_jobs=-1))
    else:
        parallel = Parallel(n_jobs=-1)
        work = parallel(delayed(select_on_split)(X[train_index], y[train_index], n_jobs=1)
                        for train_index, test_index in kf.split(X))

    feature_freqs = np.zeros(shape=(X.shape[1], X.shape[1]))
    accuracy = np.zeros(shape=(f_num, cv))
    j = 0
    for subset in work:
        for key, value in subset.items():
            for featureID in value['feature_idx']:
                feature_freqs[key - 1, featureID] += 1
            accuracy[key-1, j] = value['avg_score']
        j += 1
    feature_freqs /= cv
    return feature_freqs, accuracy

if __name__ == '__main__':

    plt.switch_backend('agg')
    np.set_printoptions(linewidth=150)

    scores = d.read_data(mode=mode, min_dist=min_dist, norm=False, inverse_pval=False)

    # print(scores.columns.values)
    # print(np.corrcoef(scores, rowvar=False))

    mode += '_dc_only_min_dist_' + str(min_dist)

    y = scores['interact']
    del scores['interact']
    X = scores.values

    feature_num = len(scores.columns)
    # feature_num = min(20, len(scores.columns))

    # client = Client('127.0.0.1:8786')
    # with parallel_backend('dask.distributed', scheduler_host='localhost:8786'):
    feature_freqs, accuracy = compute_selection_stats(X, y, feature_num, cv=cv)
    sum_freqs = feature_freqs.sum(axis=0)
    sorted_indices = sum_freqs.argsort()[::-1]
    header = np.array2string(scores.columns.values[sorted_indices])
    # print(type(header))
    name = './freqs'
    if forward:
        name += 'FFS'
    else:
        name += 'BFS'
    if floating:
        name += '_floating_'
    name += '_' + method + '_' + str(C) + '_' + mode
    if balanced:
        name += '_balanced'
    np.savetxt(name + '.csv', feature_freqs[:, sorted_indices[:feature_num]], fmt='%1.2f', header=header)

    fig, ax = plt.subplots(figsize=(14, 6))
    plt.title('Accuracy for different splits')
    for i in range(cv):
        plt.plot(range(1, accuracy.shape[0] + 1), accuracy[:, i], label=str(i + 1))
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('FeatureNum')
    name = './accuracy'
    if forward:
        name += 'FFS'
    else:
        name += 'BFS'
    name += '_' + method + '_' + str(C) + '_' + mode
    if balanced:
        name += '_balanced'
    plt.savefig(name + ".jpg")


