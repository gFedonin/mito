import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import is_classifier
from sklearn.model_selection._validation import _safe_split, _score
from sklearn.model_selection import check_cv
from sklearn.externals.joblib import Parallel, delayed
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics.scorer import check_scoring
from sklearn.svm import LinearSVC
from sklearn.utils import check_X_y

import Data as d

def _rfe_single_fit(rfe, estimator, X, y, train, test, scorer):
    """
    Return the score for a fit across one fold.
    """
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)
    return rfe._fit(
        X_train, y_train, lambda estimator, features:
        _score(estimator, X_test[:, features], y_test, scorer))


def compute_selection_stats(estimator, X, y, step=1, cv=None, scoring=None, verbose=0,
                            n_jobs=-1):
    """Fit the RFE model and compute frequencies and accuracies of selected
       features.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the total number of features.

    y : array-like, shape = [n_samples]
        Target values (integers for classification, real numbers for
        regression).
    """
    X, y = check_X_y(X, y, "csr")

    # Initialization
    checked_cv = check_cv(cv, y, is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    n_features = X.shape[1]
    n_features_to_select = 1

    if 0.0 < step < 1.0:
        step = int(max(1, step * n_features))
    else:
        step = int(step)
    if step <= 0:
        raise ValueError("Step must be >0")

    rfe = RFE(estimator=estimator,
              n_features_to_select=n_features_to_select,
              step=step, verbose=verbose)

    # Determine the number of subsets of features by fitting across
    # the train folds and choosing the "features_to_select" parameter
    # that gives the least averaged error across all folds.

    if n_jobs == 1:
        parallel, func = list, _rfe_single_fit
    else:
        parallel, func, = Parallel(n_jobs=n_jobs), delayed(_rfe_single_fit)

    rfes = parallel(
        func(rfe, estimator, X, y, train, test, scorer)
        for train, test in checked_cv.split(X, y))

    featureFreqs = np.zeros(shape=(n_features, n_features))
    accuracy = np.zeros(shape=(n_features, cv))
    j = 0
    for rfe in rfes:
        for featureID in range(n_features):
            featureFreqs[rfe.ranking_[featureID] - 1:n_features, featureID] += 1
            accuracy[featureID, j] = rfe.scores_[n_features - featureID - 1]
        j += 1
    featureFreqs /= cv
    return featureFreqs, accuracy


if __name__ == '__main__':
    mode = 'prop'
    top = 10
    balanced = True
    plt.switch_backend('agg')
    cv = 10
    C = 1
    min_dist = 6
    np.set_printoptions(linewidth=150)

    scores = d.read_data(mode=mode, min_dist=min_dist)

    mode += '_pairs_tree_norm_min_dist_' + str(min_dist)

    y = scores['interact']
    del scores['interact']
    X = scores.values

    if balanced:
        lr = LinearSVC(C=C, dual=False, class_weight='balanced')
    else:
        lr = LinearSVC(C=C, dual=False)

    # selector = RFECV(lr, step=1, cv=cv, n_jobs=-1)
    # selector = selector.fit(X, y)

    featureFreqs, accuracy = compute_selection_stats(lr, X, y, cv=cv, scoring='roc_auc')
    sumFreqs = featureFreqs.sum(axis=0)
    sortedIndices = sumFreqs.argsort()[::-1]
    header = np.array2string(scores.columns.values[sortedIndices])
    name = './freqsRFE_svm_C'
    name += str(C) + '_' + mode
    if balanced:
        name += '_balanced'
    np.savetxt(name + '.csv', featureFreqs[:, sortedIndices], fmt='%1.2f', header=header)

    fig, ax = plt.subplots(figsize=(14, 6))
    plt.title('Accuracy for different splits')
    for i in range(cv):
        plt.plot(range(1, accuracy.shape[0] + 1), accuracy[:, i], label=str(i + 1))
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('FeatureNum')
    name = './accuracyRFE_svm_C'
    name += str(C) + '_' + mode
    if balanced:
        name += '_balanced'
    plt.savefig(name + ".jpg")