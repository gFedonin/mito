import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import is_classifier, clone, BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection.base import SelectorMixin
from sklearn.model_selection._validation import _safe_split, _score
from sklearn.model_selection import check_cv, KFold
from sklearn.externals.joblib import Parallel, delayed
from sklearn.metrics.scorer import check_scoring
from sklearn.svm import LinearSVC
from sklearn.utils import check_X_y

import Data as d


class RFE:
    """Feature ranking with recursive feature elimination.

    Given an external estimator that assigns weights to features (e.g., the
    coefficients of a linear model), the goal of recursive feature elimination
    (RFE) is to select features by recursively considering smaller and smaller
    sets of features. First, the estimator is trained on the initial set of
    features and weights are assigned to each one of them. Then, features whose
    absolute weights are the smallest are pruned from the current set features.
    That procedure is recursively repeated on the pruned set until the desired
    number of features to select is eventually reached.

    Read more in the :ref:`User Guide <rfe>`.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method that updates a
        `coef_` attribute that holds the fitted parameters. Important features
        must correspond to high absolute values in the `coef_` array.

        For instance, this is the case for most supervised learning
        algorithms such as Support Vector Classifiers and Generalized
        Linear Models from the `svm` and `linear_model` modules.

    n_features_to_select : int or None (default=None)
        The number of features to select. If `None`, half of the features
        are selected.

    step : int or float, optional (default=1)
        If greater than or equal to 1, then `step` corresponds to the (integer)
        number of features to remove at each iteration.
        If within (0.0, 1.0), then `step` corresponds to the percentage
        (rounded down) of features to remove at each iteration.

    verbose : int, default=0
        Controls verbosity of output.

    Attributes
    ----------
    n_features_ : int
        The number of selected features.

    support_ : array of shape [n_features]
        The mask of selected features.

    ranking_ : array of shape [n_features]
        The feature ranking, such that ``ranking_[i]`` corresponds to the
        ranking position of the i-th feature. Selected (i.e., estimated
        best) features are assigned rank 1.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    The following example shows how to retrieve the 5 right informative
    features in the Friedman #1 dataset.

    >>> from sklearn.datasets import make_friedman1
    >>> from sklearn.feature_selection import RFE
    >>> from sklearn.svm import SVR
    >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
    >>> estimator = SVR(kernel="linear")
    >>> selector = RFE(estimator, 5, step=1)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True,  True,  True,  True,  True,
            False, False, False, False, False], dtype=bool)
    >>> selector.ranking_
    array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])

    References
    ----------

    .. [1] Guyon, I., Weston, J., Barnhill, S., & Vapnik, V., "Gene selection
           for cancer classification using support vector machines",
           Mach. Learn., 46(1-3), 389--422, 2002.
    """
    def __init__(self, estimator, use_std = True):
        self.estimator = estimator
        self.use_std = use_std

    def fit(self, X, y):
        """Fit the RFE model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        return self._fit(X, y)

    def _fit(self, X, y, step_score=None):
        X, y = check_X_y(X, y, "csc")
        # Initialization
        n_features = X.shape[1]

        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        if step_score:
            self.scores_ = []

        # Elimination
        while np.sum(support_) > 1:
            # Remaining features
            features = np.arange(n_features)[support_]

            # Rank the remaining features
            kf = KFold(n_splits=cv)
            kf.get_n_splits(X)
            coefs_array = np.zeros(shape=(cv, n_features))
            i = 0
            for train_index, test_index in kf.split(X):
                estimator = clone(self.estimator)
                estimator.fit(X[train_index, features], y[train_index])
                # Get coefs
                if hasattr(estimator, 'coef_'):
                    coefs = estimator.coef_
                else:
                    coefs = getattr(estimator, 'feature_importances_', None)
                if coefs is None:
                    raise RuntimeError('The classifier does not expose '
                                       '"coef_" or "feature_importances_" '
                                       'attributes')
                coefs_array[i] = coefs
                i += 1
            coefs = coefs_array.mean(axis=0)
            if self.use_std:
                coefs = coefs/coefs_array.std(axis=0)

            ranks = np.argsort(abs(coefs))

            # Compute step score on the previous selection iteration
            # because 'estimator' must use features
            # that have not been eliminated yet
            if step_score:
                self.scores_.append(step_score(estimator, features))
            support_[features[ranks][0]] = False
            ranking_[np.logical_not(support_)] += 1

        # Set final attributes
        features = np.arange(n_features)[support_]

        # Compute step score when only n_features_to_select features left
        if step_score:
            self.scores_.append(step_score(self.estimator_, features))
        self.ranking_ = ranking_

        return self

    def rfe_single_fit(rfe, estimator, X, y, train, test, scorer):
        """
        Return the score for a fit across one fold.
        """
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)
        return rfe._fit(
            X_train, y_train, lambda estimator, features:
            _score(estimator, X_test[:, features], y_test, scorer))


def compute_selection_stats(estimator, X, y, cv=None, scoring=None, n_jobs=-1):
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

    rfe = RFE(estimator=estimator)

    # Determine the number of subsets of features by fitting across
    # the train folds and choosing the "features_to_select" parameter
    # that gives the least averaged error across all folds.

    if n_jobs == 1:
        parallel, func = list, RFE.rfe_single_fit
    else:
        parallel, func, = Parallel(n_jobs=n_jobs), delayed(RFE.rfe_single_fit)

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
        lr = LinearSVC(C=C, dual=False, class_weight='balanced', penalty='l1')
    else:
        lr = LinearSVC(C=C, dual=False, penalty='l1')

    # selector = RFECV(lr, step=1, cv=cv, n_jobs=-1)
    # selector = selector.fit(X, y)

    featureFreqs, accuracy = compute_selection_stats(lr, X, y, cv=cv, scoring='roc_auc')
    sumFreqs = featureFreqs.sum(axis=0)
    sortedIndices = sumFreqs.argsort()[::-1]
    header = np.array2string(scores.columns.values[sortedIndices])
    name = './freqsRFECV_svm_C'
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
    name = './accuracyRFECV_svm_C'
    name += str(C) + '_' + mode
    if balanced:
        name += '_balanced'
    plt.savefig(name + ".jpg")