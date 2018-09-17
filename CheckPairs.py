import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from Data import read_data

predictors = ('fro', 'dca_stat', 'ccmpred_stat', 'wi_score', 'max_pval', 'pvalue', 'nmut_prod', 'nmut_sum', 'site_dist', 'ss_', 'asa_') #
predictors_dc = ('fro', 'dca_stat', 'ccmpred_stat', 'wi_score', 'max_pval', 'pvalue')
predictors_not_dc = ('nmut_prod', 'nmut_sum', 'ss_', 'asa_', 'AA=', 'site_dist')#

mode = 'all'
coef = 1.5
min_dist = 6
n_estimators = 10000
min_weight_fraction_leaf = 0.01
C = 1
method = 'FSSVM'
print_non_dc_plus_single_dc = False
norm = True
forward = True
floating = False
cv = 10


def get_classifier(max_f_num, n_jobs=1):
    if method == 'RF':
        return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1,
                                      min_weight_fraction_leaf=min_weight_fraction_leaf)#, class_weight='balanced'
    elif method == 'LR':
        return LogisticRegression(C=C, penalty="l2", dual=False, n_jobs=-1)
    elif method == 'SVM':
        return LinearSVC(C=C, dual=False)
    elif method == 'FSSVM':
        return SFS(LinearSVC(C=C, dual=False),
               k_features=(1, max_f_num),
               forward=forward,
               floating=floating,
               # verbose=2,
               scoring='roc_auc',
               cv=cv, n_jobs=n_jobs)
    else:
        raise ValueError('Undefined clf type: {}'.format(method))


def get_ranks(clf, X):
    if callable(getattr(clf, "predict_proba", None)):
        return clf.predict_proba(X)[:, 1]
    elif callable(getattr(clf, "decision_function", None)):
        return clf.decision_function(X)
    else:
        raise ValueError('No way to order with clf: {}'.format(method))


def train_no_pairs(predictor_dc, predictor_non_dc, X_train, y_train, X_test, y_test):
    c_list = [x for x in X_train.columns if predictor_non_dc in x and '*' not in x]
    c_list.append(predictor_dc)
    clf = get_classifier(len(c_list))
    clf.fit(X_train[c_list].values, y_train)
    if type(clf) is SFS:
        X_train_sfs = clf.transform(X_train[c_list].values)
        X_test_sfs = clf.transform(X_test[c_list].values)
        clf = clf.estimator
        clf.fit(X_train_sfs, y_train)
        probs = get_ranks(clf, X_test_sfs)
    else:
        probs = get_ranks(clf, X_test[c_list])
    FPR, TPR, thresholds = roc_curve(y_test, probs)
    return predictor_dc, predictor_non_dc, auc(FPR, TPR)


def train_with_pairs(predictor_dc, predictor_non_dc, X_train, y_train, X_test, y_test):
    c_list = [x for x in X_train.columns if predictor_non_dc in x and '*' not in x]
    c_list.append(predictor_dc)
    c_list.extend([x for x in X_train.columns if predictor_dc in x and predictor_non_dc in x])
    clf = get_classifier(len(c_list))
    clf.fit(X_train[c_list].values, y_train)
    if type(clf) is SFS:
        X_train_sfs = clf.transform(X_train[c_list].values)
        X_test_sfs = clf.transform(X_test[c_list].values)
        clf = clf.estimator
        clf.fit(X_train_sfs, y_train)
        probs = get_ranks(clf, X_test_sfs)
    else:
        probs = get_ranks(clf, X_test[c_list])
    FPR, TPR, thresholds = roc_curve(y_test, probs)
    return predictor_dc, predictor_non_dc, auc(FPR, TPR)


def print_cv(df):
    f = open("pairs_" + method + "_" + mode + "_min_dist_" + str(min_dist) + ".txt", "a")
    print('mode = ' + mode, file=f)
    print('CV=' + str(cv), file=f)
    if mode == 'top':
        print('coef = ' + str(coef), file=f)
    print('method = ' + method, file=f)
    if method == 'RF':
        print('n_estimators = ' + str(n_estimators), file=f)
        print('min_weight_fraction_leaf = ' + str(min_weight_fraction_leaf), file=f)
    elif method == 'FSSVM':
        print('forward = ' + str(forward), file=f)
        print('C = ' + str(C), file=f)
    print('min_dist = ' + str(min_dist), file=f)
    print('features: ' + str(Data.features), file=f)

    no_pairs = {}
    with_pairs = {}
    for predictor in predictors_dc:
        no_pairs[predictor] = {}
        with_pairs[predictor] = {}
        for predictor_ndc in predictors_not_dc:
            no_pairs[predictor][predictor_ndc] = []
            with_pairs[predictor][predictor_ndc] = []
    df_pos = df[df['interact'] == 1]
    pos_num = df_pos.shape[0]
    for i in range(cv):
        df_neg = df[df['interact'] == 0].sample(n=pos_num, random_state=i)
        X = df_neg.append(df_pos)
        y = X['interact']
        del X['interact']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/cv, random_state=i)
        tasks = Parallel(n_jobs=-1)(delayed(train_no_pairs)(predictor_dc, predictor_not_dc, X_train, y_train, X_test, y_test)
                                    for predictor_dc in predictors_dc for predictor_not_dc in predictors_not_dc)
        for task in tasks:
            predictor_dc, predictor_non_dc, score = task
            no_pairs[predictor_dc][predictor_non_dc].append(score)

        tasks = Parallel(n_jobs=-1)(delayed(train_with_pairs)(predictor_dc, predictor_not_dc, X_train, y_train, X_test, y_test)
                                    for predictor_dc in predictors_dc for predictor_not_dc in predictors_not_dc)
        for task in tasks:
            predictor_dc, predictor_non_dc, score = task
            with_pairs[predictor_dc][predictor_non_dc].append(score)

    s = 'dc predictor'
    for predictor in predictors_not_dc:
        s += '\t' + predictor
    print(s, file=f)
    for predictor in predictors_dc:
        print(predictor, file=f)
        s = 'no pairs'
        for predictor_ndc in predictors_not_dc:
            l = no_pairs[predictor][predictor_ndc]
            s += '\t' + str(np.mean(l)) + " +/- " + str(np.std(l))
        print(s, file=f)
        s = 'with pairs'
        for predictor_ndc in predictors_not_dc:
            l = with_pairs[predictor][predictor_ndc]
            s += '\t' + str(np.mean(l)) + " +/- " + str(np.std(l))
        print(s, file=f)
    print(file=f)
    print(file=f)
    f.close()


if __name__ == '__main__':
    data = read_data(mode=mode, min_dist=min_dist, norm=norm, inverse_pval=False, keep_prot_name=False, top=coef)
    print_cv(data)
