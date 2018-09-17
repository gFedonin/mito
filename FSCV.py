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
import Data

predictors = ('fro', 'dca_stat', 'ccmpred_stat', 'wi_score', 'max_pval', 'pvalue', 'nmut_prod', 'nmut_sum', 'ss_', 'asa_') #, 'site_dist'
predictors_dc = ('fro', 'dca_stat', 'ccmpred_stat', 'wi_score', 'max_pval', 'pvalue')
predictors_not_dc = ('nmut_prod', 'nmut_sum', 'ss_', 'asa_', 'AA=')#, 'site_dist'

mode = 'all'
coef = 1.5
min_dist = 6
n_estimators = 10000
min_weight_fraction_leaf = 0.01
C = 1
method = 'RF'
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


def train_SVC(predictor, X_train, y_train, X_test, y_test):
    if predictor == 'ss_' or predictor == 'AA=' or predictor == 'asa_':
        c_list = [x for x in X_train.columns if predictor in x and '*' not in x] # with predictor but no pairs
        clf = get_classifier(len(c_list))
        # print(c_list)
        clf.fit(X_train[c_list].values, y_train)
        if type(clf) is SFS:
            X_train_sfs = clf.transform(X_train[c_list].values)
            X_test_sfs = clf.transform(X_test[c_list].values)
            clf = clf.estimator
            clf.fit(X_train_sfs, y_train)
            probs = get_ranks(clf, X_test_sfs)
        else:
            probs = get_ranks(clf, X_test[c_list])
    else:
        clf = LinearSVC(C=C, dual=False)
        clf.fit(X_train[predictor].values.reshape(-1, 1), y_train)
        probs = get_ranks(clf, X_test[predictor].values.reshape(-1, 1))
    FPR, TPR, thresholds = roc_curve(y_test, probs)
    return predictor, auc(FPR, TPR)


def train_clf_all_but_one(col_list, predictor, X_train, y_train, X_test, y_test):
    c_list = [x for x in col_list if predictor not in x]
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
    return predictor, auc(FPR, TPR)


def train_clf_one_dc(col_list, predictor, X_train, y_train, X_test, y_test):

    c_list = [] # list of all columns with non-dc predictors, no pairs
    for pred in predictors_not_dc:
        for col in col_list:
            if pred in col and '*' not in col:
                c_list.append(col)
    #now add all columns containing given predictor
    for col in col_list:
        if predictor in col:
            c_list.append(col)
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
    return predictor, auc(FPR, TPR)


def print_cv(df):
    f = open("contacts_CV_FS_" + mode + "_min_dist_" + str(min_dist) + ".txt", "a")
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

    individual_scores = {}
    for predictor in predictors:
        individual_scores[predictor] = []
    all_features_scores = []
    all_features_no_pairs= []
    no_wi_scores = []
    all_but_one_scores = {}
    for predictor in predictors:
        all_but_one_scores[predictor] = []
    no_dc_scores = []
    non_dc_plus_one = {}
    for predictor in predictors_dc:
        non_dc_plus_one[predictor] = []
    df_pos = df[df['interact'] == 1]
    pos_num = df_pos.shape[0]
    for i in range(cv):
        df_neg = df[df['interact'] == 0].sample(n=pos_num, random_state=i)
        X = df_neg.append(df_pos)
        y = X['interact']
        del X['interact']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/cv, random_state=i)
        tasks = Parallel(n_jobs=-1)(delayed(train_SVC)(predictor, X_train, y_train, X_test, y_test)
                                    for predictor in predictors)
        for task in tasks:
            predictor, score = task
            individual_scores[predictor].append(score)

        col_list = X.columns.tolist()
        clf = get_classifier(len(col_list), n_jobs=-1)
        clf.fit(X_train[col_list].values, y_train)
        if type(clf) is SFS:
            X_train_sfs = clf.transform(X_train[col_list].values)
            X_test_sfs = clf.transform(X_test[col_list].values)
            svm = clf.estimator
            svm.fit(X_train_sfs, y_train)
            probs = get_ranks(svm, X_test_sfs)
        else:
            probs = get_ranks(clf, X_test[col_list])
        FPR, TPR, thresholds = roc_curve(y_test, probs)
        all_features_scores.append(auc(FPR, TPR))

        c_list = [x for x in col_list if '*' not in x]
        clf = get_classifier(len(c_list), n_jobs=-1)
        clf.fit(X_train[c_list].values, y_train)
        if type(clf) is SFS:
            X_train_sfs = clf.transform(X_train[c_list].values)
            X_test_sfs = clf.transform(X_test[c_list].values)
            svm = clf.estimator
            svm.fit(X_train_sfs, y_train)
            probs = get_ranks(svm, X_test_sfs)
        else:
            probs = get_ranks(clf, X_test[c_list])
        FPR, TPR, thresholds = roc_curve(y_test, probs)
        all_features_no_pairs.append(auc(FPR, TPR))

        c_list = [x for x in col_list if x.split('*')[0] not in ['max_pval', 'wi_score', 'pvalue']]
        clf = get_classifier(len(c_list), n_jobs=-1)
        clf.fit(X_train[c_list].values, y_train)
        if type(clf) is SFS:
            X_train_sfs = clf.transform(X_train[c_list].values)
            X_test_sfs = clf.transform(X_test[c_list].values)
            svm = clf.estimator
            svm.fit(X_train_sfs, y_train)
            probs = get_ranks(svm, X_test_sfs)
        else:
            probs = get_ranks(clf, X_test[c_list])
        FPR, TPR, thresholds = roc_curve(y_test, probs)
        no_wi_scores.append(auc(FPR, TPR))

        tasks = Parallel(n_jobs=-1)(delayed(train_clf_all_but_one)(col_list, predictor, X_train, y_train, X_test, y_test)
                                    for predictor in predictors)
        for task in tasks:
            predictor, score = task
            all_but_one_scores[predictor].append(score)

        c_list = [x for x in col_list if x.split('*')[0] not in predictors_dc]
        if len(c_list) > 0:
            clf = get_classifier(len(c_list), n_jobs=-1)
            clf.fit(X_train[c_list].values, y_train)
            if type(clf) is SFS:
                X_train_sfs = clf.transform(X_train[c_list].values)
                X_test_sfs = clf.transform(X_test[c_list].values)
                svm = clf.estimator
                svm.fit(X_train_sfs, y_train)
                probs = get_ranks(svm, X_test_sfs)
            else:
                probs = get_ranks(clf, X_test[c_list])
            FPR, TPR, thresholds = roc_curve(y_test, probs)
            no_dc_scores.append(auc(FPR, TPR))

        if print_non_dc_plus_single_dc:
            tasks = Parallel(n_jobs=-1)(delayed(train_clf_one_dc)(col_list, predictor, X_train, y_train, X_test, y_test)
                                        for predictor in predictors_dc)
            for task in tasks:
                predictor, score = task
                non_dc_plus_one[predictor].append(score)

    for predictor in predictors:
        scores = individual_scores[predictor]
        if len(scores) == 0:
            print(predictor + '_ind is empty')
        print(predictor + ': ' + str(np.mean(scores)) + " +/- " + str(np.std(scores)), file=f)
    print(method + ' all features: ' + str(np.mean(all_features_scores)) + " +/- " + str(np.std(all_features_scores)), file=f)
    print(method + ' all features, no pairs: ' + str(np.mean(all_features_no_pairs)) + " +/- " +
          str(np.std(all_features_no_pairs)), file=f)
    if len(all_features_scores) == 0:
        print('all is empty')
    print('without wi_score and (max_)pvalue: ' + str(np.mean(no_wi_scores)) + " +/- " + str(np.std(no_wi_scores)), file=f)
    if len(no_wi_scores) == 0:
        print('no_wi is empty')
    for predictor in predictors:
        scores = all_but_one_scores[predictor]
        if len(scores) == 0:
            print(predictor + '_all_but_one is empty')
        print('without ' + predictor + ': ' + str(np.mean(scores)) + " +/- " + str(np.std(scores)), file=f)
    print('without dc: ' + str(np.mean(no_dc_scores)) + " +/- " + str(np.std(no_dc_scores)), file=f)
    if len(no_dc_scores) == 0:
        print('no_dc is empty')
    if print_non_dc_plus_single_dc:
        for predictor in predictors_dc:
            scores = non_dc_plus_one[predictor]
            if len(scores) == 0:
                print(predictor + '_non_dc is empty')
            print(method + ' non_dc + ' + predictor + ': ' + str(np.mean(scores)) + " +/- " + str(np.std(scores)), file=f)
    print(file=f)
    print(file=f)
    f.close()


if __name__ == '__main__':
    data = read_data(mode=mode, min_dist=min_dist, norm=norm, inverse_pval=False, keep_prot_name=False, top=coef)
    print_cv(data)
