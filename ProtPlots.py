import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from Data import read_data

predictors = ('fro', 'dca_stat', 'ccmpred_stat', 'wi_score', 'max_pvalue', 'site_dist', 'nmut_prod', 'nmut_sum')
predictors_dc = ('fro', 'dca_stat', 'ccmpred_stat', 'wi_score', 'max_pvalue')

mode = 'all'
coef = 1.5
min_dist = 2
n_estimators = 10000
min_samples_leaf = 100
C = 1
method = 'FSSVM'
print_method_single_dc = False
norm = True
forward = True
floating = False
cv = 10


def get_classifier(max_f_num, n_jobs=1):
    if method == 'RF':
        return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, class_weight='balanced',
                                     min_samples_leaf=min_samples_leaf)
    elif method == 'LR':
        return LogisticRegression(C=C, penalty="l2", dual=False, class_weight='balanced', n_jobs=-1)
    elif method == 'SVM':
        return LinearSVC(C=C, dual=False, class_weight='balanced')
    elif method == 'FSSVM':
        return SFS(LinearSVC(C=C, dual=False, class_weight='balanced'),
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


def train_LR(predictor, X_train, y_train, X_test, y_test):
    clf = LogisticRegression(C=0.001, penalty="l2", dual=False, class_weight='balanced', n_jobs=-1)
    clf.fit(X_train[predictor].values.reshape(-1, 1), y_train)
    return predictor, roc_curve(y_test, clf.predict_proba(X_test[predictor].values.reshape(-1, 1))[:, 1])


def train_RF_include(col_list, predictor, X_train, y_train, X_test, y_test):
    c_list = [x for x in col_list if predictor not in x]
    clf = get_classifier(len(c_list))
    clf.fit(X_train[c_list].values, y_train)
    if type(clf) is SFS:
        X_train_sfs = clf.transform(X_train.values)
        X_test_sfs = clf.transform(X_test.values)
        clf = clf.estimator
        clf.fit(X_train_sfs, y_train)
        probs = get_ranks(clf, X_test_sfs)
    else:
        probs = get_ranks(clf, X_test[c_list])
    return predictor, roc_curve(y_test, probs)


def train_RF_exclude(col_list, predictor, X_train, y_train, X_test, y_test):
    pred_list = [x for x in predictors_dc if x != predictor]
    c_list = [x for x in col_list if x.split('*')[0] not in pred_list]
    clf = get_classifier(len(c_list))
    clf.fit(X_train[c_list].values, y_train)
    if type(clf) is SFS:
        X_train_sfs = clf.transform(X_train.values)
        X_test_sfs = clf.transform(X_test.values)
        clf = clf.estimator
        clf.fit(X_train_sfs, y_train)
        probs = get_ranks(clf, X_test_sfs)
    else:
        probs = get_ranks(clf, X_test[c_list])
    return predictor, roc_curve(y_test, probs)


def train_LR_per(predictor, X_train, y_train, X_test, y_test, pair_to_keep):
    clf = LogisticRegression(C=0.001, penalty="l2", dual=False, class_weight='balanced', n_jobs=-1)
    clf.fit(X_train[predictor].values.reshape(-1, 1), y_train)
    probs = clf.predict_proba(X_test[predictor].values.reshape(-1, 1))[:, 0]
    return predictor, np.cumsum(y_test[np.argsort(probs)][:pair_to_keep])


def train_RF_include_per(col_list, predictor, X_train, y_train, X_test, y_test, pair_to_keep):
    c_list = [x for x in col_list if predictor not in x]
    clf = get_classifier(len(c_list))
    clf.fit(X_train[c_list].values, y_train)
    if type(clf) is SFS:
        X_train_sfs = clf.transform(X_train.values)
        X_test_sfs = clf.transform(X_test.values)
        clf = clf.estimator
        clf.fit(X_train_sfs, y_train)
        probs = get_ranks(clf, X_test_sfs)
    else:
        probs = get_ranks(clf, X_test[c_list])
    return predictor, np.cumsum(y_test[np.argsort(probs)][-pair_to_keep:])


def train_RF_exclude_per(col_list, predictor, X_train, y_train, X_test, y_test, pair_to_keep):

    pred_list = [x for x in predictors_dc if x != predictor]
    c_list = [x for x in col_list if x.split('*')[0] not in pred_list]
    clf = get_classifier(len(c_list))
    clf.fit(X_train[c_list].values, y_train)
    if type(clf) is SFS:
        X_train_sfs = clf.transform(X_train.values)
        X_test_sfs = clf.transform(X_test.values)
        clf = clf.estimator
        clf.fit(X_train_sfs, y_train)
        probs = get_ranks(clf, X_test_sfs)
    else:
        probs = get_ranks(clf, X_test[c_list])
    return predictor, np.cumsum(y_test[np.argsort(probs)][-pair_to_keep:])


def print_rocs(X, y):
    for prot_name in Data.prot_names:
        X_train = X[X['prot_name'] != prot_name]
        y_train = y[X['prot_name'] != prot_name]
        X_test = X[X['prot_name'] == prot_name]
        y_test = y[X['prot_name'] == prot_name]
        fig, ax = plt.subplots(figsize=(20, 10))
        plt.title('Receiver Operating Characteristic')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        tasks = Parallel(n_jobs=-1)(delayed(train_LR)(predictor, X_train, y_train, X_test, y_test)
                                    for predictor in predictors if predictor in X.columns)
        for task in tasks:
            predictor = task[0]
            FPR, TPR, thresholds = task[1]
            plt.plot(FPR, TPR, label=predictor + ' AUC = %0.2f' % auc(FPR, TPR))

        col_list = X.columns.tolist()
        col_list.remove('prot_name')
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
        plt.plot(FPR, TPR, label=method + ' AUC = %0.2f' % auc(FPR, TPR))

        c_list = [x for x in col_list if x.split('*')[0] not in ['max_pvalue', 'wi_score']]
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
        plt.plot(FPR, TPR, ls='-.', label=method + ' no wi_score and max_pvalue AUC = %0.2f' % auc(FPR, TPR))

        tasks = Parallel(n_jobs=-1)(delayed(train_RF_include)(col_list, predictor, X_train, y_train, X_test, y_test)
                                    for predictor in predictors_dc)
        for task in tasks:
            predictor = task[0]
            FPR, TPR, thresholds = task[1]
            plt.plot(FPR, TPR, ls='-.', label='without ' + predictor + ' AUC = %0.2f' % auc(FPR, TPR))

        c_list = [x for x in col_list if x.split('*')[0] not in predictors_dc]
        if len(c_list) > 0:
            clf = get_classifier(len(c_list), n_jobs=-1)
            clf.fit(X_train[c_list], y_train)
            if type(clf) is SFS:
                X_train_sfs = clf.transform(X_train[c_list].values)
                X_test_sfs = clf.transform(X_test[c_list].values)
                svm = clf.estimator
                svm.fit(X_train_sfs, y_train)
                probs = get_ranks(svm, X_test_sfs)
            else:
                probs = get_ranks(clf, X_test[col_list])
            FPR, TPR, thresholds = roc_curve(y_test, probs)
            plt.plot(FPR, TPR, ls='--', label=method + ' without dc AUC = %0.2f' % auc(FPR, TPR))

        if print_method_single_dc:
            tasks = Parallel(n_jobs=-1)(delayed(train_RF_exclude)(col_list, predictor, X_train, y_train, X_test, y_test)
                                        for predictor in predictors_dc)
            for task in tasks:
                predictor = task[0]
                FPR, TPR, thresholds = task[1]
                plt.plot(FPR, TPR, ls='--', label=method + ' only ' + predictor + ' AUC = %0.2f' % auc(FPR, TPR))

        plt.legend()
        plt.savefig("./ProtPlots/" + prot_name + '_' + method + "_auc_all_dc_only_min_dist_" + str(min_dist) + ".png")
        plt.clf()


def print_performance(X, y, coef):
    mut_num = pd.read_csv('./mitohondria/mutNum.csv', sep=';')
    for prot_name in Data.prot_names:
        mut_num_i = mut_num[prot_name]
        prot_len = len(mut_num_i[mut_num_i.notnull()])
        pair_to_keep = int(coef * prot_len)

        X_train = X[X['prot_name'] != prot_name]
        y_train = y[X['prot_name'] != prot_name]
        X_test = X[X['prot_name'] == prot_name]
        y_test = y[X['prot_name'] == prot_name].values

        fig, ax = plt.subplots(figsize=(20, 10))
        plt.title('Performance')
        plt.ylabel('True Positive')
        plt.xlabel('Rank')

        tasks = Parallel(n_jobs=-1)(delayed(train_LR_per)(predictor, X_train, y_train, X_test, y_test, pair_to_keep)
                                    for predictor in predictors if predictor in X.columns)
        for task in tasks:
            predictor, tp = task
            plt.plot(range(1, len(tp) + 1), tp, label=predictor + ' acc = %0.3f' % (tp[-1] / np.sum(y_test)))


        col_list = X.columns.tolist()
        col_list.remove('prot_name')


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
        tp = np.cumsum(y_test[np.argsort(probs)][-pair_to_keep:])
        plt.plot(range(1, len(tp) + 1), tp, label=method + ' acc =  %0.3f' % (tp[-1] / np.sum(y_test)))

        c_list = [x for x in col_list if x.split('*')[0] not in ['max_pvalue', 'wi_score']]

        clf = get_classifier(len(c_list), n_jobs=-1)
        clf.fit(X_train[c_list].values, y_train)
        if type(clf) is SFS:
            X_train_sfs = clf.transform(X_train[c_list].values)
            X_test_sfs = clf.transform(X_test[c_list].values)
            svm = clf.estimator
            svm.fit(X_train_sfs, y_train)
            probs = get_ranks(svm, X_test_sfs)
        else:
            probs = get_ranks(clf, X_test[col_list])
        tp = np.cumsum(y_test[np.argsort(probs)][-pair_to_keep:])
        plt.plot(range(1, len(tp) + 1), tp, ls='-.',
                 label=method + ' no wi_score and max_pvalue acc = %0.3f' % (tp[-1] / np.sum(y_test)))

        tasks = Parallel(n_jobs=-1)(delayed(train_RF_include_per)(col_list, predictor, X_train, y_train, X_test,
                                                                  y_test, pair_to_keep)
                                    for predictor in predictors_dc)
        for task in tasks:
            predictor, tp = task
            plt.plot(range(1, len(tp) + 1), tp, ls='-.',
                     label='without ' + predictor + ' acc = %0.3f' % (tp[-1] / np.sum(y_test)))

        c_list = [x for x in col_list if x.split('*')[0] not in predictors_dc]

        if len(c_list) > 0:
            clf = get_classifier(len(c_list), n_jobs=-1)
            clf.fit(X_train[c_list], y_train)
            if type(clf) is SFS:
                X_train_sfs = clf.transform(X_train[c_list].values)
                X_test_sfs = clf.transform(X_test[c_list].values)
                svm = clf.estimator
                svm.fit(X_train_sfs, y_train)
                probs = get_ranks(svm, X_test_sfs)
            else:
                probs = get_ranks(clf, X_test[col_list])
            tp = np.cumsum(y_test[np.argsort(probs)][-pair_to_keep:])
            plt.plot(range(1, len(tp) + 1), tp, ls='--', label=method + ' without dc acc = %0.3f' % (tp[-1] / np.sum(y_test)))

        if print_method_single_dc:
            tasks = Parallel(n_jobs=-1)(delayed(train_RF_exclude_per)(col_list, predictor, X_train, y_train, X_test,
                                                                  y_test, pair_to_keep)
                                        for predictor in predictors_dc)
            for task in tasks:
                predictor, tp = task
                plt.plot(range(1, len(tp) + 1), tp, ls='--',
                         label=method + ' only ' + predictor + ' acc = %0.3f' % (tp[-1] / np.sum(y_test)))

        plt.legend(loc='best')
        plt.savefig("./ProtPlots/" + prot_name + ' ' + method + "_per_all_dc_only_min_dist_" + str(min_dist) + ".png")
        plt.clf()
        # f.close()


if __name__ == '__main__':
    plt.switch_backend('agg')
    np.set_printoptions(linewidth=150)

    scores = read_data(mode=mode, min_dist=min_dist, norm=norm, inverse_pval=False, keep_prot_name=True, top=coef)
    y = scores['interact']
    del scores['interact']
    X = scores
    print_rocs(X, y)
    # print_performance(X, y, coef)
