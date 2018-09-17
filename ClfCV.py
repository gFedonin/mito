from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import delayed, Parallel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import Data as d

mode = 'prop'
coef = 1.5
min_dist = 6
n_estimators = 10000
min_weight_fraction_leaf = 0.01
oob_score = False
C = 1
method = 'RF'
print_method_single = True
norm = False
forward = True
floating = False
cv = 10

predictors = ('fro', 'dca_stat', 'ccmpred_stat', 'wi_score', 'max_pval', 'site_dist', 'nmut_prod', 'nmut_sum',
              'ss_', 'asa_', 'AA=', 'pvalue')#
predictors_dc = ('fro', 'dca_stat', 'ccmpred_stat', 'wi_score', 'max_pval', 'pvalue')
predictors_not_dc = ('site_dist', 'nmut_prod', 'nmut_sum', 'ss_', 'asa_', 'AA=')
predictors_no_wi = ('fro', 'dca_stat', 'ccmpred_stat', 'site_dist', 'nmut_prod', 'nmut_sum',
              'ss_', 'asa_', 'AA=')


def get_classifier(max_f_num, n_jobs=1):
    if method == 'RF':
        return RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, class_weight='balanced',
                                      min_weight_fraction_leaf=min_weight_fraction_leaf)
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


def train_LR(predictor, X, y):
    clf = LinearSVC(C=C, dual=False)
    if predictor == 'ss_' or predictor == 'AA=' or predictor == 'asa_':
        c_list = [x for x in X.columns if predictor in x and '*' not in x] # with predictor but no pairs
        cv_score = cross_val_score(clf, X[c_list], y, cv=cv, scoring='roc_auc')
    else:
        cv_score = cross_val_score(clf, X[predictor].values.reshape(-1, 1), y, cv=cv, scoring='roc_auc')
    cv_score_mean = cv_score.mean()
    cv_score_std_dev = cv_score.std()
    return predictor, cv_score_mean, cv_score_std_dev


def train_clf_all_but_one(col_list, predictor, X, y):
    c_list = [x for x in col_list if predictor not in x]
    clf = get_classifier(len(c_list))
    cv_score = cross_val_score(clf, X[c_list], y, cv=cv, scoring='roc_auc')
    cv_score_mean = cv_score.mean()
    cv_score_std_dev = cv_score.std()
    return predictor, cv_score_mean, cv_score_std_dev


def train_clf_one_dc(col_list, predictor, X, y):

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
    cv_score = cross_val_score(clf, X[c_list], y, cv=cv, scoring='roc_auc')
    cv_score_mean = cv_score.mean()
    cv_score_std_dev = cv_score.std()
    return predictor, cv_score_mean, cv_score_std_dev


if __name__ == '__main__':
    scores = d.read_data(mode=mode, min_dist=min_dist, norm=False, inverse_pval=False)
    y = scores['interact']
    del scores['interact']
    X = scores
    # clf = clf.fit(X, y)
    f = open("contacts_CV_" + mode + "_min_dist_" + str(min_dist) + ".txt", "a")
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
    print('features: ' + str(d.features), file=f)

    tasks = Parallel(n_jobs=-1)(delayed(train_LR)(predictor, X, y) for predictor in predictors)
    for task in tasks:
        predictor, cv_score_mean, cv_score_std_dev = task
        print(predictor + ': ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)

    col_list = X.columns.tolist()
    clf = get_classifier(max_f_num=len(col_list), n_jobs=-1)
    cv_score = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    cv_score_mean = cv_score.mean()
    cv_score_std_dev = cv_score.std()
    print(method + ' all features: ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)

    c_list = [x for x in col_list if x.split('*')[0] not in ['max_pval', 'wi_score', 'pvalue']]
    cv_score = cross_val_score(clf, X[c_list], y, cv=cv, scoring='roc_auc')
    cv_score_mean = cv_score.mean()
    cv_score_std_dev = cv_score.std()
    print('without wi_score and (max_)pvalue: ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)

    tasks = Parallel(n_jobs=-1)(delayed(train_clf_all_but_one)(col_list, predictor, X, y)
                                for predictor in predictors_no_wi)
    for task in tasks:
        predictor, cv_score_mean, cv_score_std_dev = task
        print('without ' + predictor + ': ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)

    c_list = [x for x in col_list if x.split('*')[0] not in predictors_dc]
    if len(c_list) > 0:
        cv_score = cross_val_score(clf, X[c_list], y, cv=cv, scoring='roc_auc')
        cv_score_mean = cv_score.mean()
        cv_score_std_dev = cv_score.std()
        print('without dc: ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)

    if print_method_single:
        tasks = Parallel(n_jobs=-1)(delayed(train_clf_one_dc)(col_list, predictor, X, y)
                                    for predictor in predictors_dc)
        for task in tasks:
            predictor, cv_score_mean, cv_score_std_dev = task
            print(method + ' non_dc + ' + predictor + ': ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)

    print(file=f)
    f.close()
