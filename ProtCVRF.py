import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

from Data import read_data
from sklearn.model_selection._validation import cross_val_score

predictors = ('fro','dca_stat','ccmpred_stat','wi_score','max_pvalue', 'site_dist', 'nmut_prod')
predictors_dc = ('fro','dca_stat','ccmpred_stat','wi_score','max_pvalue')

if __name__ == '__main__':
    mode = 'all'
    balanced = True
    plt.switch_backend('agg')
    min_dist = 6
    np.set_printoptions(linewidth=150)

    scores = read_data(mode=mode, min_dist=min_dist, norm=False, inverse_pval=False, keep_prot_name=True)
    y = scores['interact']
    del scores['interact']
    X = scores.copy()
    groups = X['prot_name']
    del X['prot_name']

    cv = LeaveOneGroupOut()

    f = open("./ProtResults.txt", "a")
    print('aa comp + dist + nmut + scores', file=f)
    for predictor in predictors:
        clf = LogisticRegression(C=0.001, penalty="l2", dual=False, class_weight='balanced', n_jobs=-1)
        cv_score = cross_val_score(clf, X[predictor].values.reshape(-1, 1), y, groups=groups, cv=cv,
                                   scoring='roc_auc')
        cv_score_mean = cv_score.mean()
        cv_score_std_dev = cv_score.std()
        print(predictor + " auc = " + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)

    print('Random Forest', file=f)
    n_estimators = 10000
    min_samples_leaf = 100
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, class_weight='balanced',
                                 min_samples_leaf=min_samples_leaf)
    print('n_estimators = ' + str(n_estimators), file=f)
    print('min_samples_leaf = ' + str(min_samples_leaf), file=f)
    cv_score = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring='roc_auc')
    cv_score_mean = cv_score.mean()
    cv_score_std_dev = cv_score.std()
    print('with wi_score: ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)
    del X['max_pvalue']
    del X['wi_score']
    cv_score = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring='roc_auc')
    cv_score_mean = cv_score.mean()
    cv_score_std_dev = cv_score.std()
    print('without wi_score and max_pvalue: ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)
    X['max_pvalue'] = scores['max_pvalue']
    X['wi_score'] = scores['wi_score']
    for predictor in predictors_dc:
        del X[predictor]
        cv_score = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring='roc_auc')
        cv_score_mean = cv_score.mean()
        cv_score_std_dev = cv_score.std()
        print('without ' + predictor + ': ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)
        X[predictor] = scores[predictor]
    for predictor in predictors_dc:
        del X[predictor]
    cv_score = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring='roc_auc')
    cv_score_mean = cv_score.mean()
    cv_score_std_dev = cv_score.std()
    print('without dc: ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)
    for predictor in predictors_dc:
        X[predictor] = scores[predictor]
        cv_score = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring='roc_auc')
        cv_score_mean = cv_score.mean()
        cv_score_std_dev = cv_score.std()
        print('only ' + predictor + ': ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)
        del X[predictor]
    print(file=f)
    f.close()

