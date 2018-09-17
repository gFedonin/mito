from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import Data as d

predictors = ('fro','dca_stat','ccmpred_stat','wi_score','max_pvalue','site_dist','nmut_prod','nmut_sum')
predictors_dc = ('fro','dca_stat','ccmpred_stat','wi_score','max_pvalue')

if __name__ == '__main__':
    mode = 'prop'
    min_dist = 6
    cv = 10
    n_estimators = 10000
    min_samples_leaf = 100
    oob_score = False
    scores = d.read_data(mode=mode, min_dist=min_dist, norm=False, inverse_pval=False)
    y = scores['interact']
    del scores['interact']
    X = scores.copy()
    # clf = clf.fit(X, y)
    f = open("./RandomForest.txt", "a")
    print('with aa comp, mode = ' + str(mode), file=f)
    for predictor in predictors:
        clf = LogisticRegression(C=0.001, penalty="l2", dual=False, class_weight='balanced', n_jobs=-1)
        cv_score = cross_val_score(clf, scores[predictor].values.reshape(-1, 1), y, cv=cv, scoring='roc_auc')
        cv_score_mean = cv_score.mean()
        cv_score_std_dev = cv_score.std()
        print(predictor + " auc = " + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)
    print('random forest')
    clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, class_weight='balanced',
                                 min_samples_leaf=min_samples_leaf, oob_score=oob_score)
    print('n_estimators = ' + str(n_estimators), file=f)
    print('min_samples_leaf = ' + str(min_samples_leaf), file=f)
    print('oob_score = ' + str(oob_score), file=f)
    cv_score = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    cv_score_mean = cv_score.mean()
    cv_score_std_dev = cv_score.std()
    print('with wi_score: ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)
    del X['max_pvalue']
    del X['wi_score']
    cv_score = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    cv_score_mean = cv_score.mean()
    cv_score_std_dev = cv_score.std()
    print('without wi_score and max_pvalue: ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)
    X['max_pvalue'] = scores['max_pvalue']
    X['wi_score'] = scores['wi_score']
    for predictor in predictors_dc:
        del X[predictor]
        cv_score = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
        cv_score_mean = cv_score.mean()
        cv_score_std_dev = cv_score.std()
        print('without ' + predictor + ': ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)
        X[predictor] = scores[predictor]
    for predictor in predictors_dc:
        del X[predictor]
    cv_score = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    cv_score_mean = cv_score.mean()
    cv_score_std_dev = cv_score.std()
    print('without dc: ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)
    for predictor in predictors_dc:
        X[predictor] = scores[predictor]
        cv_score = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
        cv_score_mean = cv_score.mean()
        cv_score_std_dev = cv_score.std()
        print('only ' + predictor + ': ' + str(cv_score_mean) + " +/- " + str(cv_score_std_dev), file=f)
        del X[predictor]
    print(file=f)
    f.close()
