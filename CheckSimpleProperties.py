from sklearn.metrics import roc_curve, auc
import numpy as np

from Data import read_data, features, predictors

mode = 'all'
coef = 1.5
min_dist = 6
cv = 1

real = ('nmut_prod', 'nmut_sum', 'site_dist')
nominal = ('ss', 'asa')
binary = ()

def print_table(data):
    f = open("predictors_" + mode + "_min_dist_" + str(min_dist) + ".csv", "w")
    print('mode = ' + mode, file=f)
    print('CV=' + str(cv), file=f)
    print('min_dist = ' + str(min_dist), file=f)
    print('features: ' + str(features), file=f)
    s = 'feature'
    for predictor in predictors:
        s += '\t' + predictor
    print(s, file=f)
    pos = data[data['interact'] == 1]
    neg = data[data['interact'] == 0]
    samples = []
    for i in range(cv):
        neg_sample = neg.sample(n=pos.shape[0], random_state=i)
        samples.append(pos.append(neg_sample))
    for feature in real:
        auc_high = {}
        auc_low = {}
        auc_all = {}
        # median = data[feature].median()
        for predictor in predictors:
            auc_low[predictor] = []
            auc_high[predictor] = []
            auc_all[predictor] = []
        print(feature)
        for X in samples:
            print(X.shape[0])
            median = X[feature].median()
            print(median)
            for predictor in predictors:
                FPR, TPR, thresholds = roc_curve(X['interact'], X[predictor])
                auc_all[predictor].append(auc(FPR, TPR))
            X_high = X[X[feature] > median]
            print(X_high.shape[0])
            for predictor in predictors:
                FPR, TPR, thresholds = roc_curve(X_high['interact'], X_high[predictor])
                auc_high[predictor].append(auc(FPR, TPR))
            X_low = X[X[feature] <= median]
            print(X_low.shape[0])
            for predictor in predictors:
                FPR, TPR, thresholds = roc_curve(X_low['interact'], X_low[predictor])
                auc_low[predictor].append(auc(FPR, TPR))
        print(feature, file=f)
        s = 'high'
        for predictor in predictors:
            s += '\t%1.4f +/- %1.4f' % (np.mean(auc_high[predictor]), np.std(auc_high[predictor]))
        print(s, file=f)
        s = 'low'
        for predictor in predictors:
            s += '\t%1.4f +/- %1.4f' % (np.mean(auc_low[predictor]), np.std(auc_low[predictor]))
        print(s, file=f)
        s = 'all'
        for predictor in predictors:
            s += '\t%1.4f +/- %1.4f' % (np.mean(auc_all[predictor]), np.std(auc_all[predictor]))
        print(s, file=f)
    for feature in nominal:
        values = data[feature].unique()
        auc_v = {}
        value_freqs = {}
        for v in values:
            auc_v[v] = {}
            for predictor in predictors:
                auc_v[v][predictor] = []
            value_freqs[v] = data[data[feature] == v].shape[0]/data.shape[0]
        for X in samples:
            for value in values:
                X_v = X[X[feature] == value]
                for predictor in predictors:
                    FPR, TPR, thresholds = roc_curve(X_v['interact'], X_v[predictor])
                    auc_v[value][predictor].append(auc(FPR, TPR))
        print(feature, file=f)
        for value in values:
            s = value + '(%1.2f)' % value_freqs[value]
            for predictor in predictors:
                s += '\t%1.4f +/- %1.4f' % (np.mean(auc_v[value][predictor]), np.std(auc_v[value][predictor]))
            print(s, file=f)
    print()


if __name__ == '__main__':
    data = read_data(mode=mode, min_dist=min_dist, norm=False, inverse_pval=True, keep_prot_name=False, top=coef, simple_ss=True)
    print_table(data)
