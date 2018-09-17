import pandas as pd
import numpy as np
from os import listdir

from os.path import isfile
from sklearn import preprocessing


prot_names = ('atp6', 'cox1', 'cox2', 'cox3', 'cytb', 'nd4', 'nd5')#
features_to_filter = ('fro', 'dca_stat', 'ccmpred_stat', 'wi_score', 'dist')
features_to_delete = ('s_sco', 'r_sco', 'sym_zscore_pairs_stat', 'MI', 'prob')
predictors = ['fro','dca_stat','ccmpred_stat','wi_score','max_pval','pvalue']#

features = []

ss_labels = ('ss_pcc', 'ss_phh', 'ss_pee', 'ss_pce', 'ss_phe', 'ss_phc')
asa_labels =('asa_pbb', 'asa_pmm', 'asa_pee', 'asa_pbm', 'asa_pbe', 'asa_pme')


def read_prot(prot_name):
    if 'pvalue' in predictors:
        return pd.read_csv('./mitohondria/unord_pairs/' + prot_name + '.csv', sep=';', decimal=',',
                           dtype={'site1': int, 'aa1': str, 'site2': int, 'aa2': str, 'MI': float, 'fro': float,
                                     'r_sco': float, 's_sco': float, 'prob': float, 'dca_stat': float,
                                     'ccmpred_stat': float,
                                     'sym_zscore_pairs_stat': float, 'wi_score': float, 'dist': float,
                                     'sq_bg_less_fg_upper_pvalue': float, 'sq_bg_greater_fg_upper_pvalue': float,
                                     'bg_nmut': float, 'fg_nmut': float, 'pvalue': float})
    else:
        return pd.read_csv('./mitohondria/' + prot_name + '.csv', sep=';', decimal=',',
                           dtype={'site1': int, 'aa1': str, 'site2': int, 'aa2': str, 'MI': float, 'fro': float,
                                  'r_sco': float, 's_sco': float, 'prob': float, 'dca_stat': float,
                                  'ccmpred_stat': float,
                                  'sym_zscore_pairs_stat': float, 'wi_score': float, 'dist': float,
                                  'sq_bg_less_fg_upper_pvalue': float, 'sq_bg_greater_fg_upper_pvalue': float,
                                  'bg_nmut': float, 'fg_nmut': float})


def read_full_dataset():
    scores = pd.DataFrame()
    for prot_name in prot_names:
        score_i = read_prot(prot_name=prot_name)
        score_i['prot_name'] = prot_name
        for feature in features_to_delete:
            if feature in score_i.columns:
                del score_i[feature]
        for column in features_to_filter:
            score_i = score_i[score_i[column].notnull()]
        scores = scores.append(score_i, ignore_index=True)
    return scores


def read_full_dataset_norm_nmut():
    mut_num = pd.read_csv('./mitohondria/mutNum.csv', sep=';')
    scores = pd.DataFrame()
    for prot_name in prot_names:
        score_i = read_prot(prot_name=prot_name)
        mut_num_i = mut_num[prot_name]
        mut_num_av = mut_num_i.mean()
        score_i['bg_nmut'] /= mut_num_av
        score_i['fg_nmut'] /= mut_num_av
        score_i['prot_name'] = prot_name
        for feature in features_to_delete:
            del score_i[feature]
        for column in features_to_filter:
            score_i = score_i[score_i[column].notnull()]
        scores = scores.append(score_i, ignore_index=True)
    return scores


def read_full_dataset_norm_nmut_tree():
    trees_lengths = pd.read_csv('./mitohondria/trees_lengths.csv', sep=';')
    scores = pd.DataFrame()
    for prot_name in prot_names:
        score_i = read_prot(prot_name=prot_name)
        tree_len = trees_lengths.loc[trees_lengths['name'] == prot_name, 'length']
        score_i['bg_nmut'] /= tree_len.values
        score_i['fg_nmut'] /= tree_len.values
        score_i['prot_name'] = prot_name
        for feature in features_to_delete:
            if feature in score_i.columns:
                del score_i[feature]
        for column in features_to_filter:
            score_i = score_i[score_i[column].notnull()]
        scores = scores.append(score_i, ignore_index=True)
    return scores


def pick_top(coef):
    mut_num = pd.read_csv('./mitohondria/mutNum.csv', sep=';')
    scores = pd.DataFrame()
    pvalue_is_present = 'pvalue' in predictors
    predictors.remove('max_pval')
    if pvalue_is_present:
        predictors.remove('pvalue')
    for prot_name in prot_names:
        score_i = read_prot(prot_name=prot_name)
        mut_num_i = mut_num[prot_name]
        pair_to_keep = int(coef*len(mut_num_i[mut_num_i.notnull()]))
        # print(prot_name + ' ' + str(pair_to_keep))
        for feature in features_to_delete:
            del score_i[feature]
        for feature in features_to_filter:
            score_i = score_i[score_i[feature].notnull()]
        score_i.reset_index(drop=True, inplace=True)
        score_i = compute_max_p_value(score_i, inverse=False)
        score_i['keep'] = 0
        for key in predictors:
            score_i.sort_values(key, ascending=False, inplace=True)
            score_i.loc[:pair_to_keep, 'keep'] = 1
        score_i.sort_values('max_pval', ascending=True)
        score_i.loc[:pair_to_keep, 'keep'] = 1
        if 'pvalue' in score_i.columns:
            score_i.sort_values('pvalue', ascending=True)
            score_i.loc[:pair_to_keep, 'keep'] = 1
        mut_num_av = mut_num_i.mean()
        score_i['bg_nmut'] /= mut_num_av
        score_i['fg_nmut'] /= mut_num_av
        score_i['prot_name'] = prot_name
        score_i = score_i[score_i['keep'] == 1]
        del score_i['keep']
        scores = scores.append(score_i, ignore_index=True)
    predictors.append('max_pval')
    if pvalue_is_present:
        predictors.append('pvalue')
    return scores


def pick_prop():
    scores = read_full_dataset_norm_nmut_tree()
    counts = scores.groupby(['prot_name'], sort=False).count()['site1']
    min_count = counts.min()
    scores_subsample = pd.DataFrame()
    for prot_name in prot_names:
        scores_subsample = scores_subsample.append(scores[scores['prot_name'] == prot_name].sample(n=min_count, random_state=1))
    return scores_subsample


def pick_positive_prop():
    scores = read_full_dataset_norm_nmut_tree()
    # print(scores.shape[0])
    for score in predictors:
        scores = scores[scores[score] > 0]
    # print(scores.shape[0])
    counts = scores.groupby(['prot_name'], sort=False).count()['site1']
    min_count = counts.min()
    scores_subsample = pd.DataFrame()
    for prot_name in prot_names:
        scores_subsample = scores_subsample.append(scores[scores['prot_name'] == prot_name].sample(n=min_count, random_state=1))
    return scores_subsample


def compute_max_p_value(scores, inverse):
    if 'max_pval' not in scores.columns:
        scores['max_pval'] = scores[['sq_bg_less_fg_upper_pvalue', 'sq_bg_greater_fg_upper_pvalue']].max(axis=1)
        if inverse:
            scores['max_pval'] = 1 - scores['max_pval']
        scores.drop(['sq_bg_less_fg_upper_pvalue', 'sq_bg_greater_fg_upper_pvalue'], inplace=True, axis=1)
    return scores


def compute_interact(scores):
    scores['interact'] = np.where(scores['dist'] < 8.0, 1, 0)
    del scores['dist']
    return scores


def compute_nmut_prod(scores):
    features.append('nmut_prod')
    scores['nmut_prod'] = scores['bg_nmut'] * scores['fg_nmut']
    # scores['nmut_prod'] = scores['nmut_prod'] / (1 + scores['nmut_prod'])
    return scores


def compute_nmut_sum(scores):
    features.append('nmut_sum')
    scores['nmut_sum'] = scores['bg_nmut'] + scores['fg_nmut']
    return scores


def compute_site_dist(scores, log=False, sqr=False):
    features.append('site_dist')
    scores['site_dist'] = abs(scores['site2'] - scores['site1'])
    # scores.drop(['site1', 'site2'], axis=1, inplace=True)
    if log:
        scores['site_dist_log'] = np.log(scores['site_dist'])
        features.append('site_dist_log')
    if sqr:
        scores['site_dist_sqr'] = scores['site_dist']**2
        features.append('site_dist_sqr')
    return scores


def compute_site_aa_comp_binarized(scores):
    features.append('AA=')
    aa_pairs = np.where(scores['aa1'] < scores['aa2'],
                                 scores['aa1'] + scores['aa2'], scores['aa2'] + scores['aa1'])
    enc = preprocessing.LabelBinarizer()
    enc.fit(aa_pairs)
    scores = scores.join(pd.DataFrame(data=enc.transform(aa_pairs), columns=['AA=' + s for s in enc.classes_], index=scores.index))
    scores.drop(['aa1', 'aa2'], axis=1, inplace=True)
    return scores, enc.classes_


# def compute_site_aa_comp(scores):
#     scores['aa_comp'] = np.where(scores['aa1'] < scores['aa2'],
#                                  scores['aa1'] + scores['aa2'], scores['aa2'] + scores['aa1'])
#     le = preprocessing.LabelEncoder()
#     scores['aa_comp'] = le.fit_transform(scores['aa_comp'])
#     scores.drop(['aa1', 'aa2'], axis=1, inplace=True)
#     return scores


def compute_pairs(scores, label):
    features.append('dc*' + label)
    for predictor in predictors:
        scores[predictor + '*' + label] = scores[predictor] * scores[label]
    return scores


def compute_pairs_list(scores, labels, label_name):
    features.append('dc*'+label_name)
    for predictor in predictors:
        for label in labels:
            scores[predictor + '*' + label] = scores[predictor] * scores[label]
    return scores


def normalize(scores):
    y = scores['interact']
    del scores['interact']
    subset = scores.select_dtypes(include=[np.float_, np.int_])
    # for column in subset.columns:
    #     print(column)
    #     print(subset[subset[column].isnull()])
    preprocessing.scale(subset, copy=False)
    scores[subset.columns] = subset
    scores['interact'] = y
    return scores


def delete_columns(scores, prot_name=True):
    scores.drop(['aa1', 'aa2', 'bg_nmut', 'fg_nmut', 'site1', 'site2'], inplace=True, axis=1, errors='ignore')
    if prot_name:
        del scores['prot_name']


def delete_site_dist(scores):
    scores.drop(['site_dist', 'site_dist_sqr', 'site_dist_log'], inplace=True, axis=1, errors='ignore')
    features.remove('site_dist')
    if 'site_dist_sqr' in features:
        features.remove('site_dist_sqr')
    if 'site_dist_log' in features:
        features.remove('site_dist_log')


def negative_to_zeros(scores, replace):
    if replace:
        for score in predictors:
            scores[score] = np.where(scores[score] > 0, scores[score], 0)
    else:
        for score in predictors:
            scores[score + '_z'] = np.where(scores[score] > 0, scores[score], 0)
    return scores


def negative_to_nan(scores):
    for score in predictors:
        scores[score + '_na'] = np.where(scores[score] > 0, scores[score], np.nan)
    return scores


def compute_ss_comp_psipred(scores):
    features.append('ss_comp')
    prot_name_to_ss = {}
    for prot_name in prot_names:
        ss = ''
        with open('./mitohondria/psipred/' + prot_name + '.psipass2', 'r') as f:
            for line in f:
                if 'Pred:' in line:
                    ss += line[6:]
        prot_name_to_ss[prot_name] = ss
    ss_pairs = prot_name_to_ss[scores['prot_name']][scores['site1']] + \
               prot_name_to_ss[scores['prot_name']][scores['site2']]
    enc = preprocessing.LabelBinarizer()
    enc.fit(ss_pairs)
    scores = scores.join(pd.DataFrame(data=enc.transform(ss_pairs), columns=enc.classes_, index=scores.index))
    return scores, enc.classes_


def compute_ss_comp_spider(scores):
    features.append('ss_')
    features.append('asa')
    prot_name_to_ss = {}
    prot_name_to_asa = {}
    with open('./mitohondria/prot_seqs.fasta') as f:
        prot_name = ''
        for line in f:
            if '>' in line:
                prot_name = line[1:len(line) - 1]
            else:
                asa = np.zeros(len(line), dtype=float)
                pc = np.zeros(len(line), dtype=float)
                pe = np.zeros(len(line), dtype=float)
                ph = np.zeros(len(line), dtype=float)
                prot_name_to_ss[prot_name] = {'pc': pc, 'pe': pe,'ph': ph}
                prot_name_to_asa[prot_name] = asa

    for prot_name in prot_names:
        with open('./mitohondria/spider2/' + prot_name + '.spider', 'r') as f:
            ss = prot_name_to_ss[prot_name]
            asa = prot_name_to_asa[prot_name]
            f.readline()
            for line in f:
                tokens = line.split()
                pos = int(tokens[0]) - 1
                ss['pc'][pos] = float(tokens[8])
                ss['pe'][pos] = float(tokens[9])
                ss['ph'][pos] = float(tokens[10])
                asa[pos] = float(tokens[3])
    for prot_name in prot_names:
        asa = prot_name_to_asa[prot_name]
        ss = prot_name_to_ss[prot_name]
        scores_prot = scores.loc[scores['prot_name'] == prot_name]
        pos1 = scores_prot['site1'] - 1
        pos2 = scores_prot['site2'] - 1
        scores.loc[scores['prot_name'] == prot_name, 'asa'] = asa[pos1]*asa[pos2]
        scores.loc[scores['prot_name'] == prot_name, 'ss_pcc'] = ss['pc'][pos1]*ss['pc'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'ss_phh'] = ss['ph'][pos1]*ss['ph'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'ss_pee'] = ss['pe'][pos1]*ss['pe'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'ss_pce'] = ss['pc'][pos1] * ss['pe'][pos2] + ss['pe'][pos1] * ss['pc'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'ss_phe'] = ss['ph'][pos1] * ss['pe'][pos2] + ss['pe'][pos1] * ss['ph'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'ss_phc'] = ss['pc'][pos1] * ss['ph'][pos2] + ss['ph'][pos1] * ss['pc'][pos2]
    return scores


def compute_ss_comp_raptor(scores):
    features.append('ss_')
    features.append('asa_')
    prot_name_to_ss = {}
    prot_name_to_asa = {}
    with open('./mitohondria/prot_seqs.fasta') as f:
        prot_name = ''
        for line in f:
            if '>' in line:
                prot_name = line[1:len(line) - 1]
            else:
                p_acc_b = np.zeros(len(line), dtype=float)
                p_acc_m = np.zeros(len(line), dtype=float)
                p_acc_e = np.zeros(len(line), dtype=float)
                pc = np.zeros(len(line), dtype=float)
                pe = np.zeros(len(line), dtype=float)
                ph = np.zeros(len(line), dtype=float)
                prot_name_to_ss[prot_name] = {'pc': pc, 'pe': pe,'ph': ph}
                prot_name_to_asa[prot_name] = {'pb': p_acc_b, 'pm': p_acc_m,'pe': p_acc_e}

    for prot_name in prot_names:
        for f in listdir('./mitohondria/raptor/' + prot_name):
            if isfile('./mitohondria/raptor/' + prot_name + '/' + f):
                if f.endswith('.acc.txt'):
                    # print('reading ' + f)
                    with open('./mitohondria/raptor/' + prot_name + '/' + f, 'r') as f:
                        asa = prot_name_to_asa[prot_name]
                        f.readline()
                        f.readline()
                        f.readline()
                        for line in f:
                            tokens = line.split()
                            pos = int(tokens[0]) - 1
                            asa['pb'][pos] = float(tokens[3])
                            asa['pm'][pos] = float(tokens[4])
                            asa['pe'][pos] = float(tokens[5])
                elif f.endswith('.ss3.txt'):
                    # print('reading ' + f)
                    with open('./mitohondria/raptor/' + prot_name + '/' + f, 'r') as f:
                        ss = prot_name_to_ss[prot_name]
                        f.readline()
                        f.readline()
                        for line in f:
                            tokens = line.split()
                            pos = int(tokens[0]) - 1
                            ss['ph'][pos] = float(tokens[3])
                            ss['pe'][pos] = float(tokens[4])
                            ss['pc'][pos] = float(tokens[5])
    for prot_name in prot_names:
        asa = prot_name_to_asa[prot_name]
        ss = prot_name_to_ss[prot_name]
        scores_prot = scores.loc[scores['prot_name'] == prot_name]
        pos1 = scores_prot['site1'] - 1
        pos2 = scores_prot['site2'] - 1
        scores.loc[scores['prot_name'] == prot_name, 'asa_pbb'] = asa['pb'][pos1] * asa['pb'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'asa_pmm'] = asa['pm'][pos1] * asa['pm'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'asa_pee'] = asa['pe'][pos1] * asa['pe'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'asa_pbm'] = asa['pb'][pos1] * asa['pm'][pos2] + \
                                                                       asa['pm'][pos1] * asa['pb'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'asa_pbe'] = asa['pb'][pos1] * asa['pe'][pos2] + \
                                                                       asa['pe'][pos1] * asa['pb'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'asa_pme'] = asa['pe'][pos1] * asa['pm'][pos2] + \
                                                                       asa['pm'][pos1] * asa['pe'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'ss_pcc'] = ss['pc'][pos1] * ss['pc'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'ss_phh'] = ss['ph'][pos1] * ss['ph'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'ss_pee'] = ss['pe'][pos1] * ss['pe'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'ss_pce'] = ss['pc'][pos1] * ss['pe'][pos2] + \
                                                                 ss['pe'][pos1] * ss['pc'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'ss_phe'] = ss['ph'][pos1] * ss['pe'][pos2] + \
                                                                 ss['pe'][pos1] * ss['ph'][pos2]
        scores.loc[scores['prot_name'] == prot_name, 'ss_phc'] = ss['pc'][pos1] * ss['ph'][pos2] + \
                                                                 ss['ph'][pos1] * ss['pc'][pos2]
    return scores


def compute_ss_comp_raptor_simple(scores):
    features.append('ss')
    features.append('asa')
    prot_name_to_ss = {}
    prot_name_to_asa = {}

    for prot_name in prot_names:
        for f in listdir('./mitohondria/raptor/' + prot_name):
            if isfile('./mitohondria/raptor/' + prot_name + '/' + f):
                if f.endswith('.acc_simp.txt'):
                    # print('reading ' + f)
                    with open('./mitohondria/raptor/' + prot_name + '/' + f, 'r') as f:
                        f.readline()
                        f.readline()
                        prot_name_to_asa[prot_name] = np.array(list(f.readline()))
                elif f.endswith('.ss3_simp.txt'):
                    # print('reading ' + f)
                    with open('./mitohondria/raptor/' + prot_name + '/' + f, 'r') as f:
                        f.readline()
                        f.readline()
                        prot_name_to_ss[prot_name] = np.array(list(f.readline()))
    for prot_name in prot_names:
        asa = prot_name_to_asa[prot_name]
        ss = prot_name_to_ss[prot_name]

        scores_prot = scores.loc[scores['prot_name'] == prot_name]
        pos1 = scores_prot['site1'].values - 1
        pos2 = scores_prot['site2'].values - 1
        scores.loc[scores['prot_name'] == prot_name, 'asa1'] = asa[pos1]
        scores.loc[scores['prot_name'] == prot_name, 'asa2'] = asa[pos2]
        scores.loc[scores['prot_name'] == prot_name, 'ss1'] = ss[pos1]
        scores.loc[scores['prot_name'] == prot_name, 'ss2'] = ss[pos2]
    scores['asa'] = np.where(scores['asa1'] < scores['asa2'],
                                 scores['asa1'] + scores['asa2'], scores['asa2'] + scores['asa1'])
    scores['ss'] = np.where(scores['ss1'] < scores['ss2'],
                                 scores['ss1'] + scores['ss2'], scores['ss2'] + scores['ss1'])
    scores.drop(['asa1', 'asa2', 'ss1', 'ss2'], inplace=True, axis=1)
    return scores


def read_data(mode, min_dist, norm, inverse_pval, keep_prot_name=False, top=10, simple_ss=False):
    if mode == 'prop':
        scores = pick_prop()
    elif mode == 'all':
        scores = read_full_dataset()
    elif mode == 'top':
        scores = pick_top(top)
    elif mode == 'positive':
        scores = pick_positive_prop()
    else:
        raise ValueError('Undefined mode: {}'.format(mode))
    # scores.reset_index(drop=True, inplace=True)
    scores = compute_site_dist(scores)
    scores = scores[scores['site_dist'] >= min_dist]
    scores.reset_index(drop=True, inplace=True)
    if inverse_pval and 'pvalue' in predictors:
        scores['pvalue'] = 1 - scores['pvalue']
    scores = compute_max_p_value(scores, inverse=inverse_pval)
    scores = compute_interact(scores)
    scores = compute_nmut_prod(scores)
    scores = compute_nmut_sum(scores)
    # scores, aa_labels = compute_site_aa_comp_binarized(scores)
    if simple_ss:
        scores = compute_ss_comp_raptor_simple(scores)
    else:
        scores = compute_ss_comp_raptor(scores)
    delete_columns(scores, prot_name=not keep_prot_name)
    if norm:
        scores = normalize(scores)
    # scores = compute_pairs(scores, 'nmut_prod')
    # scores = compute_pairs(scores, 'nmut_sum')
    # scores = compute_pairs(scores, 'site_dist')
    # scores = compute_pairs_list(scores, asa_labels, 'asa')
    # scores = compute_pairs_list(scores, ss_labels, 'ss')
    # scores = compute_pairs_list(scores, aa_labels)

    # scores = negative_to_zeros(scores, replace=True)

    if norm:
        scores = normalize(scores)

    # delete_site_dist(scores)

    return scores


if __name__ == '__main__':
    mode = 'all'
    top = 10

    np.set_printoptions(linewidth=150)

    # scores = pd.read_csv('./data/scores' + mode + '_norm.csv', sep=';', dtype=float)

    # print(scores.columns.values)
    # print(np.corrcoef(scores, rowvar=False))

    scores = read_data(mode, min_dist=6, norm=False, inverse_pval=False, top=top, simple_ss=True)
    #
    # print(scores['ss_pcc'])
    # print(scores['asa_pbb'])
    # print(scores[scores['interact'] == 1].shape[0])
    # scores[['asa', 'ss']].to_csv('./test.csv', sep=';', index=False)