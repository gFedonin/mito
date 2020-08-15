from os import listdir
from os.path import isdir

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

path_to_data = '../res/for_reviewer/G10/'
dist_threshold = 4.0
labels = ['a', 'b', 'c', 'd', 'e']


def print_roc():
    j = 0
    fig, axs = plt.subplots(2, 3)
    for ax in axs.flat:
        ax.set(xlabel='False positive rate', ylabel='True positive rate')
    for ax in axs.flat:
        ax.label_outer()
    # fig.delaxes(axs[1, 2])
    for fname in listdir(path_to_data):
        if not isdir(path_to_data + fname):
            continue
        our_scores = []
        our_cont = []
        our_pairs = set()
        i = fname.find('.')
        if i != -1:
            prot_name = fname[:i]
        else:
            prot_name = fname
        for l in open(path_to_data + fname + '/' + prot_name + '.pos_cor2pcor.l90.n0.all_pos.tab').readlines()[1:]:
            s = l.strip().split('\t')
            if len(s) < 6:
                continue
            our_scores.append(float(s[3]))
            our_pairs.add(s[0] + '\t' + s[1])
            if float(s[-1]) < dist_threshold:
                our_cont.append(1)
            else:
                our_cont.append(-1)

        ccmpred_scores = []
        ccmpred_cont = []
        for l in open(path_to_data + fname + '/' + prot_name + '.G10.ccmpred.tab').readlines()[1:]:
            s = l.strip().split('\t')
            if len(s) < 5:
                continue
            ccmpred_scores.append(float(s[2]))
            if s[0] + '\t' + s[1] not in our_pairs:
                our_scores.append(-100000.0)
                if float(s[-1]) < dist_threshold:
                    our_cont.append(1)
                else:
                    our_cont.append(-1)
            if float(s[-1]) < dist_threshold:
                ccmpred_cont.append(1)
            else:
                ccmpred_cont.append(-1)
        fpr_our, tpr_our, _ = roc_curve(our_cont, our_scores, pos_label=1)
        fpr_ccmpred, tpr_ccmpred, _ = roc_curve(ccmpred_cont, ccmpred_scores, pos_label=1)
        our_auc = roc_auc_score(our_cont, our_scores)
        ccmpred_auc = roc_auc_score(ccmpred_cont, ccmpred_scores)
        ax = axs[j // 3, j % 3]
        ax.plot([0, 1], [0, 1], 'k--')
        ax.plot(fpr_our, tpr_our, label='AUC=%1.2f' % (our_auc))
        ax.plot(fpr_ccmpred, tpr_ccmpred, label='AUC=%1.2f' % (ccmpred_auc))
        ax.set_title(labels[j] + '. ' + prot_name.upper())
        ax.legend(loc='best')
        j += 1
    # handles, labels = axs[0,0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center right')
    axs[1,2].plot([], [], label='epistat7.1')
    axs[1,2].plot([], [], label='ccmpred')
    axs[1,2].legend(loc='center')
    axs[1,2].patch.set_visible(False)
    axs[1,2].axis('off')
    # axs[1, 2].set_visible(False)
    fig.savefig(path_to_data + 'fig_S.png', dpi=400)


def print_tp_top():
    j = 0
    fig, axs = plt.subplots(2, 3)
    for ax in axs.flat:
        ax.set(xlabel='# of selected pairs', ylabel='Precision')
    # for ax in axs.flat:
    #     ax.label_outer()
    # fig.delaxes(axs[1, 2])
    for fname in listdir(path_to_data):
        if not isdir(path_to_data + fname):
            continue
        our_pairs = []
        i = fname.find('.')
        if i != -1:
            prot_name = fname[:i]
        else:
            prot_name = fname
        for l in open(path_to_data + fname + '/' + prot_name + '.pos_cor2pcor.l90.n0.all_pos.tab').readlines()[1:]:
            s = l.strip().split('\t')
            if len(s) < 6:
                continue
            if float(s[-1]) < dist_threshold:
                our_pairs.append((float(s[3]), 1))
            else:
                our_pairs.append((float(s[3]), -1))
        ccmpred_pairs = []
        for l in open(path_to_data + fname + '/' + prot_name + '.G10.ccmpred.2l.tab').readlines()[1:]:
            s = l.strip().split('\t')
            if len(s) < 5:
                continue
            if float(s[-1]) < dist_threshold:
                ccmpred_pairs.append((float(s[2]), 1))
            else:
                ccmpred_pairs.append((float(s[2]), -1))
        our_pairs.sort(key=lambda x: -x[0])
        ccmpred_pairs.sort(key=lambda x: -x[0])
        top_n = range(10, len(ccmpred_pairs))
        our_tp_rate = []
        ccmpred_tp_rate = []
        for i in top_n:
            c = 0
            for k in range(i):
                if our_pairs[k][1] == 1:
                    c += 1
            our_tp_rate.append(c/i)
            c = 0
            for k in range(i):
                if ccmpred_pairs[k][1] == 1:
                    c += 1
            ccmpred_tp_rate.append(c/i)
        ax = axs[j // 3, j % 3]
        # ax.plot([0, 1], [0, 1], 'k--')
        ax.plot(top_n, our_tp_rate, label='epistat7.1')
        ax.plot(top_n, ccmpred_tp_rate, label='ccmpred')
        ax.set_title(labels[j] + '. ' + prot_name.upper())
        # ax.legend(loc='best')
        j += 1
    # handles, labels = axs[0,0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='center right')
    axs[1,2].plot([], [], label='epistat7.1')
    axs[1,2].plot([], [], label='ccmpred')
    axs[1,2].legend(loc='center')
    axs[1,2].patch.set_visible(False)
    axs[1,2].axis('off')
    # axs[1, 2].set_visible(False)
    plt.tight_layout()
    fig.savefig(path_to_data + 'fig_S1.png', dpi=400)


if __name__ == '__main__':
    print_tp_top()
    print_roc()