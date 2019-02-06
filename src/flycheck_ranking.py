import numpy as np
from collections import defaultdict, Counter
from tools import foreach, swap, group_by, memoized
from sklearn import metrics
from sklearn.utils.fixes import signature
import matplotlib.pyplot as plt


def average_precision(precision, recall):
    return metrics.auc(recall, precision)
    # ap = 0
    # for j in range(len(precision) - 1):
    #     #retourne 0 quand on a un recall de 1...
    #     ap += (precision[j+1] + precision[j]) * (recall[j+1] - recall[j]) / 2.0
    # return ap

def plot_precision_recall_curve(precision, recall, data_type="train"):
    # ranking = RankingOutput(pred, target)
    # precision, recall = recall_precision(ranking)
    # average_precision = metrics.auc(recall, precision)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(data_type + ' Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()
    
def plot_precision_recall(prec, recall_points, baseline):
    plt.plot(recall_points, prec, label="precision recall curve")
    plt.plot(recall_points, [baseline for _ in range(len(recall_points))], label="baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve on train data")
    plt.legend()
    plt.show()

def points_precision_recall(precision, recall):
    recall_points = [i / 10 for i in range(1, 11)]
    indices = []
    i =0
    for k, rec in enumerate(recall):
        if rec > recall_points[i]:
            indices.append(k)
            i += 1
    if len(indices) != len(recall_points):
        indices.append(-1)
    assert(len(indices) == len(recall_points))
    pr10 = [precision[i] for i in indices]
    # pr10, rec10 = zip(*[(precision[i], recall[i]) for i in indices])
    rec10 = recall_points
    pr10 = [1] + list(pr10)
    rec10 = [0] + rec10
    return pr10, rec10

def recall_precision(ranking_output):
    nb_pos = len(ranking_output.ind_pos)
    labels = ranking_output.labelsGT
    ranks = ranking_output.ranks
    top = 0
    precision, recall  = [], []
    for i, (ind, rank) in enumerate(ranking_output.ord_y):
        if labels[ind] == 1.0:
            top += 1
        precision.append(top / (i + 1))
        recall.append(top / nb_pos)
    return precision, recall


class RankingOutput(object):
    def __init__(self, y, labelsGT):
        """ X : liste de sifts
        Y : liste de rangs"""
        self.ranks = y.copy()
        self.labelsGT = labelsGT
        self.ind_pos = []
        self.ind_neg = []

        for i, label in enumerate(labelsGT):
            if label == 1:
                self.ind_pos.append(i)
            if label == -1:
                self.ind_neg.append(i)

        ord_y = sorted(enumerate(y), key=lambda x : x[1])
        self.ord_y = ord_y
        img_index, _ = zip(*ord_y)
        self.positioning = dict(map(swap, enumerate(img_index)))
        #ord_y : (img_index, rank)
        # print(ord_y)
        M = np.zeros((len(y), len(y)))
        for i in range(len(ord_y)):
            head_ind, head_rank = ord_y[i]
            tail = ord_y[i+1:]
            for (ind, rank) in tail:
                M[head_ind][ind] = 2
            M[head_ind][head_ind] = 1
        M = M - 1
        self.M = M

class RankingInstantiation:
    # def __init__(self, *args):
    #     pass

    @memoized
    def psi(self, x, y):
        #X : list d'images
        #Y : liste de labels
        s = 0
        for pos_ind in y.ind_pos:
            for neg_ind in y.ind_neg:
                s += y.M[pos_ind][neg_ind] * (x[pos_ind, :] - x[neg_ind, : ])
        return s

    def delta(self, yi=None, y=None):
        """ y : labels, yi : ranking"""
        ranking_output = RankingOutput(yi, y)
        return 1 - average_precision(*recall_precision(ranking_output))
        

def val_optj(j, k, skp, sjn, nbPlus, nbMinus):
    jj = j + 1
    kk = k + 1
    val = 1 / nbPlus * ( jj / (jj + kk) - (jj - 1) / (jj + kk - 1)) - 2.0 * (skp - sjn) / (nbPlus*nbMinus)
    return val

def fusionList(l1 ,l2, indices):
    if len(l2) != len(indices):
        raise ValueError("l2 must be the same size than pos")
    res = l1.copy()
    for i in range(len(l2)):
        dec = 0
        for j in range(i):
            if indices[j] < indices[i]:
                dec += 1
        res.insert(int(indices[i] + dec), l2[i])
    return res

def loss_augmented_inference(x, w, labels):
    count = Counter(labels)
    nbPlus = count[1]
    nbMinus = count[-1]
    
    pairsPlus = []
    pairsMinus = []
    for i in range(len(x)):
        if labels[i] == 1:
            xi = x[i]
            value = w @ xi
            pairsPlus.append((i, value))
        if labels[i] == -1:
            xi = x[i]
            value = w @ xi
            pairsMinus.append((i, value))

    pairsPlus = sorted(pairsPlus, key=lambda x : x[1], reverse=True)
    pairsMinus = sorted(pairsMinus, key=lambda x : x[1], reverse=True)


    imaxs = np.zeros(nbMinus)
    for j in range(nbMinus):
        deltas = []
        for k in range(nbPlus):
            skp = pairsPlus[k][1]
            sjn = pairsMinus[j][1]
            deltaij = val_optj(j, k, skp, sjn, nbPlus, nbMinus)
            deltas.append(deltaij)
        maxI, maxV = max(enumerate(deltas), key=lambda x : x[1])
        imaxs[j] = maxI
    
    #inserting minus into the plus list at indices
    res = fusionList(pairsPlus, pairsMinus, imaxs)
    # print("max value (res) : ", max(res, key=lambda x : x[1]))
    ranks, _ = zip(*res)
    
    return RankingOutput(list(ranks), labels)
    
        
        
    
        


        

if __name__ == "__main__":
    # y = [5,3,2,1,6,8,10]
    """x : liste de représentation des images, y : un ordonancement pour lequel l’ensemble des images + est placé avant l’ensemble des images -"""
    y = [3, 2, 1]
    ranking = RankingOutput(y, [1, 1, -1])
    true_ranking = RankingOutput(y, [-1, 1, 1])
    M, rank, positioning = ranking.M, ranking.ranks, ranking.positioning
    print(M)
    assert (np.all(M == np.array([[0, -1, -1],
                                  [1, 0, -1],
                                  [1, 1, 0]])))
    # print(rank)
    # print(positioning)
    # precision, recall = recall_precision(ranking)
    # print(precision)
    # print(recall)
    # print("average_precision : ", average_precision(precision, recall))

    # I = RankingInstantiation(ranking)
    # true_I = RankingInstantiation(true_ranking)
    # x = np.array([[5, 3, 2],
    #               [1, 2, 3],
    #               [8, 1, 5]])
    # print("psi : ", I.psi(x, y))
    # print("delta : ", I.delta())
    # print("true delta : ", true_I.delta())
    
    # print("precision, recall : ", recall_precision(true_ranking))
