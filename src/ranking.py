import numpy as np
from collections import defaultdict, Counter
from tools import foreach, swap


def average_precision(precision, recall):
    ap = 0
    for j in range(len(precision) - 1):
        #retourne 0 quand on a un recall de 1...
        ap += (precision[j+1] + precision[j]) * (recall[j+1] - recall[j]) / 2.0
    return ap

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
    def __init__(self, ranking_output):
        """ take the true labels as input"""
        self.ranking_output = ranking_output

    def psi(self, X, Y):
        #X : list d'images
        #Y : liste de rang
        s = 0
        for pos_ind in self.ranking_output.ind_pos:
            for neg_ind in self.ranking_output.ind_neg:
                s += self.ranking_output.M[pos_ind][neg_ind] * (X[pos_ind, :] - X[neg_ind, : ])
        return s

    def delta(self, yi=None, y=None):
        #on n'utilise pas du tout le y pour l'instant !
        return 1 - average_precision(*recall_precision(self.ranking_output))
        

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

    sortedPlus = sorted(pairsPlus, key=lambda x : x[1])
    sortedMinus = sorted(pairsMinus, key=lambda x : x[1])

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
    res = fusionList(sortedPlus, sortedMinus, imaxs)

    ranks, values = zip(*res)
    
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
    print(rank)
    print(positioning)
    precision, recall = recall_precision(ranking)
    print(precision)
    print(recall)
    print("average_precision : ", average_precision(precision, recall))

    I = RankingInstantiation(ranking)
    true_I = RankingInstantiation(true_ranking)
    x = np.array([[5, 3, 2],
                  [1, 2, 3],
                  [8, 1, 5]])
    print("psi : ", I.psi(x, y))
    print("delta : ", I.delta())
    print("true delta : ", true_I.delta())
    
    print("precision, recall : ", recall_precision(true_ranking))
