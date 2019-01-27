import numpy as np
from collections import defaultdict
from tools import foreach, swap

""" X : liste de sifts
    Y : liste de rangs"""

class RankingOutput(object):
    def __init__(self, x, y, labelsGT):
        self.x = x
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
        print(ord_y)
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
        self.ranking_output = ranking_output

    def psi(self, X, Y):
        #X : list d'images
        #Y : liste de rang
        s = 0
        for pos_ind in self.ranking_output.ind_pos:
            for neg_ind in self.ranking_output.ind_neg:
                s += self.ranking_output.M[i][j] * (X[i] - X[j])
        return psi
        
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
        print("ord_y : ", ranking_output.ord_y)
        print("labels[ind] : ", labels[ind])
        if labels[ind] == 1.0:
            top += 1
        print("top : ", top)
        precision.append(top / (i + 1))
        recall.append(top / nb_pos)
    return precision, recall

    
        


        

if __name__ == "__main__":
    # y = [5,3,2,1,6,8,10]
    y = [3, 2, 1]
    ranking = RankingOutput(None, y, [1, 1, -1])
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
