#from path import Path
from imageNetParser import get_features, get_classes_image_net
from index import *
import matplotlib.pyplot as plt
import model
from sklearn.decomposition import PCA
from tools import accuracy, split, foreach, random_rank, random_labels
from sklearn.metrics import confusion_matrix
from itertools import product
from IStructInstantiation import *
import numpy as np
from ranking import *


def main_ranking():
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

    m = model.RankingStructModel(250)
    dataset = load()
    x, y = sample(dataset, 5, train=True)
    print(x.shape, y.shape)
    L = split(x)
    foreach(lambda x : print(x.shape), L)
    y = random_labels(len(L))
    print(y)

    pred = m.predict(L, y)
    print(pred)
    M, rank, positioning = pred.M, pred.ranks, pred.positioning
    print("M : ", M)
    print("rank : ", rank)
    print("positioning :", positioning)

    print("loss : ", m.loss(pred))

    # loss = m.loss(L, y)
    # print("loss : ", loss)
    # M, rank, positioning = loss.M, loss.ranks, loss.positioning
    # print("M : ", M)
    # print("rank : ", rank)
    # print("positioning :", positioning)

def main_classif():
    DATA = "../data"
    #directory = Path(DATA)
    features = []
    fname = DATA + "/taxi.txt"
    
    # feat = get_features(DATA + "/taxi.txt")
    # for f in directory.files("*.txt"):
    #     feat = get_features(f)
    #     features.append(feat)
    
    #bow = compute_bow(feat[0])
    # print("bow : ", bow)
    #plot_histo(bow)
    # dataset = get_dataset(DATA, get_classes_image_net())
    # save(dataset)
    # foreach(lambda x : print(x.shape), dataset)
    
    dataset = load()
    dimpsi = 250 * 9
    # dimpsi = 250
    
    x,y = sample(dataset, 1, train=False)
    y = y[0]
    print(x.shape, y.shape)
    

    linear = model.LinearStructModel(dimpsi)
    linear.instantiation()
    print(linear.predict(x))
    print(linear.lai(x, y))

    mc = MultiClass()
    print("psi shape : ", mc.psi(x, y).shape)

    # classes = get_classes_image_net()
    # mat = distance_normalisation(get_image_net_distances())
    # print("mat : ", mat)

    nbit=100
    classifier = model.GenericTrainingAlgorithm(dimpsi)

    # classifier = model.GenericTrainingAlgorithm(dimpsi, struct_classe=MultiClassFier, kwargs={"mat" : mat})
    # # print(accuracy(classifier, dataset))
    L = classifier.fit(dataset, nb_it=nbit, register=True, alpha=1e-6, lr=1e-2)
    train_acc, test_acc = zip(*L)
    print(accuracy(classifier, dataset))
    plt.plot(list(range(nbit)), train_acc, label="train")
    plt.plot(list(range(nbit)), test_acc, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    

    # #test
    # classes = get_classes_image_net()
    # pred = [classifier.predict(x) for x in dataset.x_test]
    # mat = confusion_matrix(dataset.y_test, pred)
    # plt.imshow(mat)
    # plt.xticks(range(9), classes, rotation=90)
    # plt.yticks(range(9), classes)
    # plt.colorbar()
    # plt.show()

    # #train
    # classes = get_classes_image_net()
    # pred = [classifier.predict(x) for x in dataset.x_train]
    # mat = confusion_matrix(dataset.y_train, pred)
    # plt.imshow(mat)
    # plt.xticks(range(9), classes, rotation=90)
    # plt.yticks(range(9), classes)
    # plt.colorbar()
    # plt.show()
    
    #save(dataset)
    # dataset = load()
    # print([x.shape for x in dataset])
    
if __name__ == "__main__":
    # main_classif()
    main_ranking()
