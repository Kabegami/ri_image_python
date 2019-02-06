from imageNetParser import get_features, get_classes_image_net
from index import *
import matplotlib.pyplot as plt
import model
from sklearn.decomposition import PCA
from tools import accuracy, split, foreach, random_rank, random_labels, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from itertools import product
from IStructInstantiation import *
import numpy as np
from ranking import *
import begin

@begin.start
def main(cls=1, nb_it=50):
    cls, nb_it = int(cls), int(nb_it)
    classes = get_classes_image_net()
    print("classe évalué : ", classes[cls])
    dimpsi = 250

    dataset = load()
    dataset = convertClassif2Ranking(dataset, cls=cls)
    x_train, y_train = dataset.x_train, dataset.y_train
    x_test, y_test = dataset.x_test, dataset.y_test


    classifier = model.GenericTrainingAlgorithm(dimpsi, struct_classe=RankingInstantiation, classe=model.RankingStructModel)
    # classifier.fit_ranking(dataset, nb_it=nb_it, alpha=1e-6, lr=10)
    classifier.fit_ranking(dataset, nb_it=nb_it, lr=1e-6, alpha=10)

    name = "../res/model_{}.bin".format(classes[cls])
    classifier.save(name)


