#from path import Path
from imageNetParser import get_features, get_classes_image_net
from index import *
import matplotlib.pyplot as plt
import model
from sklearn.decomposition import PCA
from tools import accuracy
from sklearn.metrics import confusion_matrix
from itertools import product
from IStructInstantiation import *
import numpy as np

def foreach(f, it):
    for e in it:
        f(e)

if __name__ == "__main__":
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
    
    x,y = sample(dataset, 1, train=False)
    y = y[0]
    print(x.shape, y.shape)
    
    linear = model.LinearStructModel(dimpsi)
    linear.instantiation()
    print(linear.predict(x))
    print(linear.lai(x,y))

    classes = get_classes_image_net()
    mat = distance_normalisation(get_image_net_distances())
    print("mat : ", mat)

    # mat = np.zeros((len(classes), len(classes)))
    # for i1, c1 in enumerate(classes):
    #     for i2, c2 in enumerate(classes):
    #         list_w1, list_w2 = wn.synsets(c1), wn.synsets(c2)
    #         similarities = (w1.wup_similarity(w2) for w1, w2 in product(list_w1, list_w2))
    #         similarity = max(filter(lambda x : x is not None, similarities))
    #         mat[i1][i2] = similarity            
    
    # # for (i1, c1), (i2, c2) in (product(zip(enumerate(classes), enumerate(classes)))):
    # #     print("c1 : ", c1, "c2 : ", c2)

        
    # print(mat)
        

    nbit=100
    classifier = model.GenericTrainingAlgorithm(dimpsi, struct_classe=MultiClassFier, kwargs={"mat" : mat})
    # print(accuracy(classifier, dataset))
    L = classifier.fit(dataset, nb_it=nbit, register=True, alpha=1e-6, lr=1e-2)
    train_acc, test_acc = zip(*L)
    print(accuracy(classifier, dataset))
    plt.plot(list(range(nbit)), train_acc, label="train")
    plt.plot(list(range(nbit)), test_acc, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
    

    #test
    classes = get_classes_image_net()
    pred = [classifier.predict(x) for x in dataset.x_test]
    mat = confusion_matrix(dataset.y_test, pred)
    plt.imshow(mat)
    plt.xticks(range(9), classes, rotation=90)
    plt.yticks(range(9), classes)
    plt.colorbar()
    plt.show()

    #train
    classes = get_classes_image_net()
    pred = [classifier.predict(x) for x in dataset.x_train]
    mat = confusion_matrix(dataset.y_train, pred)
    plt.imshow(mat)
    plt.xticks(range(9), classes, rotation=90)
    plt.yticks(range(9), classes)
    plt.colorbar()
    plt.show()
    
    #save(dataset)
    # dataset = load()
    # print([x.shape for x in dataset])
    
    



