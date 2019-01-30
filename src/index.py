from imageNetParser import get_dico_size, get_features, get_classes_image_net
import numpy as np
from collections import namedtuple
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
import pickle
from itertools import product
import matplotlib.pyplot as plt
from nltk.corpus import wordnet as wn

Dataset = namedtuple("Dataset", ["x_train", "x_test", "y_train", "y_test"])
TrainingSample = namedtuple("TrainingSample", ["x", "y"])

def sample(dataset, size=1, train=True):
    if train:
        ind = np.random.randint(0, len(dataset.x_train), size)  
        return dataset.x_train[ind], dataset.y_train[ind]
    else:
        ind = np.random.randint(0, len(dataset.x_test), size)  
        return dataset.x_test[ind], dataset.y_test[ind]

def compute_bow(feature_list):
    v = np.zeros(get_dico_size())
    for w in feature_list.get_words():
        v[w] += 1
    norm2 = np.linalg.norm(v, ord=2)
    v /= norm2
    return v

def file_to_dataset(fname, label):
    features = get_features(fname)
    X = [compute_bow(feat) for feat in features]
    Y = np.array([label for _ in range(len(features))])
    return X, Y, len(features)

def get_dataset(PATH, class_list):
    X, Y, listN = [], [], []
    for label, cls in enumerate(class_list):
        fname = PATH + "/{}.txt".format(cls)
        xi, yi, n = file_to_dataset(fname, label)
        X.append(xi), Y.append(yi), listN.append(n)
        
    print("to numpy")

    cat = lambda x : np.concatenate(x, axis=0)
    
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)

    # print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)



    print("pca")
    X = PCA(n_components = 250).fit_transform(X)


    print("split train test")
    x_train, x_test, y_train, y_test = [[] for i in range(4)]
    acc = 0
    for n in listN:
        x_train.append(X[acc:acc+800]), x_test.append(X[acc+800: acc + n])
        y_train.append(Y[acc:acc+800]), y_test.append(Y[acc+800: acc+ n])
        acc += n

    x_train, x_test, y_train, y_test = list(map(cat, (x_train, x_test, y_train, y_test)))
        
    print("shuffle")
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)
        
    print("done !")
    return Dataset(x_train, x_test, y_train, y_test)
 
def plot_histo(bow):
    
    bar_width = 5. # set this to whatever you want
    positions = np.arange(1000)
    plt.bar(positions, bow, bar_width)
    #plt.xticks(positions + bar_width / 2, ('0', '1', '2', '3'))
    plt.show()
def save(obj, fname="../res/dataset.bin"):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load(fname="../res/dataset.bin"):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data

def get_image_net_distances():
    classes = get_classes_image_net()
    mat = np.zeros((len(classes), len(classes)))
    for i1, c1 in enumerate(classes):
        for i2, c2 in enumerate(classes):
            list_w1, list_w2 = wn.synsets(c1), wn.synsets(c2)
            similarities = (w1.wup_similarity(w2) for w1, w2 in product(list_w1, list_w2))
            similarity = max(filter(lambda x : x is not None, similarities))
            mat[i1][i2] = similarity         
    return 1 - mat
    
def distance_normalisation(mat):
    return mat * 1.9 + 0.1

def convertClassif2Ranking(dataset, cls=1):
    x_train, x_test, y_train, y_test = dataset
    y_train = ((y_train == cls) * 2 - 1).astype(int)
    y_test = ((y_test == cls) * 2 - 1).astype(int)
    return Dataset(x_train, x_test, y_train, y_test)
    
    

