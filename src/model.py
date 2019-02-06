import numpy as np
from IStructInstantiation import *
from tqdm import tqdm
from tools import accuracy, accuracy_, map2
from ranking import *
import pickle

class LinearStructModel:
    
    def __init__(self, dimpsi):
        # self.w = np.random.randn(dimpsi)
        self.w = np.zeros(dimpsi)
        
    def predict(self, x):
        return np.argmax([self.mc.psi(x,y).dot(self.w) for y in range(9)])
    
    def lai(self, xi, yi):
        return np.argmax([self.mc.delta(yi, y)+self.mc.psi(xi, y).dot(self.w) for y in range(9)])
    
    def loss(self, xi, yi):
        return np.max([self.mc.delta(yi, y)+self.mc.psi(xi, y).dot(self.w) for y in range(9)])
    
    def instantiation(self, classe=MultiClass, kwargs={}):
        self.mc = classe(**kwargs)
    
    def getParameters(self):
        return self.w

    def save(self, fname="../res/model.bin"):
        with open(fname, "wb") as f:
            pickle.dump(self.w, f)

    def load(self, fname="../res/model.bin"):
        with open(fname, "rb") as f:
            self.w = pickle.load(f)


class RankingStructModel(LinearStructModel):
    def __init__(self, dimpsi):
        super(RankingStructModel, self).__init__(dimpsi)

    def predict(self, X):
        """X : liste de vecteurs,
        ranking : trier la liste par ordre decroissant de <w, phi(x)>"""
        it = list(enumerate(map(lambda x: self.w @ x, X)))
        sorted_it = sorted(it, key=lambda x : x[1], reverse=True)
        positions, value = zip(*sorted_it)
        it = sorted(list(enumerate(positions)), key=lambda x : x[1], reverse=True)
        ranks, indexes = zip(*it)
        return list(ranks)
        # return RankingOutput(list(ranks), labelsGT)

    # def predict(self, X, labelsGT):
    #     return loss_augmented_inference(X, self.w, labelsGT)
        
    def lai(self, xi, yi):
        """lai = loss_augmented_inference"""
        return loss_augmented_inference(xi, self.w, yi)

    def loss(self, xi, yi):
        ranking_output = self.lai(xi, yi)
        return 1 - average_precision(*recall_precision(ranking_output))

    # def instantiation(self, classe=RankingInstantiation, kwargs={}):
    #     self.mc = classes(**kwargs)

    def save(self, fname="../res/model.bin"):
        with open(fname, "wb") as f:
            pickle.dump(self.w, f)

    def load(self, fname="../res/model.bin"):
        with open(fname, "rb") as f:
            self.w = pickle.load(f)
        

class GenericTrainingAlgorithm(object):
    def __init__(self, dimpsi, classe=LinearStructModel, struct_classe=MultiClass, kwargs={}):
        self.dimpsi = dimpsi
        self.model = classe(dimpsi)
        self.model.instantiation(struct_classe, kwargs)
        
    
    def fit(self, dataset, alpha=0.1, nb_it=10, lr=0.01, nb_samples=None, register=False):
        """ feature_map : fonction qui associe à chaque couple (x,y) un vecteur de dimension d,
            alpha : coefficient de régularisation
            nb_samples : le nombre de point tiré à chaque échantillon"""
            
        #Dataset = namedtuple("Dataset", ["x_train", "x_test", "y_train", "y_test"])

        x_train, x_test, y_train, y_test = dataset
        batch_size, nb_features = x_train.shape
        
        L = []
        if nb_samples is None:
            nb_samples = batch_size

        for i in tqdm(range(nb_it)):
            indexes = np.random.randint(0, batch_size-1, nb_samples)
            for ind in indexes:
                xi, yi = x_train[ind], y_train[ind]
                #losses = ((y, loss(y, yi) + feature_map(xi, y) @ model.w) for y in y_train)
                #yhat = max(losses, key=lambda x : x[1])[0]
                yhat = self.model.lai(xi, yi)
                grad = self.model.mc.psi(xi, yhat) - self.model.mc.psi(xi, yi)
                # print("grad shape : ", grad.shape)
                # print("self.w shape : ", self.model.w.shape)
                # input("wait")
                self.model.w = self.model.w - lr * (alpha * self.model.w + grad)
            if register:
                L.append((accuracy(self, dataset, train=True), accuracy(self, dataset, train=False)))
            
                #print(yhat)
        return L

    def fit_ranking(self, dataset, alpha=0.1, nb_it=10, lr=0.01):
        res = sorted(enumerate(dataset.y_train), key=lambda x : x[1], reverse=True)
        indexes, _ = zip(*res)
        rank, _ = zip(*sorted(enumerate(indexes), key=lambda x : x[1]))
        y = RankingOutput(list(rank), dataset.y_train)
        for i in tqdm(range(nb_it)):
            xi, yi = dataset.x_train, dataset.y_train
            yhat = self.model.lai(xi, yi)
            # print("loss : ", self.model.mc.delta(yhat.rank, yi))
            grad = self.model.mc.psi(xi, yhat) - self.model.mc.psi(xi, y)
            self.model.w = self.model.w - lr * (alpha * self.model.w + grad)


    def predict(self, x):
        return self.model.predict(x)
 
    def save(self, fname="../res/model.bin"):
        self.model.save(fname)

    def load(self, fname="../res/model.bin"):
        self.model.load(fname)
