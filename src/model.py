import numpy as np
from IStructInstantiation import *
from tqdm import tqdm
from tools import accuracy, accuracy_

class LinearStructModel:
    
    def __init__(self, dimpsi):
        #self.w = np.random.randn(dimpsi)
        self.w = np.zeros(dimpsi)
        
    def predict(self, x):
        return np.argmax([self.mc.psi(x,y).dot(self.w) for y in range(9)])
    
    def lai(self, xi, yi):
        return np.argmax([self.mc.delta(yi, y)+self.mc.psi(xi, y).dot(self.w) for y in range(9)])
    
    def loss(self, xi, yi):
        return np.max([self.mc.delta(yi, y)+self.mc.psi(xi, y).dot(self.w) for y in range(9)])

        #, np.argmax([self.mc.psi(xi, y).dot(self.w) for y in range(9)])
    
    def instantiation(self, classe=MultiClass, kwargs={}):
        self.mc = classe(**kwargs)
    
    def getParameters():
        self.w

        

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
                self.model.w = self.model.w - lr * (alpha * self.model.w + grad)
            if register:
                L.append((accuracy(self, dataset, train=True), accuracy(self, dataset, train=False)))
            
                #print(yhat)
        return L
                
    def predict(self, x):
        return self.model.predict(x)
 

