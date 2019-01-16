import numpy as np

class GenericTrainingAlgorithm:
    def fit(self, x_train, y_train, feature_map, loss, alpha, nb_it, lr, d, nb_samples=None):
        """ feature_map : fonction qui associe à chaque couple (x,y) un vecteur de dimension d,
            alpha : coefficient de régularisation
            d : dimension  de sortie de la feature_map
            nb_samples : le nombre de point tiré à chaque échantillon"""
        batch_size, nb_features = x_train.shape
        self.w = np.random.randn(nb_features, d)
        if nb_samples is None:
            nb_samples = batch_size

        for i in range(nb_it):
            indexes = np.random.randint(0, batch_size-1, nb_samples)
            for ind in indexes:
                x, y = x_train[ind], y_train[ind]
                
                
        
    
