import numpy as np
import random

def accuracy_(model, x, y):
    s = 0
    for xi,yi in zip(x,y):
        pred = model.predict(xi)
        if pred == yi:
            s += 1
    N = len(x)
    return s / N

def accuracy(model, dataset, train=True):
    if train:
        return accuracy_(model, dataset.x_train, dataset.y_train)
    else:
        return accuracy_(model, dataset.x_test, dataset.y_test)

def one_hot(target, nb_target):
    v = np.zeros(nb_target)
    v[target] = 1
    return v
    
def foreach(f, it):
    for e in it:
        f(e)

def swap(t):
    "(x1, x2) -> (x2, x1)"
    return tuple(reversed(t))

def map2(f, it):
    return list(map(f, it))

def split(matrix, axis=0):
    """matrix -> list of vectors"""
    a,b = matrix.shape
    res = []
    if axis == 0:
        for i in range(a):
            res.append(matrix[i, :])
        return res

    if axis == 1:
        for j in range(b):
            res.append(matrix[:, j])
        return res

def random_rank(size):
    L = list(range(size))
    random.shuffle(L)
    return L

def random_labels(size):
    population = [-1, 1]
    return random.choices(population, k=size)
