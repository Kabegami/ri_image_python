import numpy as np
import random
from collections import defaultdict
import collections
import functools
import pickle
import matplotlib.pyplot as plt
import itertools

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

def group_by(it):
    res = defaultdict(list)
    for i, e in enumerate(it):
        res[e].append(i)
    return res

# class memoized(object):
#    '''Decorator. Caches a function's return value each time it is called.
#    If called later with the same arguments, the cached value is returned
#    (not reevaluated).
#    '''
#    def __init__(self, func):
#       self.func = func
#       self.cache = {}
#    def __call__(self, *args):
#       if not isinstance(args, collections.Hashable):
#          # uncacheable. a list, for instance.
#          # better to not cache than blow up.
#          return self.func(*args)
#       if args in self.cache:
#          return self.cache[args]
#       else:
#          value = self.func(*args)
#          self.cache[args] = value
#          return value
#    def __repr__(self):
#       '''Return the function's docstring.'''
#       return self.func.__doc__
#    def __get__(self, obj, objtype):
#       '''Support instance methods.'''
#       return functools.partial(self.__call__, obj)

def memoized(f, dico={}):
    d = dico
    def helper(this, x, y):
        key = id(x) + id(y)
        if key not in d:
            v = f(this, x, y)
            d[key] = v
        return d[key]
    return helper

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

        
