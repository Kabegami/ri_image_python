from imageNetParser import get_dico_size
import numpy as np

def compute_bow(feature_list):
    v = np.zeros(get_dico_size())
    for w in feature_list.get_words():
        v[w] += 1
    norm2 = np.linalg.norm(v, ord=2)
    v /= norm2
    print(np.linalg.norm(v, ord=2))
    return v


