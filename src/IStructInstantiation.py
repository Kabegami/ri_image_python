# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 23:33:21 2019

@author: dabi-
"""
import numpy as np
from tools import one_hot
        
class MultiClass:
    
    def __init__(self):
        pass

    def psi(self, x, y):
        ar = np.zeros(len(self.enumerateY()) * 250)
        ar[250*y:(250*y)+250]=x;        
        return ar;


        # xi = x.flatten()
        # v = np.zeros(xi.shape)
        # # print("v : ", v.shape)
        # v[y] = 1
        # return xi * 
    
    
    
    def delta(self, y1, y2):
        if y1==y2:
            return 0
        else:
            return 1
    
    def enumerateY(self):
        return range(9)
    

class MultiClassFier:
    def __init__(self, mat):
        self.mat = mat

    def psi(self, x, y):
        ar = np.zeros(len(self.enumerateY()) * 250)
        ar[250*y:(250*y)+250]=x;
        #print(ar)
        return ar;

    def enumerateY(self):
        return range(9)

    def delta(self, y1, y2):
        return self.mat[y1][y2]

    
