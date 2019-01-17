#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 12:03:41 2019

@author: 3603138
"""

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