# Neural network for font gen,
# potential of using Class


import numpy as n
import random as r
import math as ma

# 1 bias term is add into all input

inputsize = 49*49
lay1size = 100
lay2size = 100
inputsize += 1
w1 = n.matrix[[r.uniform(0,0.01) for i in range(trainingsize)] for x in range(lay1size)]
b1 = n.matrix[[r.uniform(0,0.01)] for i in range(trainingsize)]


def xtoy(input, wi, bi):

    return wi * input + bi


def propagate(input, w, b)
    if len(w) != len(b):
        print("some len of w does not equal to some len of b")
        return
    for i in len(range(w)):
        input = xtoy(input, wi[i], bi[i])
    return input


def activation(x):
    return 1/(1+ma.exp(x))



