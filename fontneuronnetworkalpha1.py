# Neural network for font gen,
# potential of using Class

import json
import numpy as n
import random as r
import math as ma
from generate import *
# 1 bias term is add into all input
imagesize=50
trainsame=10000
traindifferent=10000
testnumber=2000

generatedata(imagesize,trainsame,traindifferent,testnumber)

inputsize = 49*49
lay1size = 100
lay2size = 100
inputsize += 1
w1 = n.matrix([[r.uniform(0,0.01) for i in range(inputsize)] for x in range(lay1size)])
b1 = n.matrix([[r.uniform(0,0.01)] for i in range(inputsize)])


def xtoy(input, wi, bi):

    return wi * input + bi


def propagate(input, w, b):
    if len(w) != len(b):
        print("Size of w does not equal to size len of b")
        return
    for i in range(len(w)):
        input = xtoy(input, w[i], b[i])
        input = activation(input)
    return input


def activation(x):
    return 1/(1+ma.exp(x))
activation = n.vectorize(activation)


