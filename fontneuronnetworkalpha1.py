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

traininput1,traininput2,testinput1,testinput2,y_train,y_test=generatedata(imagesize,trainsame,traindifferent,testnumber)

# 1 bias term is added to all inputs
# more layers could be added and their sizes can be changed

inputsize = 49*49
lay1size = 100
lay2size = 100
lay3size = 100
lay4size = 100

size = [inputsize, lay1size, lay2size, lay3size, lay4size, inputsize]
lsize = len(size)

w = [0] * lsize
b = [0] * lsize

for j in range(lsize - 1):
    w[j] = n.matrix([[r.uniform(0,0.01) for i in range(size[j] + 1)] for x in range(size[j+1])])
    b[j] = n.matrix([[r.uniform(0,0.01)] for i in range(size[j] + 1)])

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
