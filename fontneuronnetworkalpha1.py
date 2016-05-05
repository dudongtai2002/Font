# Neural network for font gen,
# potential of using Class

import json
import numpy as n
import random as r
import math as ma
import generate
from generate import *
from NeuralNets import *
# 1 bias term is add into all input
imagesize=36
trainsame=1000
traindifferent=1000
testnumber=2000
batch_size=50
n_epochs = 1
n_train_batches = 40

traininput1,traininput2,testinput1,testinput2,y_train,y_test=generatedata(imagesize,trainsame,traindifferent,testnumber)

traininput1=traininput1.transpose()
traininput2=traininput2.transpose()
"""
traininput1=traininput1.reshape((trainsame+traindifferent,imagesize))
traininput2=traininput2.reshape((traindifferent+traindifferent,imagesize))
"""

testinput1=testinput1.transpose()  #2000*1296(36^2)
testinput2=testinput2.transpose()  #2000*1296(36^2)
y_train=y_train.reshape(trainsame+trainsame,1)

trainInput1,trainInput2,y_Train  = shared_dataset(traininput1, traininput2, y_train)
#testInput1,testInput2,y_Test  = shared_dataset(testinput1, testinput2, y_test)

#y_train(y_train=np.zeros(trainsame+traindifferent))
#traininput1=np.zeros(transpose(imagesize*imagesize,trainsame+traindifferent))
learning_rate=1
#trainInput1/2,testinput1/2:2000*1296
#y_train,y_test:2000


index = T.lscalar()  # index to a [mini]batch
x1 = T.matrix('x1')
x2 = T.matrix('x2')
y = T.imatrix('y')

# 1 bias term is added to all inputs
# more layers could be added and their sizes can be changed
print("building the model!")
"""
x1->layer000->layer100(3*3->3*3)
  ->layer001->layer101(4*4->3*3)   ->layer2->layer3->layer4(output)
x2->layer010->layer110(3*3->3*3)
  ->layer011->layer111(4*4->3*3)


"""
layer00_input=x1.reshape((batch_size,1,imagesize,imagesize))  #50,1,36,36
layer01_input=x2.reshape((batch_size,1,imagesize,imagesize))

# first layer, 36-3+1=34, 34/2=17;
#36-5+1=32,32/2=16

layer000 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer00_input,
        image_shape=(batch_size, 1, imagesize, imagesize),   # input image shape
        filter_shape=(1, 1, 3, 3),
        poolsize=(2, 2)
    )

layer001 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer00_input,
        image_shape=(batch_size, 1, imagesize, imagesize),   # input image shape
        filter_shape=(1, 1, 5, 5),
        poolsize=(2, 2)
    )

layer010 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer01_input,
        image_shape=(batch_size, 1, imagesize, imagesize),   # input image shape
        filter_shape=(1, 1, 3, 3),
        poolsize=(2, 2)
    )

layer011 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer01_input,
        image_shape=(batch_size, 1, imagesize, imagesize),   # input image shape
        filter_shape=(1, 1, 5, 5),
        poolsize=(2, 2)
    )
#second layer, 17-3+1=15 ,15/2=?, 16-3+1=14,14/2=7
layer100 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer000.output,
        image_shape=(batch_size, 1, 17, 17),
        filter_shape=(1, 1, 3, 3),
        poolsize=(2, 2)
    )
layer101 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer001.output,
        image_shape=(batch_size, 1, 16, 16),
        filter_shape=(1, 1, 3, 3),
        poolsize=(2, 2)
    )
layer110 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer010.output,
        image_shape=(batch_size, 1, 17, 17),
        filter_shape=(1, 1, 3, 3),
        poolsize=(2, 2)
    )
layer111 = LeNetConvPoolLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer011.output,
        image_shape=(batch_size, 1, 16, 16),
        filter_shape=(1, 1, 3, 3),
        poolsize=(2, 2)
    )

#third layer
layer2_input = T.concatenate([layer100.output.flatten(2), layer101.output.flatten(2), layer110.output.flatten(2), layer111.output.flatten(2)],
                              axis = 1)

layer2 = HiddenLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer2_input,
        n_in=4*7*7,
        n_out=50,
        activation=T.nnet.sigmoid
    )
layer3 = HiddenLayer(
        np.random.RandomState(np.random.randint(10000)),
        input=layer2.output,
        n_in=50,
        n_out=50,
        activation=T.nnet.sigmoid
    )

layer4 = BinaryLogisticRegression(
        np.random.RandomState(np.random.randint(10000)),
        input=layer3.output,
        n_in=50,
        n_out=1
    )

cost = layer4.negative_log_likelihood(y)
error = ((y - layer4.y_pred)**2).sum()
params = (layer4.params
        + layer3.params
        + layer2.params
        + layer100.params + layer101.params + layer110.params + layer111.params
        + layer000.params + layer001.params + layer010.params + layer011.params)
grads = T.grad(cost, params)

updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]



train_model = theano.function(
        inputs = [index],
        outputs = cost,
        updates=updates,
        givens={
            x1: trainInput1[index * batch_size: (index + 1) * batch_size],    #50*1296
            x2: trainInput2[index * batch_size: (index + 1) * batch_size],    #50*1296
            y: y_Train[index * batch_size: (index + 1) * batch_size]
        }
    )
epoch=0

while (epoch < n_epochs):
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):
        minibatch_avg_cost = train_model(minibatch_index)
        iter = (epoch - 1) * n_train_batches + minibatch_index
        print(('   epoch %i, minibatch %i/%i.') % (epoch, minibatch_index +1, n_train_batches))




predict_model = theano.function(
        inputs = [x1,x2],
        outputs = layer4.p_y_given_x,
        on_unused_input='ignore'
    )

predicted_values = predict_model(testinput1[0:50],testinput2[0:50])

f=open('output.txt','w+')
f.write(np.array_str(predicted_values))
f.close()

f1=open('real.txt','w+')
f1.write(np.array_str(y_test[0:50]))
f1.close()
"""

data = {'generate':predicted_values, 'testing':y_test[0:50]}
with open('output.json', 'w') as outfile:
        json.dump(data, outfile)

"""
"""
inputsize = 49*49
lay1size = 100
lay2size = 100
lay3size = 100
lay4size = 100

size = [inputsize, lay1size, lay2size, lay3size, lay4size, inputsize]
lsize = len(size)
-
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
"""