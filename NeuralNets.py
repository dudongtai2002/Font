# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 15:39:49 2016

@author: shengx
"""

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample



class BinaryLogisticRegression(object): 
    """Multi-class Logistic Regression Class 
    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting 
    data points onto a set of hyperplanes, the distance to which is used 
    to determine a class membership probability. 
    """ 

    def __init__(self, rng, input, n_in, n_out): 
        """ Initialize the parameters of the logistic regression 

        :type n_outs: list of int 
        :param n_outs: number of output units in each group 

        """ 
        self.n_groups = n_out 
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out) 
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.h = T.dot(input, self.W) + self.b 
        self.p_y_given_x = T.nnet.sigmoid(self.h) 
        self.y_pred = (self.p_y_given_x > 0.5)
        
        # parameters of the model 
        self.params = [self.W, self.b] 

    def negative_log_likelihood(self, y): 
        """Return the mean of the negative log-likelihood of the 
        prediction of this model under a given target distribution. 

        .. math:: 
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) = 
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\ 
                \ell (\theta=\{W,b\}, \mathcal{D}) 

        :type y: theano.tensor.TensorType 
        :param y: corresponds to a vector that gives for each example 
                the correct label 

        Note: we use the mean instead of the sum so that 
              the learning rate is less dependent on the batch size 
        """ 
        # y.shape[0] is (symbolically) the number of rows in y, i.e., 
        # number of examples (call it n) in the minibatch 
        # T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1] 
        # T.log(self.p_y_given_x) is a matrix of Log-Probabilities 
        # (call it LP) with one row per example and one column per class 
        # LP[T.arange(y.shape[0]),y] is a vector v containing 
        # [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]] 
        # and T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across 
        # inibatch examples) of the elements in v, 
        # i.e., the mean log-likelihood across the minibatch. 
        cost = -T.mean(T.log(self.p_y_given_x) * y 
            + T.log(1-self.p_y_given_x) * (1 - y))
        #cost = -T.mean(
        #                T.log(self.p_y_given_x)[T.arange(y.shape[0]), y]
        #              + T.log(1 - self.p_y_given_x)[T.arange(y.shape[0]), 1 - y]
        #              )
        return cost 

    def errors(self, ys): 
        errs = [] 
        for idx in xrange(self.n_groups): 
            if ys[:,idx].ndim != self.y_pred[idx].ndim: 
                raise TypeError('y should have the same shape as self.y_pred', 
                    ('y', ys[:,idx].type, 'y_pred', self.y_pred[idx].type)) 
            # check if y is of the correct datatype 
            if ys[:,idx].dtype.startswith('int'): 
                # the T.neq operator returns a vector of 0s and 1s, where 1 
                # represents a mistake in prediction 
                errs.append( T.mean(T.neq(self.y_pred[idx], ys[:,idx]))) 
            else: 
                raise NotImplementedError() 
        return errs


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is sigmoid

        Hidden unit activation is given by: sigmoid(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
        
        
class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) //
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.nnet.sigmoid(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input




