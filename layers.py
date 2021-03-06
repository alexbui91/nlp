import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
import numpy as np

import utils


#class layer
class NetworkLayer(object):

    def initHyperParamsFromValue(self, W, b, name="values"):
        W_param = theano.shared(value=W, borrow=True, name=("W_" + name))
        b_param = theano.shared(value=b, borrow=True, name=("b_" + name))
        self.W = W_param
        self.b = b_param
        self.params = [self.W, self.b]
    
# convolution layer
class ConvolutionLayer(NetworkLayer):
    def __init__(self, rng=None, filter_shape=None, input_shape=None, poolsize=(2, 2), non_linear="tanh"):
        assert input_shape[1] == filter_shape[1]
        self.input = input
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.non_linear = non_linear
        self.poolsize = poolsize
        self.rng = rng

    def initHyperParams(self):
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(self.filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (self.filter_shape[0] * np.prod(self.filter_shape[2:]) / np.prod(self.poolsize))
        # initialize weights with random weights
        if self.non_linear == "none" or self.non_linear == "relu":
            self.W = theano.shared(np.asarray(rng.uniform(low=-0.01, high=0.01, size=self.filter_shape), dtype=theano.config.floatX), borrow=True, name="W_conv")
        else:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(self.rng.uniform(low=-W_bound, high=W_bound, size=self.filter_shape), dtype=theano.config.floatX), borrow=True, name="W_conv")
        b_values = np.zeros((self.filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        self.params = [self.W, self.b]

    def predict(self, new_data):
        conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=self.input_shape)
        if self.non_linear == "tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = pool.pool_2d(input=conv_out_tanh, ws=self.poolsize, ignore_border=True)
        if self.non_linear == "relu":
            conv_out_tanh = utils.ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = pool.pool_2d(input=conv_out_tanh, ws=self.poolsize, ignore_border=True)
        else:
            pooled_out = pool.pool_2d(input=conv_out, ws=self.poolsize, ignore_border=True)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output


# perform hidden layer before full connect layer
class HiddenLayer(NetworkLayer):
    # n_in is the dimension of incomming vector, n_out is the dimension of out_comming vector
    # activation is the performing function (tanh, softmax, anw)
    # rng: random probability to to perform dropout later
    def __init__(self, rng=None, activation="ReLU", n_in=None, n_out=None, W=None, b=None, input_vectors=None):

        self.rng = rng
        self.activation = activation
        self.input = input_vectors
        self.n_in = n_in
        self.n_out = n_out
        self.W = W
        self.b = b
        if self.n_in is not None and self.n_out is not None:
            self.initHyperParams()

    def setInput(self, input_vectors):
        self.input = input_vectors

    def initHyperParams(self):
        if self.W is None:
            if self.activation.func_name == "ReLU":
                W_values = np.asarray(0.01 * self.rng.standard_normal(size=(self.n_in, self.n_out)), dtype=theano.config.floatX)
            else:
                W_values = np.asarray(self.rng.uniform(low=-np.sqrt(6. / (self.n_in + self.n_out)), high=np.sqrt(6. / (self.n_in + self.n_out)), size=(self.n_in, self.n_out)), dtype=theano.config.floatX)
            self.W = theano.shared(value=W_values, borrow=True, name='W')
        if self.b is None:
            b_values = np.zeros((self.n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True, name='b')
        self.params = [self.W, self.b]

    def setDropOutRate(self, drop_out_rate):
        self.drop_out_rate = drop_out_rate

    def setDropOut(self, isDropOut):
        self.isDropOut = isDropOut

    def predict(self):
        lin_output = T.dot(self.input, self.W) + self.b
        self.output = (lin_output if self.activation is None else self.activation(lin_output))


class HiddenLayerDropout(HiddenLayer):
    
    def __init__(self, rng, activation="ReLU", dropout_rate=0.5, n_in=None, n_out=None, W=None, b=None, input_vectors=None):
        super(self.__class__, self).__init__(rng, activation, n_in, n_out, W, b, input_vectors)
        self.dropout_rate = dropout_rate
    
    def dropout(self):
        self.input_vectors = utils.dropout_from_layer(self.rng, self.input, self.dropout_rate)    


# full connect here is final layer, logistic regression => prob to calculate cost function y^ = softmax (W^T * input + b)
class FullConnectLayer(NetworkLayer):

    def __init__(self, rng=None, layers_size=None):
        self.rng = rng
        self.layers_size = layers_size

    def initHyperParams(self):
        W_bound = np.sqrt(6. / (self.layers_size[0] + self.layers_size[1]))
        w_size = (self.layers_size[0], self.layers_size[1])
        # random weight, bias with initial values
        # self.W = theano.shared(np.asarray(self.rng.uniform(low=-W_bound, high=W_bound, size=w_size), dtype=theano.config.floatX), borrow=True, name="W_full_connect")
        # b_values = np.zeros((self.layers_size[1],), dtype=theano.config.floatX)
        # self.b = theano.shared(value=b_values, name='b')
        # random weight, bias with zeros values
        self.W = theano.shared(
                    value=np.zeros(w_size, dtype=theano.config.floatX),
                    name='W')
        self.b = theano.shared(
                    value=np.zeros((self.layers_size[1],), dtype=theano.config.floatX),
                    name='b')
        self.params = [self.W, self.b]
    
    def setInput(self, inp):
        self.input_vector = inp

    def predict(self):
        y_p = self.predict_p()
        self.y_pred = T.argmax(y_p, axis=1)

    def predict_p(self):
        self.y_prob = T.nnet.softmax(T.dot(self.input_vector, self.W) + self.b)
        return self.y_prob

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.y_prob)[T.arange(y.shape[0]), y])

    def soft_negative_log_likelihood(self, y):
        return -T.mean(T.sum(T.log(self.y_pred) * y, axis=1))

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
