import numpy as np
import theano
import theano.tensor as T
import pickle
import os.path as path
from data import Data


def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)


def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)


def Tanh(x):
    y = T.tanh(x)
    return(y)


def Iden(x):
    y = x
    return(y)


def dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
"""
    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, theano.config.floatX)
    return output


def find_largest_number(num1, num2, num3):
    largest = num1
    if (num1 >= num2) and (num1 >= num3):
       largest = num1
    elif (num2 >= num1) and (num2 >= num3):
       largest = num2
    else:
       largest = num3
    return largest


def save_file(name, obj):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_file(pathfile):
    if path.exists(pathfile):
        with open(pathfile, 'rb') as f:
            data = pickle.load(f)
        return data 


def make_sentence_idx(vocabs, sent, max_sent_length):
    sent_v = list()
    sent_length = len(sent)
    for i in xrange(max_sent_length):
        if i < sent_length:
            if sent[i] in vocabs:
                sent_v.append(vocabs[sent[i]])
            else: 
                sent_v.append(0)
        else:
            sent_v.append(0)
    return np.asarray(sent_v, dtype='int32')


def loadWordVectors(file, data_path):
    d = Data(file)
    d.loadWordVectorsFromText(data_path)
    return d.vectors, d.vocabs

def float_x(arr):
        return np.asarray(arr, dtype=theano.config.floatX)