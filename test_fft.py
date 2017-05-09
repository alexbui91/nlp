import numpy as np
import theano
import theano.tensor as T
from theano.tensor import fft



def test_output_theano(x,y):
    data_x = [1.0,2.0,3.0,4.0,5.0]
    f = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    return f
    # return T.mean(T.concatenate([x, y]))


def test_update_theano(a):
    for x in a:
        x += 1

def shared_dataset(data_xy, borrow=True):
    
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)

    return shared_x, T.cast(shared_y, 'int32')


def test_theano():
    x = T.matrix('x')
    y = T.matrix('y')
    index = T.lscalar()
    a = [1.0,2.0,3.0,4.0,5.0]
    data_x = [1.0,2.0,3.0,4.0,5.0]
    f = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
    y_out = theano.shared(np.asarray(a, dtype=theano.config.floatX), borrow=True)
    return f + y_out
    # x_out = theano.shared(np.asarray(a, dtype=theano.config.floatX), borrow=True)
    # y_out = y_out.flatten()
    # y_out = T.cast(y_out, 'int32')
    # train_model = theano.function([index], test_output_theano(x, y),
    #                                 givens={
    #                                     x: f[0:3],
    #                                     y: y_out[1:4]
    #                                 },allow_input_downcast = True)


def initFourier():
    x = T.matrix('x', dtype='float64')
    rfft = fft.rfft(x, norm='ortho')
    f_rfft = theano.function([x], rfft)


def test():
    N = 1024
    box = np.zeros((1, N), dtype='float64')
    box[:, N//2-10: N//2+10] = 1
    out = f_rfft(box)
    c_out = np.asarray(out[0, :, 0] + 1j*out[0, :, 1])
    abs_out = abs(c_out)