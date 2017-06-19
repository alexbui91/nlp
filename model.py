import time

import numpy as np
import theano
import theano.tensor as T
import theano.printing as printing

from layers import ConvolutionLayer, HiddenLayer, HiddenLayerDropout, FullConnectLayer
import utils

floatX = theano.config.floatX

class Model:

    def __init__(self, word_vectors=None, training_data=None, dev_data=None, test_data=None,
                 img_width=300, img_height=53, hidden_units=[100, 2],
                 dropout_rate=0.5, filter_size=[3, 4, 5], batch_size=50,
                 epochs=20, patience=10000, patience_frq=2, learning_rate=0.13, conv_non_linear="tanh", 
                 gradient=None, gradient_d=None):
        
        self.word_vectors = word_vectors
        self.img_width = img_width
        self.img_height = img_height
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.filter_size = filter_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.conv_non_linear = conv_non_linear
        self.patience = patience
        self.learning_rate = learning_rate
        self.training_data = training_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.gradient = gradient
        self.gradient_d = gradient_d
        
    # def initFastFourier(self):
    #     x = T.matrix('x', dtype='float64')
    #     rfft = fft.rfft(x, norm='ortho')
    #     self.fft = theano.function([x], rfft)

    def trainNet(self):
        # init data for model
        test_len = len(self.test_data[0])
        n_train_batches = len(self.training_data[0]) // self.batch_size
        n_val_batches = len(self.dev_data[0]) // self.batch_size
        n_test_batches = test_len // self.batch_size
        test_set_x, test_set_y = self.shared_dataset(self.test_data)
        val_set_x, val_set_y = self.shared_dataset(self.dev_data)
        train_set_x, train_set_y = self.shared_dataset(self.training_data)

        index = T.lscalar()
        # init model with 1 convolution layer, 1 hidden layer, 1 full connect layer
        # given index => x = batch_size * index => batch_size * (index + 1)
        x = T.matrix('x')
        y = T.ivector('y')
        Words = theano.shared(value=self.word_vectors, name="Words", borrow=True)
        # resign from batch_size * 300 * height => batch_size * 1 * height * width
        # cast word index to vector in dictionary
        layer0_input = T.cast(Words[T.cast(x.flatten(), dtype="int32")], dtype=floatX).reshape((self.batch_size, 1, self.img_height, self.img_width))
        layer1_inputs = list()
        # init networks
        rng = np.random.RandomState(3435)
        # create convolution network for each window size 3, 4, 5
        cnet = list()
        for window_size in self.filter_size:
            pool_height = self.img_height - window_size + 1
            conv_layer = ConvolutionLayer(rng, (self.hidden_units[0], 1, window_size, self.img_width),
                                          (self.batch_size, 1, self.img_height, self.img_width),
                                          [pool_height, 1])
            conv_layer.initHyperParams()
            conv_output = conv_layer.predict(layer0_input)
            # size of layer0_input: B x 1 x 100 x 1
            layer1_input = conv_output.flatten(2)
            # size of layer1_input: B x 1 x 1 x100
            cnet.append(conv_layer)
            layer1_inputs.append(layer1_input)
        # final vector z = concatenate of all max features B x 1 x 1 x 300
        output_convs = T.concatenate(layer1_inputs, 1)
        final_vector_dim = self.hidden_units[0] * 3
        hidden_layer = HiddenLayer(rng, utils.Tanh, final_vector_dim, self.hidden_units[0])
        # hidden layer dropout still use weight and bias of hidden layer. It just
        # cuts off some neuron randomly with drop_out_rate
        hidden_layer_dropout = HiddenLayerDropout(rng, utils.Tanh, self.dropout_rate, final_vector_dim, self.hidden_units[0], hidden_layer.W, hidden_layer.b)
        # apply full connect layer to final vector
        full_connect = FullConnectLayer(rng, (self.hidden_units[0], 2))
        full_connect.initHyperParams()
        #perform filter
        hidden_layer.setInput(output_convs)
        hidden_layer_dropout.setInput(output_convs)
        hidden_layer.predict()
        full_connect.setInput(hidden_layer.output)
        full_connect.predict()
        # create a list of all model parameters to be fit by gradient descent
        # params = hidden_layer.params + full_connect.params + conv_layer.params
        params = hidden_layer.params + full_connect.params
        for conv in cnet:
            params += conv.params
        # calculate cost for normal model
        cost = full_connect.negative_log_likelihood(y)
        # create a list of gradients for all model parameters
        delta = self.perform_gradient(self.gradient, cost, params, 'N')
        # dropout to evaluate overfitting
        hidden_layer_dropout.dropout()
        hidden_layer_dropout.predict()
        full_connect.setInput(hidden_layer_dropout.output)
        full_connect.predict()

        cost_d = full_connect.negative_log_likelihood(y)
        delta_d = self.perform_gradient(self.gradient_d, cost_d, params, 'D')
        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        
        updates = [
            (param_i, param_i - (d_i + dd_i))
            for param_i, d_i, dd_i in zip(params, delta, delta_d)
        ]

        train_model = theano.function([index], cost, updates=updates, givens={
            x: train_set_x[(index * self.batch_size):((index + 1) * self.batch_size)],
            y: train_set_y[(index * self.batch_size):((index + 1) * self.batch_size)]
        })
        val_model = theano.function([index], full_connect.errors(y),
                                    givens={
                                        x: val_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                                        y: val_set_y[index * self.batch_size: (index + 1) * self.batch_size],
        })
        test_model = theano.function(inputs=[index], outputs=full_connect.errors(y), givens={
            x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
            y: test_set_y[index * self.batch_size: (index + 1) * self.batch_size]
        })
        validation_frequency = min(n_train_batches, self.patience // 2)
        val_batch_lost = 1.
        best_batch_lost = 1.
        best_test_lost = 1.
        stop_count = 0
        epoch = 0
        done_loop = False
        current_time_step = 0
        improve_threshold = 0.995
        iter_list = range(n_train_batches)
        while(epoch < self.epochs and done_loop is not True):
            epoch_cost_train = 0.
            epoch += 1
            batch_train = 0
            print("Start epoch: %i" % epoch)
            start = time.time()
            random.shuffle(iter_list)
            for mini_batch, m_b_i in zip(iter_list, xrange(n_train_batches)):
                current_time_step = (epoch - 1) * n_train_batches + m_b_i
                epoch_cost_train += train_model(mini_batch)
                batch_train += 1
                if (current_time_step + 1) % validation_frequency == 0:
                    val_losses = [val_model(i) for i in xrange(n_val_batches)]
                    val_losses = np.array(val_losses)
                    val_batch_lost = np.mean(val_losses)
                    if val_batch_lost < best_batch_lost:
                        if best_batch_lost * improve_threshold > val_batch_lost:
                            self.patience = max(self.patience, current_time_step * self.patience_frq)
                            best_batch_lost = val_batch_lost
                            # test it on the test set
                            test_losses = [
                                test_model(i)
                                for i in range(n_test_batches)
                            ]
                            current_test_lost = np.mean(test_losses)
                            print(('epoch %i minibatch %i test accuracy of %i example is: %.5f') % (epoch, m_b_i, test_len, (1 - current_test_lost) * 100.))
                            if best_test_lost > current_test_lost:
                                best_test_lost = current_test_lost
                if self.patience <= current_time_step:
                    print(self.patience)
                    done_loop = True
                    break
            print('epoch: %i, training time: %.2f secs; with avg cost: %.5f' % (epoch, time.time() - start, epoch_cost_train / batch_train))
        print('Best test accuracy is: %.5f' % (1 - best_test_lost))
        self.save_trained_params(cnet, hidden_layer, full_connect)

    def shared_dataset(self, data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(
            data_x, dtype=floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(
            data_y, dtype=floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')

    def save_trained_params(self, conv_layers, hidden_layer, full_connect):
        data = dict()
        if full_connect:
            data['full_connect_layer'] = [param.get_value() for param in full_connect.params]
        if hidden_layer:
            data['hidden_layer'] = [param.get_value() for param in hidden_layer.params]
        if conv_layers:
            conv_params = dict()
            for conv_layer, window_size in zip(conv_layers, self.filter_size):
                conv_params[window_size] = [param.get_value() for param in conv_layer.params]
            data['conv_layers'] = conv_params
        if data:
            utils.save_file('data/trained_params.txt', data)
        return data
        
    def load_trained_params(self, path='data/trained_params.txt'):
        data = utils.load_file(path)
        full_connect_params = data['full_connect_layer']
        hidden_layer_params = data['hidden_layer']
        conv_layers_params = data['conv_layers']
        return conv_layers_params, hidden_layer_params, full_connect_params
    
    def build_test_model(self, data):
        conv_p, hidden_p, full_p = self.load_trained_params()
        conv_layers, hidden_layer, full_connect = self.init_model_from_params(conv_p, hidden_p, full_p)
        data_x, data_y, input_height = data
        test_len = len(data_x)
        n_test_batches = test_len // self.batch_size
        x = T.matrix('x')
        y = T.ivector('y')
        index = T.lscalar()
        Words = theano.shared(value=self.word_vectors, name="Words", borrow=True)
        layer0_input = T.cast(Words[T.cast(x.flatten(), dtype="int32")], dtype=floatX).reshape((self.batch_size, 1, input_height, self.img_width))
        layer1_inputs = list()
        for conv_layer in conv_layers:
            conv_output = conv_layer.predict(layer0_input)
            # size of layer0_input: B x 1 x 100 x 1
            layer1_input = conv_output.flatten(2)
            # size of layer1_input: B x 1 x 1 x100
            layer1_inputs.append(layer1_input)
        output_conv = T.concatenate(layer1_inputs, 1)
        hidden_layer.setInput(output_conv)
        hidden_layer.predict()
        full_connect.setInput(hidden_layer.output)
        full_connect.predict()
        test_data_x = theano.shared(np.asarray(data_x, dtype=floatX), borrow=True)
        test_data_y = theano.shared(np.asarray(data_y, dtype='int32'), borrow=True)
      
        errors = 0.
        if test_len == 1:
            test_model = theano.function([index],outputs=full_connect.get_predict(), on_unused_input='ignore', givens={
                x: test_data_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: test_data_y[index * self.batch_size: (index + 1) * self.batch_size]
            })
            index = 0
            avg_errors = test_model(index)
        else:
            test_model = theano.function([index], outputs=full_connect.errors(y), givens={
                x: test_data_x[index * self.batch_size: (index + 1) * self.batch_size],
                y: test_data_y[index * self.batch_size: (index + 1) * self.batch_size]
            })
            for i in xrange(n_test_batches):
                errors += test_model(i)
            avg_errors = errors / n_test_batches
        return avg_errors
    
    def init_model_from_params(self, conv_layers_params, hidden_layer_params, full_connect_params):
        rng = np.random.RandomState(3435)
        full_connect_layer = FullConnectLayer(rng)
        full_connect_layer.initHyperParamsFromValue(full_connect_params[0], full_connect_params[-1], 'full_connect')
        hidden_layer = HiddenLayer(rng, utils.Tanh)
        hidden_layer.initHyperParamsFromValue(hidden_layer_params[0], hidden_layer_params[-1], 'hidden_layer')
        conv_layers = list()
        for window_size in self.filter_size:
            if window_size in conv_layers_params:
                conv_param = conv_layers_params[window_size]
                conv_layer = ConvolutionLayer(rng,(self.hidden_units[0], 1, window_size, self.img_width),
                                              (self.batch_size, 1, self.img_height, self.img_width),
                                              [self.img_height - window_size + 1, 1])
                conv_layer.initHyperParamsFromValue(conv_param[0], conv_param[-1], 'conv_layer')
                conv_layers.append(conv_layer)
        return conv_layers, hidden_layer, full_connect_layer

    def perform_gradient(self, grd, cost, params, name):
        params_length = len(params)
        e_grad, e_delta_prev = self.init_hyper_values(params_length, name)
        grads = T.grad(cost, params)
        if grd is "adadelta":
            delta = self.adadelta(grads, e_grad, e_delta_prev)
            grads = delta
        else: 
            grads = self.sgd(grads)
        return grads


    def init_hyper_values(self, length, name="N"):
        e_grad = theano.shared(np.zeros(length, dtype=floatX), name="e_grad" + name)
        e_delta = theano.shared(np.zeros(length, dtype=floatX), name="e_delta" + name)
        # delta = theano.shared(np.zeros(length, dtype=floatX), name="delta" + name)
        # e_grad = np.zeros(length, dtype=floatX)
        # e_delta = np.zeros(length, dtype=floatX)
        # delta = np.zeros(length, dtype=floatX)
        return e_grad, e_delta

    def sgd(self, grads):
        return [utils.float_X(self.learning_rate) * g for g in grads]

    #e_delta_prev is e of delta of two previous step
    def adadelta(self, grads, e_g_prev, e_delta):
        #calculate e value for grad from e g previous and current grad
        e_grad = self.average_value(e_g_prev, grads)
        #calculate rms for grad
        rms_g = self.RMS(e_grad)
        #rms0 = sqrt(epsilon)
        rms_e_del_prev = self.RMS(e_delta)
        #delta of current time
        delta = [rd / rg * g for rd, rg, g in zip(rms_e_del_prev, rms_g, grads)]
        #e value of delta of time t
        e_delta_1 = self.average_value(e_delta, delta)
        
        return e_grad, e_delta_1, delta

    def rms_prop(self, grads, e_g_prev):
        e_grad = self.average_value(e_g_prev, grads)
        rms_g = self.RMS(e_grad)
        delta = delta = self.cal_delta(rms_g, grads)
        return e_grad, delta

    def RMS(self, values):
        return [T.sqrt(e + utils.float_X(properties.epsilon)) for e in  values]

    def average_value(self, E_prev, grads):
        # grads_ = [T.cast(i, floatX) for i in grads]
        # return E_prev * properties.gamma + (1 - properties.gamma) * grads_
        f_gm = utils.float_x(properties.gamma)
        return [e * f_gm + (utils.float_x(1.) - f_gm) * (g**2.) for e, g in  zip(E_prev, grads)]

