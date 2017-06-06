import time

import numpy as np
import theano
import theano.tensor as T
import theano.printing as printing

from layers import ConvolutionLayer, HiddenLayer, HiddenLayerDropout, FullConnectLayer
import utils


class Model:

    def __init__(self, word_vectors=None, training_data=None, dev_data=None, test_data=None,
                 img_width=300, img_height=53, hidden_units=[100, 2],
                 dropout_rate=0.5, filter_size=[3, 4, 5], batch_size=50,
                 epochs=20, patience=20, learning_rate=0.13, conv_non_linear="tanh"):
        
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
        layer0_input = T.cast(Words[T.cast(x.flatten(), dtype="int32")], dtype=theano.config.floatX).reshape((self.batch_size, 1, self.img_height, self.img_width))
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
        grads = T.grad(cost, params)
        # dropout to evaluate overfitting
        hidden_layer_dropout.dropout()
        hidden_layer_dropout.predict()
        full_connect.setInput(hidden_layer_dropout.output)
        full_connect.predict()
        cost_d = full_connect.negative_log_likelihood(y)
        grads_dropout = T.grad(cost_d, params)
        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - self.learning_rate * (grad_i + grad_drop_i))
            for param_i, grad_i, grad_drop_i in zip(params, grads, grads_dropout)
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
        val_batch_lost = 1.
        best_batch_lost = 1.
        stop_count = 0
        epoch = 0
        while(epoch < self.epochs):
            epoch_cost_train = 0.
            average_test_epoch_score = 0.
            test_epoch_score = 0.
            total_test_time = 0
            epoch += 1
            print("Start epoch: %i" % epoch)
            start = time.time()
            for mini_batch in xrange(n_train_batches):
                epoch_cost_train += train_model(mini_batch)
                # perform early stopping to avoid overfitting (check with frequency or check every iteration)
                # iter = (epoch - 1) * n_train_batches + minibatch_index
                # if (iter + 1) % validation_frequency == 0
                # eval
                val_losses = [val_model(i) for i in xrange(n_val_batches)]
                val_losses = np.array(val_losses)
                # in valuation phase (dev phase, error need to be reduce gradually and not upturn)
                # if val_gain > best_gain => re assign and stop_count = 0 else
                # stop_count ++.
                # average of losses during evaluate => this number may be larger than 1
                val_batch_lost = np.mean(val_losses)
                if val_batch_lost < best_batch_lost:
                    best_batch_lost = val_batch_lost
                    stop_count = 0
                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    avg_test_lost = np.mean(test_losses)
                    print("test lost: %f" % avg_test_lost)
                    test_epoch_score += avg_test_lost
                    total_test_time += 1
                else:
                    stop_count += 1
                if stop_count == self.patience:
                    break
            average_test_epoch_score = test_epoch_score / total_test_time
            print(('epoch %i, test error of %i example is: %.5f') %
                  (epoch, test_len, average_test_epoch_score * 100.))
            print('epoch: %i, training time: %.2f secs; with cost: %.2f' %
                  (epoch, time.time() - start, epoch_cost_train))
        self.save_trained_params(cnet, hidden_layer, full_connect)

    def shared_dataset(self, data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(
            data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(
            data_y, dtype=theano.config.floatX), borrow=borrow)
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
        layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((self.batch_size, 1, input_height, self.img_width))
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
        test_data_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=True)
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