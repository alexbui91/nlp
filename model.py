import time

import numpy as np
import theano
import theano.tensor as T

from layers import ConvolutionLayer, HiddenLayer, HiddenLayerDropout, FullConnectLayer
import utils


class Model:

    def __init__(self, word_vectors, training_data, dev_data, test_data, img_width=300, img_height=53,
                 hidden_units=[100, 2], dropout_rate=0.5, filter_size=[3, 4, 5],
                 batch_size=50, epochs=11, patience=20, learning_rate=0.13, conv_non_linear="tanh"):
        self.word_vectors = word_vectors
        self.fft = None
        self.W_conv = None
        self.B_conv = None
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
        n_train_batches = len(self.training_data[0]) // self.batch_size
        n_val_batches = len(self.dev_data[0]) // self.batch_size
        n_test_batches = len(self.test_data[0]) // self.batch_size
        print(n_train_batches, n_val_batches, n_test_batches)
        test_set_x, test_set_y = self.shared_dataset(self.test_data)
        val_set_x, val_set_y = self.shared_dataset(self.dev_data)
        train_set_x, train_set_y = self.shared_dataset(self.training_data)

        index = T.lscalar()
        # init model with 1 convolution layer, 1 hidden layer, 1 full connect layer
        # given index => x = batch_size * index => batch_size * (index + 1)
        x = T.matrix('x')
        y = T.ivector('y')
        Words = theano.shared(value=self.word_vectors, name="Words")
        # resign from batch_size * 300 * height => batch_size * 1 * height * width
        # cast word index to vector in dictionary
        layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape(
            (self.batch_size, 1, self.img_height, self.img_width))
        layer1_inputs = list()

        #init networks
        rng = np.random.RandomState(3435)
        # create convolution network for each window size 3, 4, 5
        cnet = list()
        for fh in self.filter_size:
            conv_layer = ConvolutionLayer(rng, (self.hidden_units[0], 1, fh, self.img_width),
                                                    (self.batch_size, 1, self.img_height, self.img_width), 
                                                    [self.img_height - 3 + 1, 1])
            cnet.append(conv_layer)
            conv_output = conv_layer.predict(layer0_input)
            # size of layer0_input: B x 1 x 100 x 1
            layer1_input = conv_output.flatten(2)
            # size of layer1_input: B x 1 x 1 x100
            layer1_inputs.append(layer1_input)
        # final vector z = concatenate of all max features B x 1 x 1 x 300
        layer1_input = T.concatenate(layer1_inputs, 1)
        print(T.shape(layer1_input))
        final_vector_dim = self.hidden_units[0]
        hidden_layer = HiddenLayer(rng, utils.Tanh, final_vector_dim, self.hidden_units[0])
        # hidden layer dropout still use weight and bias of hidden layer. It just
        # cuts off some neuron randomly with drop_out_rate
        hidden_layer_dropout = HiddenLayerDropout(rng, utils.Tanh, self.dropout_rate, final_vector_dim, self.hidden_units[0], hidden_layer.W, hidden_layer.b)
        # apply full connect layer to final vector
        full_connect = FullConnectLayer(rng, (final_vector_dim, 2))

        hidden_layer.setInput(layer1_input)
        hidden_layer_dropout.setInput(layer1_input)
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
        val_batch_gain = 0
        best_batch_gain = 0
        stop_count = 0
        epoch = 0
        total_cost_train = 0.
        average_test_score = 0.
        total_test_score = 0.
        while(epoch < self.epochs):
            total_cost_train = 0.
            average_test_score = 0.
            total_test_score = 0.
            epoch += 1
            start = time.time()
            for mini_batch in xrange(n_train_batches):
                total_cost_train += train_model(mini_batch)
                # perform early stopping to avoid overfitting (check with frequency or check every iteration)
                # iter = (epoch - 1) * n_train_batches + minibatch_index
                # if (iter + 1) % validation_frequency == 0
                # eval
                val_losses = [val_model(i) for i in xrange(n_val_batches)]
                val_losses = np.array(val_losses)
                # in valuation phase (dev phase, error need to be reduce gradually and not upturn)
                # gain = 1 - lost
                # if val_gain > best_gain => re assign and stop_count = 0 else stop_count ++.
                val_batch_gain = 1 - np.mean(val_losses)  # average of losses during evaluate
                if val_batch_gain > best_batch_gain:
                    best_batch_gain = val_batch_gain
                    stop_count = 0
                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    total_test_score += np.mean(test_losses)
                else:
                    stop_count += 1
                if stop_count == self.patience:
                    break
            average_test_score = total_test_score / n_test_batches 
            print(('epoch %i, test error of %f example is: %.2f') % (epoch, n_train_batches, average_test_score * 100.))
            print('epoch: %i, training time: %.2f secs; with cost: %.2f' % (epoch, time.time() - start, total_cost_train))

    def initNetwork(self):
        rng = np.random.RandomState(3435)
        cnet = list()
        for fh in self.filter_size:
            # feature map size is 100,2
            # image size is 300 * length of valid word in a sentence
            # pool_size result is the number of bag_of_words generated by picking windows
            # pool_size_width = img_width - filter_width + 1, pool_size_height = img_height - filter_height + 1
            # filter_shape: (number of filters, number of channel ,height, width) feature_size is the number of features,  stack_size is number of channel
            # input_shape: (batch_size, number of channel, image_height, image_width),
            conv_layer = ConvolutionLayer(rng, (self.hidden_units[0], 1, fh, self.img_width),
                                          (self.batch_size, 1, self.img_height, self.img_width), [self.img_height - fh + 1, 1])
            cnet.append(conv_layer)
            
        final_vector_dim = self.hidden_units[0] * len(self.filter_size)
        hidden_layer = HiddenLayer(rng, utils.Tanh, self.hidden_units[0], final_vector_dim)
        # hidden layer dropout still use weight and bias of hidden layer. It just
        # cuts off some neuron randomly with drop_out_rate
        hidden_layer_dropout = HiddenLayerDropout(rng, utils.Tanh, self.dropout_rate, final_vector_dim, final_vector_dim, hidden_layer.W, hidden_layer.b)
        # apply full connect layer to final vector
        full_connect = FullConnectLayer(rng, (final_vector_dim, 2))

        return cnet, hidden_layer, hidden_layer_dropout, full_connect

    def shared_dataset(self, data_xy, borrow=True):
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')