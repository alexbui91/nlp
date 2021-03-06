from model import Model
import pickle
import numpy as np
import theano

import os.path
import sys
import utils

from data import Data

word_vectors, vocabs = None, None

def process_data(data, isGetFirst=True):
    default_del = '|999999|'
    #results_x with size: sentence_size * sentences_words_length * word_vector_size
    results_y = list()
    sents = list()
    max_sent_length = 0
    sent_length = 0
    words = None
    # word_indices = None
    cols = None
    c1 = 0
    for row in data:
        words = None
        if isGetFirst:
            cols = row.split(default_del)
            c1 = int(cols[0])
            if c1 == 4:
                c1 = 1
            else:
                c1 = 0
            results_y.append(c1)
            words = cols[-1].split(' ')
            words[-1].replace('\n', '')
        else: 
            words = row.split(' ')
            words[-1].replace('\n', '')
        sent_length = len(words)
        # word_indices = list()
        # for w in words:
        #     if w in vocabs:
        #         word_indices.append(vocabs[w])
        # if word_indices:
        #     sents.append([word_indices])
        sents.append(words)
        if sent_length > max_sent_length:
            max_sent_length = sent_length
    return results_y, sents, max_sent_length


def make_sentence_idx(vocabs, sents, max_sent_length):
    results_x = list()
    w_vector = None
    for sent in sents:
        sent_length = len(sent)
        sent_v = list()
        for i in xrange(max_sent_length):
            if i < sent_length:
                if sent[i] in vocabs:
                    sent_v.append(vocabs[sent[i]])
                else: 
                    sent_v.append(0)
            else:
                sent_v.append(0)
        if sent_v: 
            results_x.append(sent_v)
    return results_x


def loadWordVectors():
    d = Data('../data/glove_text8.txt')
    # d.loadWordVectors()
    d.loadWordVectorsFromText()
    return d.vectors, d.vocabs


def init_test(test_path=''):
    global word_vectors, vocabs
    if word_vectors is None or vocabs is None:
        word_vectors, vocabs = loadWordVectors()
    if not test_path:
        while True:
            sent = input("Give me a sentence: ")
            words = sent.split(' ')
            test_x = make_sentence_idx(vocabs, [words], 1)
            model = Model(img_width=50, img_height=1)
            y_pred = model.build_test_model((test_x, None, 1))
            print(y_pred)
    else: 
        #auto test path_file
        with open(test_path, 'r') as test:
            test_data = test.readlines()
            test_y, test_sent, test_len = process_data(test_data, False)
            test_x = make_sentence_idx(vocabs, test_sent, test_len)
            model = Model(img_width=50, img_height=test_len)
            y_pred = model.build_test_model((test_x, test_y, test_len))
            print(y_pred)


def exe(path = '../data/', training_path='training_twitter_small.txt', dev_path='dev_twitter_small.txt', test_path='test_twitter.txt'):
    # you can modify this data path. Currently, this path is alongside with code directory
    global word_vectors, vocabs 
    datafile = 'data/sentiment_dataset.txt'
    training_path = path + training_path
    dev_path = path + dev_path
    test_path = path + test_path
    if word_vectors is None or vocabs is None:
        word_vectors, vocabs = loadWordVectors()
    img_width = 50
    if os.path.exists(datafile):
        with open(datafile, 'rb') as f:
            dataset = pickle.load(f)
            model = Model(word_vectors, dataset['train'], dataset['dev'], dataset['test'], img_width, dataset['max_sent_length'], epochs=1)
            model.trainNet()
    else:
        with open(training_path, 'r') as train, open(dev_path, 'r') as dev, open(test_path, 'r') as test:
            training_data = train.readlines()
            dev_data = dev.readlines()
            test_data = test.readlines()
            train_y, train_sent, train_len = process_data(training_data)
            dev_y, dev_sent, dev_len = process_data(dev_data)
            test_y, test_sent, test_len = process_data(test_data)
            max_sent_length = utils.find_largest_number(train_len, dev_len, test_len)

            train_x = make_sentence_idx(vocabs, train_sent, max_sent_length)
            del(train_sent)
            dev_x = make_sentence_idx(vocabs, dev_sent, max_sent_length)
            del(dev_sent)
            test_x = make_sentence_idx(vocabs, test_sent, max_sent_length)
            del(test_sent)
            dataset = dict()

            dataset['train'] = (train_x, train_y)
            dataset['test'] = (test_x, test_y)
            dataset['dev'] = (dev_x, dev_y)
            dataset['max_sent_length'] = max_sent_length
            del train_x, train_y, test_x, test_y, dev_x, dev_y, max_sent_length
            utils.save_file(datafile, dataset)
            model = Model(word_vectors, dataset['train'], dataset['dev'], dataset['test'], img_width, dataset['max_sent_length'], epochs=1)
            model.trainNet()
