from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import pickle
import os.path as path
import utils
import theano

class Data:

    def __init__(self, src='../data/word_vectors.bin'):
        self.vectors = None
        self.vocabs = None
        self.src = src

    def initData(self):
        self.vocabs = dict()
        self.vocabs['UND'] = 0
        self.vectors = None

    def loadWordVectors(self):
        self.data = KeyedVectors.load_word2vec_format(self.src, binary=True)

    def loadWordVectorsFromText(self):
        self.initData()
        file_vectors = "vectors.txt"
        file_mapping = "vocabs.txt"
        if path.exists(file_vectors) and path.exists(file_mapping):
            with open(file_vectors, 'rb') as f:
                self.vectors = pickle.load(f)
            with open(file_mapping, 'rb') as f:
                self.vocabs = pickle.load(f)
        else: 
            with open(self.src, 'rb') as f:
                word_vectors = f.readlines()
                vectors_length = len(word_vectors[0].split(' ')) - 1
                tmp = list()
                tmp.append(np.zeros((1, vectors_length), dtype=theano.config.floatX))
                for index, w_v in enumerate(word_vectors): 
                    els = w_v.split(' ')
                    word = els[0]
                    els[-1] = els[-1].replace('\n', '')
                    vector = [[float(i) for i in els[1:]]]
                    tmp.append(np.array(vector, dtype=theano.config.floatX))
                    self.vocabs[word] = index
                self.vectors = np.array(tmp, dtype=theano.config.floatX)                    
                utils.save_file(file_vectors, self.vectors)
                utils.save_file(file_mapping, self.vocabs)