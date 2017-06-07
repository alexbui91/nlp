from model import Model
import argparse
import os.path
import utils

from data import Data

word_vectors, vocabs = None, None

# main.exe(path = '../data/', word_vector='glove.6B.300d.txt', training_path='training_twitter.txt', dev_path='dev_twitter.txt', test_path='test_twitter.txt', img_width=300, epochs=11, patience=20)
def exe(word_vectors_file, vector_preloaded_path, train_path, dev_path, test_path, img_width, maxlen, epochs, patience):
    # you can modify this data path. Currently, this path is alongside with code directory
    global word_vectors, vocabs
    if os.path.exists(train_path) and os.path.exists(dev_path) and os.path.exists(test_path):
        train = utils.load_file(train_path)
        dev = utils.load_file(dev_path)
        test = utils.load_file(test_path)
    else: 
        raise NotImplementedError()
    if word_vectors is None or vocabs is None:
        word_vectors, vocabs = utils.loadWordVectors(word_vectors_file, vector_preloaded_path)
    model = Model(word_vectors, train, dev, test, img_width, maxlen, epochs=epochs, patience=patience)
    model.trainNet()


parser = argparse.ArgumentParser(description='Running CNN only')
parser.add_argument('--vectors', type=str, default='/home/alex/Documents/nlp/data/glove.6B.50d.txt')
parser.add_argument('--plvec', type=str, default='/home/alex/Documents/nlp/data')
parser.add_argument('--train', type=str, default='/home/alex/Documents/nlp/code/data/50d.training_twitter_small.txt')
parser.add_argument('--dev', type=str, default='/home/alex/Documents/nlp/code/data/50d.dev_twitter_small.txt')
parser.add_argument('--test', type=str, default='/home/alex/Documents/nlp/code/data/50d.test_twitter.txt')
parser.add_argument('--width', type=int, default=50)
parser.add_argument('--max', type=int, default=140)
parser.add_argument('--patient', type=int, default=20)
parser.add_argument('--epochs', type=int, default=20)

args = parser.parse_args()


exe(args.vectors, args.plvec, args.train, args.dev, args.test, args.width, args.max, args.epochs, args.patient)
