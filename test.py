from model import Model
import argparse
import os.path
import utils

from data import Data


word_vectors, vocabs = None, None


def it(word_vector, word_vector_preload, sent, test_path, word_vector, dimension, maxlen):
    global word_vectors, vocabs
    if word_vectors is None or vocabs is None:
        word_vectors, vocabs = loadWordVectors(word_vector, word_vector_preload)
    if not test_path:
        if sent: 
            words = sent.split(' ')
            sent_length = len(words)
            if sent_length < 5:
                sent_length = 5
            test_x = make_sentence_idx(vocabs, [words], sent_length)
            test_y = [1]
            model = Model(word_vectors, img_width=dimension, img_height=sent_length, batch_size=1)
            pred = model.build_test_model((test_x, test_y, sent_length))            
            if pred:
                print "sentiment is positive"
            else: 
                print "sentiment is negative"
    else: 
        #auto test path_file
        test_x, test_y = utils.load_file(test_path)
        model = Model(word_vectors, img_width=dimension, img_height=maxlen)
        errors = model.build_test_model((test_x, test_y, maxlen))
        print(errors)


parser = argparse.ArgumentParser(description='Running CNN only')
parser.add_argument('--vectors', type=str, default='/home/alex/Documents/nlp/data/glove.6B.50d.txt')
parser.add_argument('--plvec', type=str, default='/home/alex/Documents/nlp/data')
parser.add_argument('--test', type=str, default='/home/alex/Documents/nlp/code/data/50d.test_twitter.txt')
parser.add_argument('--sent', type=str, default='')
parser.add_argument('--width', type=int, default=50)
parser.add_argument('--max', type=int, default=140)
parser.add_argument('--patient', type=int, default=20)
args = parser.parse_args()


id(args.vectors, args.plvec, args.test, args.sent, args.width, args.max)
