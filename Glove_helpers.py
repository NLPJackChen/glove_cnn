# -*- coding: utf-8 -*-

'''
python Glove_helpers.py input_file output_model_file output_vector_file
'''

# import modules & set up logging
import numpy as np
import os
import sys
import logging
import multiprocessing
import time
import re
import json


def output_vocab(vocab):
    for k, v in vocab.items():
        print(k)

def embedding_sentences(sentences, embedding_size=100 ,window=5, min_count=5, file_to_load=None, file_to_save=None):
    glove_word = {}
    with open('glove_word.txt', 'r') as r1:
        for i in r1.readlines():

            string = re.search(r"[\u4e00-\u9fff]*", i)
            a = string.group()
            string = re.sub(r"[\u4e00-\u9fff]+.*? ", '', i)
            c = []
            count = 0
            for i in string:
                if i == ' ':
                    b = string[count:]
                    index = b.index(i)
                    c.append(float(b[0:index]))
                    count = index + count + 1
            c.append(float(string[count:]))
            d = np.array(c)
            glove_word[a] = d

    all_vectors = []
    embeddingDim = 100
    embeddingUnknown = [0 for i in range(embeddingDim)]
    for sentence in sentences:
        this_vector = []
        for word in sentence:
            if word in glove_word:
                this_vector.append(glove_word[word])
            else:
                this_vector.append(embeddingUnknown)
        all_vectors.append(this_vector)
    return all_vectors


def run_main():
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 4:
        print (globals()['__doc__'] % locals())
        sys.exit(1)
    input_file, output_model_file, output_vector_file = sys.argv[1:4]

def test():
    vectors = embedding_sentences([['first', 'sentence'], ['second', 'sentence']], embedding_size=4, min_count=1)
