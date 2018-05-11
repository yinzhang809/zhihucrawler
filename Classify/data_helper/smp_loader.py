#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os
from .corpus import clean_str, open_file, build_vocab, read_vocab, process_text
import jieba
from sklearn.model_selection import train_test_split
import pandas as pd
import json

def load_train_data(file_path):
    raw_data = []
    label = []
    content = open(file_path)
    jsonString=json.load(content)
    for pair in jsonString:
        raw_data.append(pair[0][0].replace(' ','').replace('\n',''))
        label.append(pair[0][1])
        label.append(pair[1][1])
        raw_data.append(pair[1][0].replace(' ', '').replace('\n', ''))
    while 1:
        line = content.readline()
        if not line:
            break
        line = line.replace('。', '').replace('?', '').replace('\n', '')
        raw_data.append(line)
    return raw_data,label

def load_test_data(filepath):
    raw_data=[]
    test_data = open(filepath)
    jsonString=json.load(test_data)
    for pair in jsonString:
        raw_data.append(pair['post'])
        raw_data.append(pair['res'])
    return raw_data


class SMPCorpus(object):
    """
    Preprocessing training data.
    """

    def __init__(self, vocab_file, max_length=5, vocab_size=5000):
        # loading data
        train_x,train_y=load_train_data('../../data/train_data.json')
        test_x=load_test_data("../../zhihu_processed.json")
        train_x,valid_x,train_y,valid_y=train_test_split(train_x,train_y,test_size=0.3,random_state=341)


        self.build_dataset(train_x, train_y, valid_x, valid_y, test_x, vocab_file, max_length, vocab_size)

    def build_dataset(self, train_x, train_y, valid_x, valid_y, test_x,vocab_file, max_length, vocab_size):
        # y_set = list(set(train_y + valid_y))
        # self.label2id = {}
        # for i in range(len(y_set)):
        #     self.label2id[y_set[i]] = float(i)
        y_train = train_y
        y_valid = valid_y

        x_train = []

        for item in train_x:
            seg_list = jieba.cut(item)
            x_train.append(" ".join(seg_list))

        x_valid = []

        for item in valid_x:
            seg_list = jieba.cut(item)
            x_valid.append(" ".join(seg_list))

        x_test = []

        for item in test_x:
            seg_list = jieba.cut(item)
            x_test.append(" ".join(seg_list))


        build_vocab(x_train + x_valid + x_test, vocab_file, vocab_size)
        self.words, self.word2id = read_vocab(vocab_file)

        for i in range(len(x_train)):
            x_train[i] = process_text(x_train[i], self.word2id, max_length, clean=False)

        for i in range(len(x_valid)):
            x_valid[i] = process_text(x_valid[i], self.word2id, max_length, clean=False)

        for i in range(len(x_test)):
            x_test[i] = process_text(x_test[i], self.word2id, max_length, clean=False)



        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_valid = np.array(x_valid)
        y_valid = np.array(y_valid)

        x_test=np.array(x_test)
        y_test=np.array(np.zeros(np.shape(x_test)[0]))

        indices = np.random.permutation(np.arange(len(x_train)))
        x_train = x_train[indices]
        y_train = y_train[indices]

        indices = np.random.permutation(np.arange(len(x_valid)))
        x_valid = x_valid[indices]
        y_valid = y_valid[indices]

        indices = np.random.permutation(np.arange(len(x_test)))
        x_test = x_test[indices]
        y_test = y_test[indices]


        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test

    def __str__(self):
        return 'Training: {}, Validing: {} Testing: {}, Vocabulary: {}'.format(len(self.x_train), len(self.x_valid),len(self.x_test), len(self.words))
