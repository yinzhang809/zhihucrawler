#!/usr/bin/python
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn import metrics
from data_helper.corpus import read_vocab, process_text
# from data_helper.mr_loader import MRCorpus
from data_helper.smp_loader import SMPCorpus
import pandas as pd

import os
import time
from datetime import timedelta

base_dir = 'data/mr'
pos_file = os.path.join(base_dir, 'rt-polarity.pos.txt')
neg_file = os.path.join(base_dir, 'rt-polarity.neg.txt')
# vocab_file = os.path.join(base_dir, 'rt-polarity.vocab.txt')
vocab_file = os.path.join('./data/', 'voc.txt')

save_path = 'checkpoints'  # model save path
if not os.path.exists(save_path):
    os.mkdir(save_path)
model_file = os.path.join(save_path, 'mr_cnn_pytorch.pt')

use_cuda = True


class TCNNConfig(object):
    """
    CNN Parameters
    """
    embedding_dim = 300
    seq_length = 5
    vocab_size = 4000

    num_filters = 156
    kernel_sizes = [1, 2, 3]

    hidden_dim = 50

    dropout_prob = 0.3
    learning_rate = 1e-3
    batch_size = 500
    num_epochs = 10

    num_classes = 6

    # dev_split = 0.1


class TextCNN(nn.Module):

    def __init__(self, config):
        super(TextCNN, self).__init__()

        V = config.vocab_size
        E = config.embedding_dim
        Nf = config.num_filters
        Ks = config.kernel_sizes
        C = config.num_classes
        Dr = config.dropout_prob

        self.embedding = nn.Embedding(V, E)

        self.convs = nn.ModuleList([nn.Conv1d(E, Nf, k) for k in Ks])
        self.dropout = nn.Dropout(Dr)
        self.fc1 = nn.Linear(3 * Nf, C)

    @staticmethod
    def conv_and_max_pool(x, conv):
        return F.relu(conv(x).permute(0, 2, 1).max(1)[0])

    def forward(self, inputs):
        embedded = self.embedding(inputs).permute(0, 2, 1)
        x = [self.conv_and_max_pool(embedded, k) for k in self.convs]
        x = self.fc1(self.dropout(torch.cat(x, 1)))

        return x


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(data, model, loss):
    model.eval()
    data_loader = DataLoader(data, batch_size=50)

    data_len = len(data)
    total_loss = 0.0
    y_true, y_pred = [], []

    for data, label in data_loader:
        data, label = Variable(data, volatile=True), Variable(label, volatile=True)
        if use_cuda:
            data, label = data.cuda(), label.cuda()

        output = model(data)
        losses = loss(output, label)

        total_loss += losses.data[0]
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data)

    acc = (np.array(y_true) == np.array(y_pred)).sum()
    return acc / data_len, total_loss / data_len


def train():
    print('Loading data...')
    start_time = time.time()
    config = TCNNConfig()
    corpus = SMPCorpus(vocab_file, max_length=5, vocab_size=config.vocab_size)
    print(corpus)
    config.vocab_size = len(corpus.words)

    train_data = TensorDataset(torch.LongTensor(corpus.x_train), torch.LongTensor(corpus.y_train))
    valid_data = TensorDataset(torch.LongTensor(corpus.x_valid), torch.LongTensor(corpus.y_valid))
    test_data=TensorDataset(torch.LongTensor(corpus.x_test),torch.LongTensor(corpus.y_test))

    print('Configuring CNN model...')
    model = TextCNN(config)
    print(model)

    if use_cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    print("Training and evaluating...")

    best_acc = 0.0
    for epoch in range(config.num_epochs):
        model.train()
        train_loader = DataLoader(train_data, batch_size=config.batch_size)
        for x_batch, y_batch in train_loader:
            inputs, targets = Variable(x_batch), Variable(y_batch)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

        train_acc, train_loss = evaluate(train_data, model, criterion)
        test_acc, test_loss = evaluate(valid_data, model, criterion)

        if test_acc > best_acc:
            best_acc = test_acc
            improved_str = '*'
            torch.save(model.state_dict(), model_file)
        else:
            improved_str = ''

        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_acc {2:>6.2%}, " \
              + "Test_loss: {3:>6.2}, Test_acc {4:>6.2%}, Time: {5} {6}"
        print(msg.format(epoch + 1, train_loss, train_acc, test_loss, test_acc, time_dif, improved_str))

    test(model, valid_data)
    predict(model,test_data)


def test(model, test_data):
    print("Testing...")
    start_time = time.time()
    test_loader = DataLoader(test_data, batch_size=50)

    # restore the best parameters
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    y_true, y_pred = [], []
    for data, label in test_loader:
        data, label = Variable(data, volatile=True), Variable(label, volatile=True)
        if use_cuda:
            data, label = data.cuda(), label.cuda()

        output = model(data)
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.data)

    test_acc = metrics.accuracy_score(y_true, y_pred)
    test_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    print("Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))

    # print("Precision, Recall and F1-Score...")
    # print(metrics.classification_report(y_true, y_pred, target_names=['POS', 'NEG']))
    #
    # print('Confusion Matrix...')
    # cm = metrics.confusion_matrix(y_true, y_pred)
    # print(cm)
    #
    print("Time usage:", get_time_dif(start_time))


def predict(model,test_data):
    print("Testing...")
    start_time = time.time()
    test_loader = DataLoader(test_data, batch_size=50)

    # restore the best parameters
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))

    y_true, y_pred = [], []
    for data, label in test_loader:
        data, label = Variable(data, volatile=True), Variable(label, volatile=True)
        if use_cuda:
            data, label = data.cuda(), label.cuda()

        output = model(data)
        pred = torch.max(output.data, dim=1)[1].cpu().numpy().tolist()
        y_pred.extend(pred)
    df=pd.DataFrame(y_pred)
    df.to_csv("test_label_300.csv",index=None,header=None)


if __name__ == '__main__':
    train()
    # config=TCNNConfig()
    # model=TextCNN(config)
    # model.load_state_dict(torch.load(model_file,map_location=lambda storage, loc: storage))
    # corpus = SMPCorpus(vocab_file, max_length=5, vocab_size=config.vocab_size)
    # print(corpus)
    # config.vocab_size = len(corpus.words)
    # model.eval()
    #
    # valid_data = TensorDataset(torch.LongTensor(corpus.x_valid), torch.LongTensor(corpus.y_valid))
    # test_data=TensorDataset(torch.LongTensor(corpus.x_test),torch.LongTensor(corpus.y_test))
    # test(model,valid_data)
    # predict(model,test_data)
