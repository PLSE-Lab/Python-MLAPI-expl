#!/usr/bin/env python
# coding: utf-8

# **BUILDING DATA CLEANING FUNCTIONS**

# In[ ]:


from collections import Counter
import numpy as np
import re
import os


# **Commonly used file reader and writer, change this to switch between python2 and python3.
# :param filename: filename
# :param mode: 'r' and 'w' for read and write respectively**

# In[ ]:


def open_file(filename, mode='r'):
    return open(filename, mode, encoding='utf-8', errors='ignore')


# **Tokenization/string cleaning for all datasets except for SST.
#     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py**

# In[ ]:


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# **Build vocabulary file from training data.**

# In[ ]:


def build_vocab(data, vocab_dir, vocab_size=8000):
    print('Building vocabulary...')

    all_data = []  # group all data
    for content in data:
        all_data.extend(content.split())

    counter = Counter(all_data)  # count and get the most common words
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))

    words = ['<PAD>'] + list(words)  # add a padding with id 0 to pad the sentence to same length
    open_file(vocab_dir, 'w').write('\n'.join(words) + '\n')


# In[ ]:


def read_vocab(vocab_file):
    """
    Read vocabulary from file.
    """
    words = open_file(vocab_file).read().strip().split('\n')
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


# **tokenizing and padding**

# In[ ]:


def process_text(text, word_to_id, max_length, clean=True):
    if clean:  # if the data needs to be cleaned
        text = clean_str(text)
    text = text.split()

    text = [word_to_id[x] for x in text if x in word_to_id]
    if len(text) < max_length:
        text = [0] * (max_length - len(text)) + text
    return text[:max_length]


# **Preprocessing training data.**

# In[ ]:


class Corpus(object):
    def __init__(self, pos_file, neg_file, vocab_file, dev_split=0.1, max_length=50, vocab_size=8000):
        # loading data
        pos_examples = [clean_str(s.strip()) for s in open_file(pos_file)]
        neg_examples = [clean_str(s.strip()) for s in open_file(neg_file)]
        x_data = pos_examples + neg_examples
        y_data = [0.] * len(pos_examples) + [1.] * len(neg_examples)  # 0 for pos and 1 for neg

        if not os.path.exists(vocab_file):
            build_vocab(x_data, vocab_file, vocab_size)

        self.words, self.word_to_id = read_vocab(vocab_file)

        for i in range(len(x_data)):  # tokenizing and padding
            x_data[i] = process_text(x_data[i], self.word_to_id, max_length, clean=False)

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # shuffle
        indices = np.random.permutation(np.arange(len(x_data)))
        x_data = x_data[indices]
        y_data = y_data[indices]

        # train/dev split
        num_train = int((1 - dev_split) * len(x_data))
        self.x_train = x_data[:num_train]
        self.y_train = y_data[:num_train]
        self.x_test = x_data[num_train:]
        self.y_test = y_data[num_train:]
    def __str__(self):
        return 'Training: {}, Testing: {}, Vocabulary: {}'.format(len(self.x_train), len(self.x_test), len(self.words))


# **IMPORTING DATA AND BUILDING CNN MODEL**

# In[ ]:


import mxnet as mx
from mxnet import gluon, autograd, metric
from mxnet import ndarray as nd
from mxnet.gluon import nn
from mxnet.gluon.data import Dataset, DataLoader
import numpy as np
from sklearn import metrics
import os
import time
from datetime import timedelta
import pandas as pd

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# *Importing Datasets*

# In[ ]:


pos_file = os.path.join("../input/rt-polarity.pos.txt")
neg_file = os.path.join("../input/rt-polarity.neg.txt")
vocab_file = os.path.join("../input/rt-polarity.vocab.txt")

# model save path
save_path = 'checkpoints'
if not os.path.exists(save_path):
    os.mkdir(save_path)
model_file = os.path.join(save_path, 'mr_cnn_mxnet.params')


# *Try GPU*

# In[ ]:


def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx


# In[ ]:


class TCNNConfig(object):
    embedding_dim = 128  # embedding vector size
    seq_length = 50  # maximum length of sequence
    vocab_size = 8000  # most common words

    num_filters = 100  # number of the convolution filters (feature maps)
    kernel_sizes = [3, 4, 5]  # three kinds of kernels (windows)

    dropout_prob = 0.5  # dropout rate
    learning_rate = 1e-3  # learning rate
    batch_size = 50  # batch size for training
    num_epochs = 10    # total number of epochs

    num_classes = 2  # number of classes

    dev_split = 0.1  # percentage of dev data


# In[ ]:


class Conv_Max_Pooling(nn.Block):
    def __init__(self, channels, kernel_size, **kwargs):
        super(Conv_Max_Pooling, self).__init__(**kwargs)

        with self.name_scope():
            self.conv = nn.Conv1D(channels, kernel_size)
            self.pooling = nn.GlobalMaxPool1D()

    def forward(self, x):
        output = self.pooling(self.conv(x))
        return nd.relu(output).flatten()


# In[ ]:


class TextCNN(nn.Block):
    def __init__(self, config, **kwargs):
        super(TextCNN, self).__init__(**kwargs)

        V = config.vocab_size
        E = config.embedding_dim
        Nf = config.num_filters
        Ks = config.kernel_sizes
        C = config.num_classes
        Dr = config.dropout_prob

        with self.name_scope():
            self.embedding = nn.Embedding(V, E)  # embedding layer

            # three different convolutional layers
            self.conv1 = Conv_Max_Pooling(Nf, Ks[0])
            self.conv2 = Conv_Max_Pooling(Nf, Ks[1])
            self.conv3 = Conv_Max_Pooling(Nf, Ks[2])
            self.dropout = nn.Dropout(Dr)  # a dropout layer
            self.fc1 = nn.Dense(C)  # a dense layer for classification

    def forward(self, x):
        x = self.embedding(x).transpose((0, 2, 1))  # Conv1D takes in NCW as input
        o1, o2, o3 = self.conv1(x), self.conv2(x), self.conv3(x)
        outputs = self.fc1(self.dropout(nd.concat(o1, o2, o3)))

        return outputs


# In[ ]:


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# In[ ]:


class MRDataset(Dataset):
    def __init__(self, x, y):
        super(MRDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index].astype(np.float32), self.y[index].astype(np.float32)

    def __len__(self):
        return len(self.x)


# In[ ]:


def evaluate(data_loader, data_len, model, loss, ctx):
    total_loss = 0.0
    acc = metric.Accuracy()

    for data, label in data_loader:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)

        with autograd.record(train_mode=False):  # set the training_mode to False
            output = model(data)
            losses = loss(output, label)

        total_loss += nd.sum(losses).asscalar()
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1], total_loss / data_len


# In[ ]:


def train():
    print("Loading data...")
    start_time = time.time()
    config = TCNNConfig()
    corpus = Corpus(pos_file, neg_file, vocab_file, config.dev_split, config.seq_length, config.vocab_size)
    print(corpus)
    config.vocab_size = len(corpus.words)

    print("Configuring CNN model...")
    ctx = try_gpu()
    model = TextCNN(config)
    model.collect_params().initialize(ctx=ctx)
    print("Initializing weights on", ctx)
    print(model)

    # optimizer and loss function
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': config.learning_rate})

    batch_size = config.batch_size
    train_loader = DataLoader(MRDataset(corpus.x_train, corpus.y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MRDataset(corpus.x_test, corpus.y_test), batch_size=batch_size, shuffle=False)

    print("Training and evaluating...")
    best_acc = 0.0
    for epoch in range(config.num_epochs):
        for data, label in train_loader:
            data, label = data.as_in_context(ctx), label.as_in_context(ctx)

            with autograd.record(train_mode=True):  # set the model in training mode
                output = model(data)
                losses = loss(output, label)

            # backward propagation and update parameters
            losses.backward()
            trainer.step(len(data))

        # evaluate on both training and test dataset
        train_acc, train_loss = evaluate(train_loader, len(corpus.x_train), model, loss, ctx)
        test_acc, test_loss = evaluate(test_loader, len(corpus.x_test), model, loss, ctx)

        if test_acc > best_acc:
            # store the best result
            best_acc = test_acc
            improved_str = '*'
            model.save_params(model_file)
        else:
            improved_str = ''

        time_dif = get_time_dif(start_time)
        msg = "Epoch {0:3}, Train_loss: {1:>7.2}, Train_acc {2:>6.2%}, "               + "Test_acc {4:>6.2%}, Time: {5} {6}"
        print(msg.format(epoch + 1, train_loss, train_acc, test_loss, test_acc, time_dif, improved_str))

    test(model, test_loader, ctx)


# In[ ]:


def test(model, test_loader, ctx):
    print("Testing...")
    start_time = time.time()
    model.load_params(model_file, ctx=ctx)   # restore the best parameters

    y_pred, y_true = [], []
    for data, label in test_loader:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        with autograd.record(train_mode=False):  # set the training_mode to False
            output = model(data)
        pred = nd.argmax(output, axis=1).asnumpy().tolist()
        y_pred.extend(pred)
        y_true.extend(label.asnumpy().tolist())

    test_acc = metrics.accuracy_score(y_true, y_pred)
    test_f1 = metrics.f1_score(y_true, y_pred, average='macro')
    print("Test accuracy: {0:>7.2%}, F1-Score: {1:>7.2%}".format(test_acc, test_f1))

    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_true, y_pred, target_names=['POS', 'NEG']))

    print('Confusion Matrix...')
    cm = metrics.confusion_matrix(y_true, y_pred)
    print(cm)

    print("Time usage:", get_time_dif(start_time))


# In[ ]:


def predict(text):
    # load config and vocabulary
    print('\npredicting....')
    config = TCNNConfig()
    _, word_to_id = read_vocab(vocab_file)
    labels = ['POS', 'NEG']

    # load model
    ctx = try_gpu()
    model = TextCNN(config)
    model.load_params(model_file, ctx=ctx)

    # process text
    print(text)
    text = process_text(text, word_to_id, config.seq_length)
    text = nd.array([text]).as_in_context(ctx)

    output = model(text)
    pred = nd.argmax(output, axis=1).asscalar()
    print("Polarity:",output[0,0])
    print("class:",labels[int(pred)])


# In[ ]:


if __name__ == '__main__':
    train()
    predict("If you sometimes like to go to the movies to have fun, wasabi is a good place to start")
    predict("This is a film well worth seeing, talking and singing heads and all")
    predict("One of the greatest family oriented, fantasy- adventure movies ever")
    predict("sadly , though many of the actors throw off a spark or two when they first appear , they can't generate enough heat in this cold vacuum of a comedy to start a reaction")
    predict("It was a taut, intelligent and psychological drama")
    predict("it's so laddish and juvenile , only teenage boys could possibly find it funny")
    predict("I really hate this movie")
    predict("This is a good movie")

