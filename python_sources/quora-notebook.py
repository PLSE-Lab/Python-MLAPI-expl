#!/usr/bin/env python
# coding: utf-8

# **SUMMARY**
# 
# *LSTM:*
# F1 score: 0.6622
# Time elapsed: 1273.20
# Accuracy: 0.9562
# 
# *FFM:*
# F1 score: 0.6049
# Time elapsed: 184.46
# Accuracy: 0.9473

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import math
import random
import time
import os
print(os.listdir("../input"))

df = pd.read_csv("../input/train.csv")
# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import *
from keras.models import *
from keras import initializers, regularizers, constraints, optimizers, layers

import torch
import torchtext
from torch import nn
import torch.nn.functional as F
from nltk import word_tokenize
from torch import optim


# In[ ]:



max_len = 50

text = torchtext.data.Field(lower=True, batch_first=True, tokenize=word_tokenize, fix_length = max_len)
target = torchtext.data.Field(sequential=False, use_vocab=False, is_target=True)
train = torchtext.data.TabularDataset(path='../input/train.csv', format='csv',
                                      fields={'question_text': ('text',text),
                                              'target': ('target',target)})


# In[ ]:


text.build_vocab(train, min_freq=1)
text.vocab.load_vectors(torchtext.vocab.Vectors("../input/embeddings/glove.840B.300d/glove.840B.300d.txt"))


# In[ ]:


random_state = random.getstate()
train, val = train.split(split_ratio=0.8, random_state=random_state)
batch_size = 512
train_iter = torchtext.data.BucketIterator(dataset=train,
                                           batch_size=batch_size,
                                           sort_key=lambda x: x.text.__len__(),
                                           shuffle=True,
                                           sort=False)

val_iter = torchtext.data.BucketIterator(dataset=val,
                                         batch_size=batch_size,
                                         sort_key=lambda x: x.text.__len__(),
                                         train=False,
                                         sort=False)


# In[ ]:


def training(epoch, model, loss_func, optimizer, train_iter, val_iter):
    step = 0
    train_record = []
    val_record = []
    losses = []
    
    for e in range(epoch):
        train_iter.init_epoch()
        for train_batch in iter(train_iter):
            step += 1
            model.train()
            x = train_batch.text.cuda()
            y = train_batch.target.type(torch.Tensor).cuda()
            model.zero_grad()
            pred = model.forward(x).view(-1)
            #print('Pred:{}'.format(pred.shape))
            #print('y:{}'.format(y.shape))
            
            loss = loss_function(pred, y)
            loss_data = loss.cpu().data.numpy()
            train_record.append(loss_data)
            loss.backward()
            optimizer.step()
            if step % 1000 == 0:
                print("Step: {:06}, loss {:.4f}".format(step, loss_data))
        model.eval()
        model.zero_grad()
        val_loss = []
        for val_batch in iter(val_iter):
            val_x = val_batch.text.cuda()
            val_y = val_batch.target.type(torch.Tensor).cuda()
            val_pred = model.forward(val_x).view(-1)
            val_loss.append(loss_function(val_pred, val_y).cpu().data.numpy())
        val_record.append({'step': step, 'loss': np.mean(val_loss)})
        print('Epoch {:02} - step {:06} - train_loss {:.4f} - val_loss {:.4f} '.format(
                    e, step, np.mean(train_record), val_record[-1]['loss']))
        train_record = []


# In[ ]:


def results(m, t):
    model = m
    model.eval()
    val_pred = []
    val_true = []
    val_iter.init_epoch()
    for val_batch in iter(val_iter):
        val_x = val_batch.text.cuda()
        val_true += val_batch.target.data.numpy().tolist()
        val_pred += torch.sigmoid(model.forward(val_x).view(-1)).cpu().data.numpy().tolist()

    tmp = [0,0,0] # idx, cur, max
    delta = 0
    for tmp[0] in np.arange(0.1, 0.501, 0.01):
        tmp[1] = metrics.f1_score(val_true, np.array(val_pred)>tmp[0])
        if tmp[1] > tmp[2]:
            delta = tmp[0]
            tmp[2] = tmp[1]

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    total = len(val_pred)
    for i in range(0,len(val_pred)):
        pred = val_pred[i] > delta
        if val_true[i] == 1:
            if pred == 1:
                tp += 1
            else:
                fp += 1
        else:
            if pred == 1:
                fn += 1
            else:
                tn += 1

    print('----TIME FOR SOME STATISCTICS!!!!----')
    print('-------------{} MODEL--------------'.format(model.name))
    print('best threshold is {:.4f} with F1 score: {:.4f}'.format(delta, tmp[2]))
    print('Time elapsed: {:.2f}'.format(time.time() - t))
    print('True Positive: {}'.format(tp))
    print('False Positive: {}'.format(fp))
    print('False Negative: {}'.format(fn))
    print('True Negative: {}'.format(tn))
    print('Accuracy: {:.4f}'.format((tp+tn)/float(total)))
    print('Precision: {:.4f}'.format(tp/(float(tp+fp))))
    print('False positive rate: {:.4f}'.format(fp/(float(tn+fp))))
    print('Recall: {:.4f}'.format(tp/(float(tp+fn))))


# In[ ]:


class LSTM(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, hidden_dim = 128, static=True):
        super(LSTM, self).__init__()
        self.name = 'LSTM'
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=0.5)
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=self.embedding.embedding_dim,
                            hidden_size=hidden_dim,
                            num_layers=2, 
                            dropout = 0.5)
        self.hidden2label = nn.Linear(hidden_dim*2, 1)
    
    def forward(self, sents):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        y = self.hidden2label(self.dropout(torch.cat([c_n[i,:, :] for i in range(c_n.shape[0])], dim=1)))
        return y


# In[ ]:


lstm = LSTM(text.vocab.vectors,
                    padding_idx=text.vocab.stoi[text.pad_token], hidden_dim=128).cuda()
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, lstm.parameters()),lr=0.0001)

t = time.time()

training(model=lstm,
         epoch=10,
         loss_func=loss_function,
         optimizer=optimizer,
         train_iter=train_iter,
         val_iter=val_iter)


# In[ ]:


results(lstm,t)


# In[ ]:


class CNN(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True):
        super(CNN, self).__init__()
        self.name = 'CNN'
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        filter_sizes = [1,2,3,5]
        if static:
            self.embedding.weight.requires_grad = False
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels = 300, out_channels = 1, kernel_size = filter_sizes[3]),
            nn.MaxPool1d(kernel_size = 2)
        )
        self.lin = nn.Linear(23,64)
        self.fc = nn.Linear(64,1)
        
        
    def forward(self, sents):
        x = self.embedding(sents)
        x = torch.transpose(x, dim0=2, dim1=1)
        c1 = self.conv1(x)
        '''c2 = self.conv2(x)
        c3 = self.conv3(x)
        c4 = self.conv4(x)
        x = torch.cat((c1,c2,c3,c4), dim=1)'''
        x = c1
        x = nn.Dropout()(x)
        x = x.reshape(x.size(0), -1)
        x = self.lin(x)
        x = nn.Dropout()(x)
        x = self.fc(x)
        return x


# In[ ]:


cnn = CNN(text.vocab.vectors,
                    padding_idx=text.vocab.stoi[text.pad_token]).cuda()
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, cnn.parameters()),lr=0.0001)

t = time.time()

training(model=cnn,
         epoch=10,
         loss_func=loss_function,
         optimizer=optimizer,
         train_iter=train_iter,
         val_iter=val_iter)


# In[ ]:


results(cnn, t)


# In[ ]:


class FFM(nn.Module):
    def __init__(self, pretrained_lm, padding_idx, static=True):
        super(FFM, self).__init__()
        self.name = 'FFM'
        self.embedding = nn.Embedding.from_pretrained(pretrained_lm)
        self.embedding.padding_idx = padding_idx
        if static:
            self.embedding.weight.requires_grad = False
        self.layer1 = nn.Sequential(
            nn.Linear(15000,1000),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(1000,600),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(600,200),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(200,90),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.Linear(90,32),
            nn.ReLU()
        )
        self.fc = nn.Linear(32,1)
        
        
    def forward(self, sents):
        x = self.embedding(sents)
        x = torch.cat([x[:, i, :] for i in range(x.shape[1])], dim=1)
        x = self.layer1(x)
        x = nn.Dropout()(x)
        x = self.layer2(x)
        x = nn.Dropout()(x)
        x = self.layer3(x)
        x = nn.Dropout()(x)
        x = self.layer4(x)
        x = nn.Dropout()(x)
        x = self.layer5(x)
        x = nn.Dropout()(x)
        x = self.fc(x)
        return x


# In[ ]:


ffm = FFM(text.vocab.vectors,
                    padding_idx=text.vocab.stoi[text.pad_token]).cuda()
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, ffm.parameters()),lr=0.0001)

t = time.time()

training(model=ffm,
         epoch=2,
         loss_func=loss_function,
         optimizer=optimizer,
         train_iter=train_iter,
         val_iter=val_iter)


# In[ ]:


results(ffm, t)


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)


# In[ ]:


train_df['len'] = train_df['question_text'].apply(lambda x: len(x.split(' ')))


# In[ ]:


train_df.describe()


# In[ ]:


def SimpleFFModel:
    model=Sequential() # Instantiate the Sequential class
    model.add(Embedding(max_features, 300, input_length=maxlen,  weights=[embedding_matrix], trainable=False)) # Creat embedding layer as described above
    model.add(layers.Flatten()) #Flatten the embedding layer as input to a Dense layer
    model.add(layers.Dense(1000, activation='relu')) # Dense layer with relu activation
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(600, activation='relu')) # Dense layer with relu activation
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(200, activation='relu')) # Dense layer with relu activation
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(90, activation='relu')) # Dense layer with relu activation
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu')) # Dense layer with relu activation
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1,activation='sigmoid')) # Dense layer with sigmoid activation for binary target
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy']) #binary cross entropy is used as the loss function and accuracy as the metric 
    return model


# In[ ]:


def ConvModel:
    inp = Input(shape=(maxlen, ))
    embed = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    #filter_sizes = [1,2,3,5]
    filter_sizes = [5]
    num_filters = 64

    conv_0 = Conv1D(num_filters, filter_sizes[0], padding='valid', kernel_initializer='normal', activation='relu')(embed)
    #conv_1 = Conv1D(num_filters, filter_sizes[1], padding='valid', kernel_initializer='normal', activation='relu')(embed)
    #conv_2 = Conv1D(num_filters, filter_sizes[2], padding='valid', kernel_initializer='normal', activation='relu')(embed)
    #conv_3 = Conv1D(num_filters, filter_sizes[3], padding='valid', kernel_initializer='normal', activation='relu')(embed)

    maxpool_0 = MaxPool1D(pool_size=(maxlen - filter_sizes[0] + 1), strides=(1), padding='valid')(conv_0)
    #maxpool_1 = MaxPool1D(pool_size=(maxlen - filter_sizes[1] + 1), strides=(1), padding='valid')(conv_1)
    #maxpool_2 = MaxPool1D(pool_size=(maxlen - filter_sizes[2] + 1), strides=(1), padding='valid')(conv_2)
    #maxpool_3 = MaxPool1D(pool_size=(maxlen - filter_sizes[3] + 1), strides=(1), padding='valid')(conv_3)

    #concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2, maxpool_3])

    #x = Flatten()(concatenated_tensor)
    x = Flatten()(maxpool_0)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outp = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])    
    return model


# In[ ]:


from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity

print(1 - spatial.distance.cosine(embs_index["White"], embs_index["Black"]))

