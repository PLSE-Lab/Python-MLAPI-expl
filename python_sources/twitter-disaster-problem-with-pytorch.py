#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import torch
import torchtext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import spacy

nlp = spacy.load("en")


# In[ ]:


def is_token_allowed(token):
    '''
    Only allow valid tokens which are not stop words
    and punctuation symbols.
    '''
    if (not token or not token.string.strip() or
        token.is_stop or token.is_punct):
        return False
    return True

def preprocess_token(token):
    # Reduce token to its lowercase lemma form
    return token.lemma_.strip().lower()

def tokenizer(text):
    return [ preprocess_token(token) for token in nlp.tokenizer(text) if is_token_allowed(token)]

from torchtext import data

text = data.Field(
        tokenize=tokenizer,
        pad_first=True
)


train_dataset = data.TabularDataset(
            path="/kaggle/input/nlp-getting-started/train.csv", format='csv',
            fields=[('id', data.Field()),
                    ('keyword', text),
                    ('location', data.Field()),
                    ('text', text),
                    ('target', data.Field())], 
            skip_header=True)

test_dataset = data.TabularDataset(
            path="/kaggle/input/nlp-getting-started/test.csv", format='csv',
            fields=[('id', data.Field()),
                    ('keyword', text),
                    ('location', data.Field()),
                    ('text', text)], 
            skip_header=True)

MIN_FREQ = 2

text.build_vocab(train_dataset, test_dataset, min_freq=MIN_FREQ)

VOCAB_SIZE = len(text.vocab)


# In[ ]:


NGRAMS = 2
BATCH_SIZE = 8
EMBED_DIM = 768

import torch.nn as nn
import torch.nn.functional as F

# This model was obtained from this tutorial: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return F.softmax(self.fc(embedded))


# In[ ]:


from torchtext.data.utils import ngrams_iterator


def generate_batch(batch):
    label = torch.tensor([int(entry.target[0]) for entry in batch])
    _text = []
    for entry in batch:
        _entry = []
        for t in entry.text:
            _entry.append(text.vocab.stoi[t])
        _text.append(torch.tensor(_entry,dtype=torch.long))
    offsets = [0] + [len(entry) for entry in _text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    _text = torch.cat(_text)
    return _text, offsets, label


from torch.utils.data import DataLoader


def train_func(sub_train_):
    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        output = model(text, offsets)
        loss = criterion(output, cls)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == cls).sum().item()

    # Adjust the learning rate
    scheduler.step()
    
    return train_loss / len(sub_train_), train_acc / len(sub_train_)

def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, offsets, cls in data:
        text, offsets, cls = text.to(device), offsets.to(device), cls.to(device)
        with torch.no_grad():
            output = model(text, offsets)
            loss = criterion(output, cls)
            loss += loss.item()
            acc += (output.argmax(1) == cls).sum().item()

    return loss / len(data_), acc / len(data_)


# In[ ]:


import time
from torch.utils.data.dataset import random_split

model = TextSentiment(VOCAB_SIZE, EMBED_DIM, 2).to(device)

N_EPOCHS = 5
min_valid_loss = float('inf')

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ =     random_split(train_dataset, [train_len, len(train_dataset) - train_len])

for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


# In[ ]:


def predict(_text, model, vocab, ngrams):
    if len(_text) == 0:
        return 0
    
    with torch.no_grad():
        _text = [vocab.stoi[token] for token in ngrams_iterator(_text, ngrams)]
        output = model(torch.tensor(_text), torch.tensor([0]))
        return output.argmax(1).item()

model = model.to("cpu")
predictions = [predict(entry.text, model, text.vocab, NGRAMS) for entry in test_dataset]
tweet_id = [entry.id[0] for entry in test_dataset]


# In[ ]:


output = pd.DataFrame({'id': tweet_id, 'target': predictions})
output.to_csv('my_submission.csv', index=False)

