#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import os, sys
import re
import string
import pathlib
import random
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from tqdm import tqdm, tqdm_notebook, tnrange
tqdm.pandas(desc='Progress')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torchtext
from torchtext import data
from torchtext import vocab

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[ ]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    text = re.sub(r'https?:/\/\S+', ' ', text)
    text = [lemmatizer.lemmatize(token) for token in text.split(' ')]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text.strip()
#for text in df.review:
#    print(clean_text(text))
#df.review = df.review.progress_apply(lambda x: clean_text(x))


# In[ ]:


get_ipython().run_cell_magic('time', '', "nlp = spacy.load('en')\ndef tokenizer(s): return [w.text.lower() for w in nlp(clean_text(s))]")


# In[ ]:


txt_field = data.Field(sequential=True, tokenize=tokenizer, include_lengths=True, use_vocab=True)
label_field = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)


# In[ ]:


train_val_fields = [
    ('review', txt_field),
    ('sentiment', label_field),
]


# In[ ]:


get_ipython().run_cell_magic('time', '', "trainds, valds, testds = data.TabularDataset.splits(path='../input/', format='csv',\n                                            train='traindf.csv', validation='valdf.csv', test='testdf.csv',\n                                            fields=train_val_fields, skip_header=True)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'txt_field.build_vocab(trainds, valds, testds, max_size=100000, vectors="glove.6B.100d")\nlabel_field.build_vocab(trainds,testds)')


# In[ ]:


txt_field.vocab.vectors[txt_field.vocab.stoi["the"]]


# In[ ]:


class BatchGenerator:
    def __init__(self, dl, x_field, y_field):
        self.dl, self.x_field, self.y_field = dl, x_field, y_field
        
    def __len__(self):
        return len(self.dl)
    
    def __iter__(self):
        for batch in self.dl:
            X = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (X,y)


# In[ ]:


vocab_size = len(txt_field.vocab)
embedding_dim = 100
n_hidden = 64
n_out = 2


# In[ ]:


class Network(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_hidden, n_out, pretrained_vec, bidirectional=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.bidirectional = bidirectional
        
        self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.emb.weight.data.copy_(pretrained_vec)
        self.emb.weight.requires_grad = False
        self.gru = nn.GRU(self.embedding_dim, self.n_hidden, bidirectional=bidirectional)
        if bidirectional:
            self.out = nn.Linear(self.n_hidden*2*2, self.n_out)
        else:
            self.out = nn.Linear(self.n_hidden*2, self.n_out)
        
    def forward(self, seq, lengths):
        bs = seq.size(1)
        self.h = self.init_hidden(bs)
        seq = seq.transpose(0,1)
        embs = self.emb(seq)
        embs = embs.transpose(0,1)
        embs = pack_padded_sequence(embs, lengths)
        gru_out, self.h = self.gru(embs, self.h)
        gru_out, lengths = pad_packed_sequence(gru_out)        
        
        avg_pool = F.adaptive_avg_pool1d(gru_out.permute(1,2,0),1).view(bs,-1)
        max_pool = F.adaptive_max_pool1d(gru_out.permute(1,2,0),1).view(bs,-1)        
        outp = self.out(torch.cat([avg_pool,max_pool],dim=1))
        return F.log_softmax(outp)
    
    def init_hidden(self, batch_size): 
        if self.bidirectional:
            return torch.zeros((2,batch_size,self.n_hidden)).to(device)
        else:
            return torch.zeros((1,batch_size,self.n_hidden)).cuda().to(device)


# In[ ]:


def fit(model, train_dl, val_dl, loss_fn, opt, epochs=3, tollerance=5):
    num_batch = len(train_dl)
    from_valacc = 0
    path_to_best_model = "../network.py"
    for epoch in tnrange(epochs):      
        y_true_train = list()
        y_pred_train = list()
        total_loss_train = 0          
        
        t = tqdm_notebook(iter(train_dl), leave=False, total=num_batch)
        for (X,lengths),y in t:
            t.set_description(f'Epoch {epoch}')
            lengths = lengths.cpu().numpy()
            
            opt.zero_grad()
            pred = model(X, lengths)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            
            t.set_postfix(loss=loss.item())
            pred_idx = torch.max(pred, dim=1)[1]
            
            y_true_train += list(y.cpu().data.numpy())
            y_pred_train += list(pred_idx.cpu().data.numpy())
            total_loss_train += loss.item()
            
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_loss = total_loss_train/len(train_dl)
        
        y_true_val = list()
        y_pred_val = list()
        total_loss_val = 0
        for (X,lengths),y in tqdm_notebook(val_dl, leave=False):
            pred = model(X, lengths.cpu().numpy())
            loss = loss_fn(pred, y)
            pred_idx = torch.max(pred, 1)[1]
            y_true_val += list(y.cpu().data.numpy())
            y_pred_val += list(pred_idx.cpu().data.numpy())
            total_loss_val += loss.item()
        valacc = accuracy_score(y_true_val, y_pred_val)
        valloss = total_loss_val/len(valdl)
        if from_valacc>valacc:
            tollerance -=1
        if from_valacc<valacc:
            from_valacc=valacc
            tollerance = 5
            torch.save(model.state_dict(),path_to_best_model)
        print(f'Epoch {epoch}: train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | val_loss: {valloss:.4f} val_acc: {valacc:.4f} | tollerance: {tollerance:}')
        if tollerance<=0:
            break
    model.load_state_dict(torch.load(path_to_best_model))
    print("Training stopped")


# In[ ]:


traindl, valdl = data.BucketIterator.splits(datasets=(trainds, valds), batch_sizes=(512,512), sort_key=lambda x: len(x.review), device=device, sort_within_batch=True, repeat=False)
train_batch_it = BatchGenerator(traindl, 'review', 'sentiment')
val_batch_it = BatchGenerator(valdl, 'review', 'sentiment')
testdl = data.BucketIterator(dataset=testds, batch_size=512, sort_key=lambda x: len(x.review), device=device, sort_within_batch=True, repeat=False)
test_batch_it = BatchGenerator(testdl, 'review', 'sentiment')


# In[ ]:


m = Network(vocab_size, embedding_dim, n_hidden, n_out, trainds.fields['review'].vocab.vectors).to(device)
opt = optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), 1e-3)

fit(model=m, train_dl=train_batch_it, val_dl=val_batch_it, loss_fn=F.nll_loss, opt=opt, epochs=50)


# In[ ]:


from sklearn import metrics
y_true_test = list()
y_pred_test = list()
for (X,lengths),y in tqdm_notebook(test_batch_it, leave=False):
    pred = m(X, lengths.cpu().numpy())
    loss = F.nll_loss(pred, y)
    pred_idx = torch.max(pred, 1)[1]
    y_true_test += list(y.cpu().data.numpy())
    y_pred_test += list(pred_idx.cpu().data.numpy())
#testacc = accuracy_score(y_true_val, y_pred_val)
print(metrics.classification_report(y_pred_test,y_true_test))

