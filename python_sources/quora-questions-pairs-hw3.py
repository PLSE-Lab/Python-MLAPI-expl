#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
#import nltk
#from nltk.corpus import stopwords
#from nltk.stem import SnowballStemmer
import re
from string import punctuation


import numpy as np 
import pandas as pd 
import os
import spacy
import string
import re
import numpy as np
from spacy.symbols import ORTH
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence 


# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


# Add the string 'empty' to empty strings
train = train.fillna('empty')
test = test.fillna('empty')


# In[ ]:


# Preview some of the pairs of questions

for i in range(10):
    print(train.question1[i])
    print(train.question2[i])


# In[ ]:


get_ipython().system('python -m spacy download en')


# ### Data pre-processing
# 

# In[ ]:



re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
def sub_br(x): return re_br.sub("\n", x)

#nlp = spacy.load("en")
nlp = spacy.load('en_core_web_sm')
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

def clean_text(text):
    ''' Pre process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    #text = text.split()

    return text

my_tok = spacy.load('en')
def spacy_tok(x): return [tok.text for tok in my_tok.tokenizer(clean_text(x))]

def remove_stop_words(tokens): return [tok for tok in tokens if tok not in spacy_stopwords]


# In[ ]:


sent = "Motorola (company): Can I hack my Charter Motorolla DCX3400?"


# In[ ]:


clean_sent = clean_text(sent)


# In[ ]:


tokens = spacy_tok(clean_sent)
tokens


# In[ ]:


remove_stop_words(tokens)


# ### Building a vocabualry
# 
# To do so, we will first concatenate train and test and get the vocabulary from both sets

# In[ ]:


combined = pd.concat([train,test])


# In[ ]:


print(train.shape)
print(test.shape)
print(combined.shape)


# In[ ]:


combined.head()


# In[ ]:


counts = Counter()
questions = ['question1', 'question2']
for question in questions:
    for sent in train[question]:
        try:
            counts.update(remove_stop_words(spacy_tok(sent)))
        except:
            pass


# In[ ]:


'''counts = Counter()
questions = ['question1', 'question2']
for question in questions:
    for sent in train[question][:10]:
        counts.update(spacy_tok(sent))'''


# In[ ]:


counts


# In[ ]:


len(counts.keys())


# In[ ]:


# Vocabulary
vocab2index = {"":0, "UNK":1}
words = ["", "UNK"]
for word in counts:
    vocab2index[word] = len(words)
    words.append(word)


# In[ ]:


len(words)


# In[ ]:


# WHat is the 99% quantile of  length of the sentence?

combined['len_q1'] = combined['question1'].apply(lambda x: len(x.split()))
combined['len_q2'] = combined['question2'].apply(lambda x: len(x.split()))


# In[ ]:


combined['len_q1'].quantile(q = 0.99)


# In[ ]:


combined['len_q2'].quantile(q = 0.99)


# In[ ]:


# note that spacy_tok takes a while run it just once
def encode_sentence(sent, vocab2index, N=30, padding_start=True):
    
    x = remove_stop_words(spacy_tok(sent))
    enc = np.zeros(N, dtype=np.int32)
    enc1 = np.array([vocab2index.get(w, vocab2index["UNK"]) for w in x])
    l = min(N, len(enc1))
    if padding_start:
        enc[:l] = enc1[:l]
    else:
        enc[N-l:] = enc1[:l]
    return enc


# In[ ]:


# Encoding questions in the train dataset
train['enc_question1'] = train['question1'].apply(lambda x: encode_sentence(x,vocab2index, N=40, padding_start=True) )
train['enc_question2'] = train['question2'].apply(lambda x: encode_sentence(x,vocab2index, N=40, padding_start=True) )


# In[ ]:


# Encoding questions in the test datset
test['enc_question1'] = test['question1'].apply(lambda x: encode_sentence(x,vocab2index, N=40, padding_start=True) )
test['enc_question2'] = test['question2'].apply(lambda x: encode_sentence(x,vocab2index, N=40, padding_start=True) )


# ###  Splitting into train and validation

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


VALID_IDX = 40000
N = len(train)
X_train = train[['enc_question1','enc_question2']][:(N-VALID_IDX)]
y_train= train['is_duplicate'][:(N-VALID_IDX)].values
X_valid = train[['enc_question1','enc_question2']][(N-VALID_IDX):]
y_valid= train['is_duplicate'][(N-VALID_IDX):].values


# In[ ]:


X_train.reset_index(inplace=True, drop=True)
X_valid.reset_index(inplace=True, drop=True)


# In[ ]:


len(y_train)


# In[ ]:


len(X_train)


# In[ ]:


len(X_valid)


# In[ ]:


len(y_valid)


# In[ ]:


X_valid.reset_index(inplace=True, drop=True)
X_valid.head()


# In[ ]:


class QuoraDataset(Dataset):
    def __init__(self, df,y, N=40, padding_start=True):
        self.df = df
        self.y = y
        self.x1 = df['enc_question1']
        self.x2 = df['enc_question2']
        # pos 1, neg 0
    
       # self.x1 = [encode_sentence(sent, vocab2index, N, padding_start) for sent in self.q1]
       # self.x2 = [encode_sentence(sent, vocab2index, N, padding_start) for sent in self.q2]
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x1 = self.x1[idx]
        x2 = self.x2[idx]
        return x1, x2, self.y[idx]


# In[ ]:


train_ds_v0 = QuoraDataset(X_train,y_train, padding_start=False)
valid_ds_v0 = QuoraDataset(X_valid,y_valid, padding_start=False)


# In[ ]:


len(y_valid)


# In[ ]:


train_ds_v0[364289]


# In[ ]:


valid_ds_v0[15468]


# In[ ]:


y_valid[0]


# In[ ]:





# In[ ]:


batch_size = 32
train_dl_v0 = DataLoader(train_ds_v0, batch_size=batch_size, shuffle=True)
valid_dl_v0 = DataLoader(valid_ds_v0, batch_size=batch_size)


# In[ ]:


x1, x2,y = next(iter(train_dl_v0))


# ### MODEL V0
# simple model: turn question into sequence of GloVe embeddings, pass them through LSTM embedding layer to get question representation as vector (I used 100-dimensional vectors), merge two vectors into one and pull it through 2 fully connected layers and softmax. No manual feature engineering and model architecture thinking.

# In[ ]:


class LSTMV0Model(torch.nn.Module) :
    def __init__(self, vocab_size, embedding_dim, hidden_dim) :
        super(LSTMV0Model,self).__init__()
        self.hidden_dim = hidden_dim
        # Layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x1, x2, batch_size):
        x1 = self.embeddings(x1)
        x1 = self.dropout(x1)
        
        x2 = self.embeddings(x2)
        x2 = self.dropout(x2)
        self.batch_size = batch_size
        
        out_pack1, (ht1, ct1) = self.lstm(x1)
        out_pack2, (ht2, ct2) = self.lstm(x2)
        
        # Distance
        if self.batch_size == 1:
            prediction = torch.exp(-torch.norm((ht1[-1].squeeze() - ht2[-1].squeeze()), 1))
        else:
            prediction = torch.exp(-torch.norm((ht1[-1].squeeze() - ht2[-1].squeeze()), 1, 1))
        #print(prediction.unsqueeze(-1).size())
        return prediction.unsqueeze(-1)


# In[ ]:


def train_epocs_v0(model, batch_size, epochs=10, lr=0.001):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x1, x2, y in train_dl:
            # s is not used in this model
            x1 = x1.long().cuda()
            x2 = x2.long().cuda()
            y = y.float().cuda()
            y_pred = model(x1, x2, batch_size)
            optimizer.zero_grad()
            loss = F.binary_cross_entropy(y_pred, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
        val_loss, val_acc = val_metrics_v0(model, val_dl)
        if i % 5 == 1:
            print("train loss %.3f val loss %.3f and val accuracy %.3f" % (sum_loss/total, val_loss, val_acc))


# In[ ]:


def val_metrics_v0(model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    for x1, x2, y in valid_dl:
        # s is not used here
        x1 = x1.long().cuda()
        x2 = x2.long().cuda()
        y = y.float().cuda().unsqueeze(1)
        y_hat = model(x1,x2, batch_size)
        loss = F.binary_cross_entropy(y_hat, y)
        y_pred = y_hat > 0.5
        correct += (y_pred.float() == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
    return sum_loss/total, correct/total


# In[ ]:


batch_size = 5000
train_dl = DataLoader(train_ds_v0, batch_size=batch_size, shuffle=True)
val_dl = DataLoader(valid_ds_v0, batch_size=batch_size)


# In[ ]:


vocab_size = len(words)
print(vocab_size)
model_v0 = LSTMV0Model(vocab_size, 50, 50).cuda()


# In[ ]:


train_epocs_v0(model_v0,batch_size = batch_size, epochs=50, lr=0.001)


# In[ ]:


test_dl = 


# In[ ]:


### Make prediction


# In[ ]:


subm = pd.read_csv("../input/sample_submission.csv")


# In[ ]:


#test = pd.read_csv('../input/test.csv')


# In[ ]:


sample_sub = pd.read_csv('../input/sample_submission.csv')
test_ds = QuoraDataset(test, sample_sub.is_duplicate)
batch_size = 5000
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
y1 = []
for x1, x2, y in test_dl:
    x1 = x1.long().cuda()
    x2 = x2.long().cuda()
    y = y.float().cuda().unsqueeze(1)
    y_hat = model_v0(x1,x2, batch_size)
    y1.append([0 if x<=0.5 else 1 for x in y_hat])
y_pred = [yi for sublist in y1 for yi in sublist]
sample_sub.is_duplicate = y_pred
sample_sub.to_csv("submit.csv", index=False)


# In[ ]:




