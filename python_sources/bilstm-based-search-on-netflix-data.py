#!/usr/bin/env python
# coding: utf-8

# # Netflix Dataset
# >and all those embedding funs

# * Create a pytorch dataset for text
# * A BiLSTM model to predict multiple genre
# * Encode the text to vectors using the model we trained
# * Search the closest description

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


get_ipython().system('pip install forgebox')


# In[ ]:


from pathlib import Path


# In[ ]:


DATA = Path("/kaggle/input/netflix-shows/netflix_titles.csv")

df = pd.read_csv(DATA)

df.sample(10)


# ### So... what Y?

# In[ ]:


df.listed_in.value_counts()


# In[ ]:


df.rating.value_counts()


# In[ ]:


df["listed_in"] = df.listed_in.str.replace("&",",").replace(" , ",",").replace(" ,",",").replace(", ",",").replace(" , ",",")


# In[ ]:


genre = list(set(i.strip() for i in (",".join(list(df.listed_in))).split(",")))


# In[ ]:


print(f"Total genre: {len(genre)}\n")
for g in genre:
    print(g,end="\t")


# In[ ]:


eye = np.eye(len(genre))
genre_dict = dict((v,eye[k]) for k,v in enumerate(genre))

def to_nhot(text):
    return np.sum(list(genre_dict[g.strip()] for g in text.split(",")),axis=0).astype(np.int)

df["genre"] = df.listed_in.apply(to_nhot)


# In[ ]:


PROCESSED = "processed.csv"


# In[ ]:


df.to_csv(PROCESSED,index = False)


# ### Process the text

# In[ ]:


from forgebox.ftorch.prepro import split_df


# In[ ]:


train_df,val_df = split_df(df,valid=0.1)
print(f"train:{len(train_df)}\tvalid:{len(val_df)}")


# In[ ]:


from nltk.tokenize import TweetTokenizer
tkz = TweetTokenizer()
def tokenize(txt):
    return tkz.tokenize(txt)


# In[ ]:


tokenize("A man returns home after being released from ")


# ### Generate vocabulary map from material

# In[ ]:


from itertools import chain
from multiprocessing import Pool
from collections import Counter
from torch.utils.data.dataset import Dataset

class Vocab(object):
    def __init__(self, iterative, max_vocab = 12000, tokenize = tkz.tokenize):
        """
        Count the most frequent words
        Make the word<=>index mapping
        """
        self.l = list(iterative)
        self.max_vocab = max_vocab
        self.tokenize = tokenize
        self.word_beads = self.word_beads_()
        self.counter()
        
    def __len__(self):
        return len(self.words)
        
    def __repr__(self):
        return f"vocab {self.max_vocab}"
        
    def word_beads_(self, nproc=10):
        self.p = Pool(nproc)
        return list(chain(*list(self.p.map(self.tokenize,self.l))))
    
    def counter(self):
        vals = np.array(list((k,v) for k,v in dict(Counter(self.word_beads)).items()))
        self.words = pd.DataFrame({"tok":vals[:,0],"ct":vals[:,1]}).sort_values(by= "ct",ascending=False)        .reset_index().drop("index",axis=1).head(self.max_vocab-2)
        self.words["idx"] = (np.arange(len(self.words))+2)
        self.words=pd.concat([self.words,pd.DataFrame({"tok":["<eos>","<mtk>"],"ct":[-1,-1],"idx":[0,1]})])
        return self.words
    
    def to_i(self):
        self.t2i = dict(zip(self.words["tok"],self.words["idx"]))
        def to_index(t):
            i = self.t2i.get(t)
            if i==None:
                return 1
            else:
                return i
        return to_index
    
    def to_t(self):
        return np.roll(self.words["tok"],2)
        
        


# In[ ]:


vocab = Vocab(df.description)


# ### Vocabulary build from training

# In[ ]:


vocab.words


# In[ ]:


class seqData(Dataset):
    def __init__(self,lines,vocab):
        self.lines = list(lines)
        self.vocab = vocab
        self.to_i = np.vectorize(vocab.to_i())
        self.to_t = vocab.to_t()
        self.bs=1
        
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self,idx):
        """
        Translate words to indices
        """
        line = self.lines[idx]
        words = self.vocab.tokenize(line)
        words = ["<eos>",]+words+["<eos>"]
        return self.to_i(np.array(words))
    
    def backward(self,seq):
        """
        This backward has nothing to do with gradrient
        Just to error proof the tokenized line
        """
        return " ".join(self.to_t[seq])
    
class arrData(Dataset):
    def __init__(self, *arrs):
        self.arr = np.concatenate(arrs,axis=1)
    
    def __len__(self):
        return self.arr.shape[0]
    
    def __getitem__(self,idx):
        return self.arr[idx]


# Build vocabulary and train dataset

# In[ ]:


vocab = Vocab(df.description)

train_seq = seqData(train_df.description,vocab)
train_y = arrData(np.stack(train_df.genre.values))

val_seq = seqData(val_df.description,vocab)
val_y = arrData(np.stack(val_df.genre.values))


# Size of train dataset

# In[ ]:


len(train_seq),len(train_y)


# In[ ]:


tokenized_line = train_seq[10]
tokenized_line


# Reconstruct the sentence from indices
# 
# >**<mtk\>** means the missing tokens, for they are less frequent than we should hav cared

# In[ ]:


train_seq.backward(tokenized_line)


# In[ ]:


import torch
from torch.utils.data.dataloader import DataLoader


# ### A custom made collate function
# 
# * Collate function will do the following:
# >Make rows of dataset output into a batch of tensor

# In[ ]:


def pad_collate(rows):
    """
    this collate will pad any sentence that is less then the max length
    """
    line_len = torch.LongTensor(list(len(row) for row in rows));
    max_len = line_len.max()
    ones = torch.ones(max_len.item()).long()
    line_pad = max_len-line_len
    return torch.stack(list(torch.cat([torch.LongTensor(row),ones[:pad.item()]]) for row,pad in zip(rows,line_pad[:,None])))


# In[ ]:


gen = iter(DataLoader(train_seq,batch_size=16, collate_fn=pad_collate))
next(gen).size()


# In[ ]:


class fuse(Dataset):
    def __init__(self, *datasets):
        """
        A pytorch dataset combining the dataset
        :param datasets:
        """
        self.datasets = datasets
        length_s = set(list(len(d) for d in self.datasets))
        assert len(length_s) == 1, "dataset lenth not matched"
        self.length = list(length_s)[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return tuple(d.__getitem__(idx) for d in self.datasets)


# 

# In[ ]:


def combine_collate(*funcs):
    def combined(rows):
        xs = list(zip(*rows))
        return tuple(func(x) for func, x in zip(funcs,xs))
    return combined

from torch.utils.data._utils.collate import default_collate


# In[ ]:


DataLoader(train_y).collate_fn


# ### Fusing data set

# In[ ]:


train_ds = fuse(train_seq,train_y)
val_ds = fuse(val_seq,val_y)


# ### Testing Generator

# In[ ]:


gen = iter(DataLoader(train_ds,batch_size=16, collate_fn=combine_collate(pad_collate,default_collate)))
x,y = next(gen)
print(x.shape,y.shape)


# ### Model

# In[ ]:


from torch import nn
import torch


# In[ ]:


class basicNLP(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.hs = hs
        self.emb = nn.Embedding(len(vocab),hs)
        self.rnn = nn.LSTM(input_size = hs,hidden_size = hs,batch_first = True)
        self.fc = nn.Sequential(*[
            nn.BatchNorm1d(hs*2),
            nn.ReLU(),
            nn.Linear(hs*2,hs*2),
            nn.BatchNorm1d(hs*2),
            nn.ReLU(),
            nn.Linear(hs*2,49),
        ])
        
    def encoder(self,x):
        x = self.emb(x)
        o1,(h1,c1) = self.rnn(x)
        # run sentence backward
        o2,(h2,c2) = self.rnn(x.flip(dims=[1]))
        return torch.cat([h1[0],h2[0]],dim=1)
        
    def forward(self,x):
        vec = self.encoder(x)
        return self.fc(vec)


# In[ ]:


model = basicNLP(100)


# In[ ]:


x[:2,:]


# ### What does embedding do?

# In[ ]:


x.shape,model.emb(x).shape


# ### What does LSTM return?

# For what is LSTM, read this [awesome blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/), from which I stole the following visualization from
# 
# #### In short version
# RNN, it's about sharing model weights throughout temporal sequence, as convolusion share weights in spatial point of view
# ![image.png](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png)
# 
# * The above green "A" area re shared linear layer
# * GRU & LSTM are advanced version of RNN, with gate control
# * The black arrows above in GRU & LSTM are controlled by gates
# * Gates, are just linear layer with sigmoid activation $\sigma(x)$, its outputs are between (0,1), hence the name gate, the following illustration is one of the gates in a lstm cell, called input gate
# ![gate controls](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png)
# * Other gates control other things like should we forget the early part of then sentence, should we output this .etc
# 
# 

# ### In terms of code

# In[ ]:


get_ipython().run_line_magic('time', '')
output,(hidden_state, cell_state) = model.rnn(model.emb(x))
for t in (output,hidden_state, cell_state):
    print(t.shape)


# Disect the iteration through the sentence

# In[ ]:


get_ipython().run_line_magic('time', '')
init_hidden = torch.zeros((1,16,100))
init_cell = torch.zeros((1,16,100))
last_h,last_c = init_hidden,init_cell
outputs = []
x_vec = model.emb(x)
for row in range(x.shape[1]):
    last_o, (last_h,last_c) = model.rnn(x_vec[:,row:row+1,:],(last_h,last_c))
    outputs.append(last_o)


# In[ ]:


manual_iteration_result = torch.cat(outputs,dim=1)


# In[ ]:


manual_iteration_result.shape


# The 2 results are the same, of course, I thought manual python iteration is slower,but they are really close by the above test

# In[ ]:


(manual_iteration_result==output).float().mean()


# ### Training

# In[ ]:


lossf = nn.BCEWithLogitsLoss()


# In[ ]:


from forgebox.ftorch.train import Trainer
from forgebox.ftorch.callbacks import stat
from forgebox.ftorch.metrics import metric4_bi


# In[ ]:


model = model.cuda()


# In[ ]:


t = Trainer(train_ds, val_dataset=val_ds,batch_size=16,callbacks=[stat], val_callbacks=[stat] ,shuffle=True,)


# In[ ]:


t.opt["adm1"] = torch.optim.Adam(model.parameters())


# Combined collate function

# In[ ]:


t.train_data.collate_fn = combine_collate(pad_collate,default_collate)
t.val_data.collate_fn = combine_collate(pad_collate,default_collate)


# In[ ]:


@t.step_train
def train_step(self):
    self.opt.zero_all()
    x,y = self.data
    y_= model(x)
    loss = lossf(y_,y.float())
    loss.backward()
    self.opt.step_all()
    acc,rec,prec,f1 = metric4_bi(torch.sigmoid(y_),y)
    return dict((k,v.item()) for k,v in zip(["loss","acc","rec","prec","f1"],(loss,acc,rec,prec,f1)))
                
@t.step_val
def val_step(self):
    x,y = self.data
    y_= model(x)
    loss = lossf(y_,y.float())
    acc,rec,prec,f1 = metric4_bi(torch.sigmoid(y_),y)
    return dict((k,v.item()) for k,v in zip(["loss","acc","rec","prec","f1"],(loss,acc,rec,prec,f1)))


# In[ ]:


t.train(10)


# ### Search similar

# In[ ]:


model = model.eval()
dl = DataLoader(train_seq, batch_size=32, collate_fn=pad_collate)


# In[ ]:


text_gen = iter(dl)
result = []
for i in range(len(dl)):
    x=next(text_gen)
    x = x.cuda()
    x_vec = model.encoder(x)
    result.append(x_vec.cpu())


# A vector representing each of the sentence

# In[ ]:


result_vec = torch.cat(result,dim=0).detach().numpy()
result_vec.shape


# In[ ]:


def to_idx(line):
    words = train_seq.vocab.tokenize(line)
    words = ["<eos>",]+words+["<eos>"]
    return train_seq.to_i(np.array(words))[None,:]


# In[ ]:


to_idx("this"), to_idx("to be or not to be")


# In[ ]:


def to_vec(line):
    vec = torch.LongTensor(to_idx(line)).cuda()
    return model.encoder(vec).cpu().detach().numpy()


# In[ ]:


to_vec("this"), to_vec("to be or not to be")


# In[ ]:


def l2norm(x):
    """
    L2 Norm
    """
    return np.linalg.norm(x,2,1).reshape(-1,1)


# In[ ]:


pd.set_option("max_colwidth",150)

def search(line):
    vec = to_vec(line)
    sim = ((vec* result_vec)/l2norm(result_vec)).sum(-1)
    return pd.DataFrame({"text":train_seq.lines,"sim":sim})        .sort_values(by="sim",ascending=False)


# In[ ]:


search("Experience our planet's natural beauty").head(10)


# In[ ]:


search("love story,marriage, girl").head(10)


# Well, usually it should be more accurate if we have more data
