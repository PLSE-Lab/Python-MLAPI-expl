#!/usr/bin/env python
# coding: utf-8

# The [winning solution of Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80568#latest-516532) used a clever way to pad sequences per batch on the fly. <br> I would like to share the same for PyTorch.<br> This should help in improving run times without affecting model performance ( in theory ).

# In[1]:


import torch
from torch.utils import data
import numpy as np
from keras.preprocessing import sequence


# In[2]:


class TextDataset(data.Dataset):
    '''
    Simple Dataset
    '''
    def __init__(self,X,y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return [self.X[idx],self.y[idx]]
        return self.X[idx]


# In[3]:


class MyCollator(object):
    '''
    Yields a batch from a list of Items
    Args:
    test : Set True when using with test data loader. Defaults to False
    percentile : Trim sequences by this percentile
    '''
    def __init__(self,test=False,percentile=100):
        self.test = test
        self.percentile = percentile
    def __call__(self, batch):
        if not self.test:
            data = [item[0] for item in batch]
            target = [item[1] for item in batch]
        else:
            data = batch
        lens = [len(x) for x in data]
        max_len = np.percentile(lens,self.percentile)
        data = sequence.pad_sequences(data,maxlen=int(max_len))
        data = torch.tensor(data,dtype=torch.long)
        if not self.test:
            target = torch.tensor(target,dtype=torch.float32)
            return [data,target]
        return [data]


# Let's create a sample dataset to test our new collate function

# In[4]:


sample_size = 1024
sizes = np.random.normal(loc=200,scale=50,size=(sample_size,)).astype(np.int32)
X = [np.ones((sizes[i])) for i in range(sample_size)]
Y = np.random.rand(sample_size).round()


# If we choose to pad this data by maximum length in the whole data, this is the length all sequences will be padded to

# In[5]:


sizes.max()


# However, this is not ideal. <br>
# Let's try padding the sequence to maximum length per batch instead of the whole dataset

# In[6]:


batch_size = 128
dataset = TextDataset(X,Y)
test_dataset = TextDataset(X)


# **Sequence sizes are smaller overall! <br>**
# *Note that the size reduction depends on the distribution of sequence sizes in the actual dataset. <br>*
# *Here, I've used a normally distributed dummy dataset. *

# In[7]:


collate = MyCollator(percentile=100)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True ,collate_fn=collate)
for X,Y in loader:
    print(X.shape,Y.shape)


# Example : Running on test set

# In[8]:


test_collate = MyCollator(test=True,percentile=100)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False , collate_fn=test_collate)
for X in test_loader:
    print(X[0].shape)


# To Furthur reduce running times, you can choose to pad by **Nth** percentile of lenghts, keeping **N** close to 100. This may or may not affect model performance, your mileage may vary. <br>
# For example, **N = 95** gave a good balance between speed and performance for quora challenge's winning team. <br>

# In[9]:


collate = MyCollator(percentile=95)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True ,collate_fn=collate)
for X,Y in loader:
    print(X.shape,Y.shape)

