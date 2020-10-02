#!/usr/bin/env python
# coding: utf-8

# # Import the libraries

# In[ ]:


import time
import os

import matplotlib.pyplot as plt
import numpy as np

from fastai import *
from fastai.vision import *
import torch
from torchvision import transforms
from sklearn.metrics import classification_report


# ## Kuzushiji-MNIST

# ### Prepare the data

# In[ ]:


X_train = np.load('../input/kmnist-train-imgs.npz')['arr_0']
X_test = np.load('../input/kmnist-test-imgs.npz')['arr_0']
y_train = np.load('../input/kmnist-train-labels.npz')['arr_0']
y_test = np.load('../input/kmnist-test-labels.npz')['arr_0']


# In[ ]:


X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)


# In[ ]:


X_train = np.repeat(X_train, 3, axis=1)
X_test = np.repeat(X_test, 3, axis=1)


# In[ ]:


mean = X_train.mean()
std = X_train.std()
X_train = (X_train-mean)/std
X_test = (X_test-mean)/std

X_train = torch.from_numpy(np.float32(X_train))
y_train = torch.from_numpy(y_train.astype(np.long))
X_test = torch.from_numpy(np.float32(X_test))
y_test = torch.from_numpy(y_test.astype(np.long))


# ### Create a custom dataset

# In[ ]:


class ArrayDataset(Dataset):
    "Sample numpy array dataset"
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.c = len(np.unique(y))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]


# In[ ]:


train_ds, valid_ds = ArrayDataset(X_train, y_train), ArrayDataset(X_test, y_test)
data = DataBunch.create(train_ds, valid_ds, bs=64)


# In[ ]:


learn = cnn_learner(data, models.resnet18, loss_func=CrossEntropyFlat(), metrics=accuracy)


# ### Find the optimal LR

# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# ### Train the model

# In[ ]:


learn.fit_one_cycle(3, 1e-2)


# ### Look at the results

# In[ ]:


char_df = pd.read_csv('../input/kmnist_classmap.csv', encoding = 'utf-8')


# In[ ]:


X,y = learn.get_preds()


# In[ ]:


print(f"Accuracy of {accuracy(X,y)}")


# In[ ]:


X = np.argmax(X,axis=1)


# In[ ]:


target_names = ["Class {} ({}):".format(i, char_df[char_df['index']==i]['char'].item()) for i in range(len(np.unique(y_test)))]
print(classification_report(y, X, target_names=target_names))


# ## Kuzushiji-49

# ### Prepare the data

# In[ ]:


X_train = np.load('../input/k49-train-imgs.npz')['arr_0']
X_test = np.load('../input/k49-test-imgs.npz')['arr_0']
y_train = np.load('../input/k49-train-labels.npz')['arr_0']
y_test = np.load('../input/k49-test-labels.npz')['arr_0']


# In[ ]:


X_train = X_train.reshape(-1, 1, 28, 28)
X_test = X_test.reshape(-1, 1, 28, 28)


# In[ ]:


X_train = np.repeat(X_train, 3, axis=1)
X_test = np.repeat(X_test, 3, axis=1)


# In[ ]:


mean = X_train.mean()
std = X_train.std()
X_train = (X_train-mean)/std
X_test = (X_test-mean)/std

# Numpy to Torch Tensor
X_train = torch.from_numpy(np.float32(X_train))
y_train = torch.from_numpy(y_train.astype(np.long))
X_test = torch.from_numpy(np.float32(X_test))
y_test = torch.from_numpy(y_test.astype(np.long))


# In[ ]:


train_ds, valid_ds = ArrayDataset(X_train, y_train), ArrayDataset(X_test, y_test)
data = DataBunch.create(train_ds, valid_ds, bs=64)


# In[ ]:


learn = cnn_learner(data, models.resnet18, loss_func=CrossEntropyFlat(), metrics=accuracy)


# ### Find the optimal LR

# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# ### Train the model

# In[ ]:


learn.fit_one_cycle(5, 1e-2)


# ### Look at the results

# In[ ]:


char_df = pd.read_csv('../input/k49_classmap.csv', encoding = 'utf-8')


# In[ ]:


X,y = learn.get_preds()


# In[ ]:


print(f"Accuracy of {accuracy(X,y)}")


# In[ ]:


X = np.argmax(X,axis=1)


# In[ ]:


target_names = ["Class {} ({}):".format(i, char_df[char_df['index']==i]['char'].item()) for i in range(len(np.unique(y_test)))]
print(classification_report(y, X, target_names=target_names))


# In[ ]:




