#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
GPU_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")

from fastai.vision import *
from fastai.train import Learner
from fastai.callbacks import SaveModelCallback
from fastai.metrics import accuracy as fastai_accuracy
from fastai.callbacks import SaveModelCallback
import torch.nn.functional as F
import torch

import pandas as pd
import numpy as np
import time


# ### Customize imagelist

# In[ ]:


class MyImageList(ImageList):
    @classmethod
    def from_df(cls, df:DataFrame, cols:IntsOrStrs=0, **kwargs)->'ItemList':        
        res = super().from_df(df, path='./', cols=cols, **kwargs)  
        if 'label' in df.columns:
            res.items = df.drop('label',axis=1).values
        else:
            res.items = df.values
        res.c,res.sizes = 1,{}
        return res
        
    def get(self, i):
        res = torch.tensor(self.items[i].reshape([28,28])).float().unsqueeze(0)
        self.sizes[i] = res.size
        return Image(res)   


# ### Create a Path instance

# In[ ]:


path = Path('../input/digit-recognizer')
path.ls()


# ### Create an ImageList instance

# In[ ]:


train = pd.read_csv(path/'train.csv')
test = pd.read_csv(path/'test.csv')


# In[ ]:


il = MyImageList.from_df(train)
il


# In[ ]:


il[0].show(cmap='gray')


# ### Create item lists for train and valid

# In[ ]:


sd = il.split_by_rand_pct(0.2)
sd


# ### Create a label list

# In[ ]:


ll = sd.label_from_df(cols='label')
ll


# ### Apply transformations

# In[ ]:


tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])


# In[ ]:


ll = ll.transform(tfms)


# ### Create a databunch instance

# In[ ]:


get_ipython().run_cell_magic('time', '', 'bs = 128\ndata = ll.databunch(bs=bs).normalize()\ndata.add_test(MyImageList.from_df(test))')


# ### Show random transformations of the same image

# In[ ]:


def _plot(i,j,ax): data.train_ds[0][0].show(ax,cmap='gray')
plot_multi(_plot, 3, 3, figsize=(8,8))


# ### show a batch of images with labels

# In[ ]:


xb,yb = data.one_batch()
print(xb.shape,yb.shape)
data.show_batch(rows=3, figsize=(10,8), cmap='gray')


# ### Create a CNN learner

# In[ ]:


class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf,nf)
        self.conv2 = conv_layer(nf,nf)
        
    def forward(self, x): return x + self.conv2(self.conv1(x))
    
def conv2(ni,nf): return conv_layer(ni,nf,stride=2)    
def conv_and_res(ni,nf): return nn.Sequential(conv2(ni, nf), res_block(nf))


# In[ ]:


model = torch.nn.Sequential(
    conv_and_res(1, 8),
    conv_and_res(8, 16),
    conv_and_res(16, 32),
    conv_and_res(32, 16),
    conv2(16, 10),
    Flatten()
)


# In[ ]:


get_ipython().run_cell_magic('time', '', "learn = Learner(data, model, loss_func = nn.CrossEntropyLoss(), metrics=accuracy)\nlearn.model_dir = '/kaggle/working/models'")


# ### find a proper learning rate

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# ### training

# In[ ]:


learn.fit_one_cycle(10,max_lr=slice(0.05),callbacks=[
            SaveModelCallback(learn, every='improvement', monitor='accuracy'),
            ])


# ### Predict and evaluate

# In[ ]:


get_ipython().run_cell_magic('time', '', 'yp,yr = learn.get_preds()\nyp = yp.numpy()\nyr = yr.numpy()')


# In[ ]:


def cross_entropy(y,yp):
    # y is the ground truch
    # yp is the prediction
    yp[yp>0.99999] = 0.99999
    yp[yp<1e-5] = 1e-5
    return np.mean(-np.log(yp[range(yp.shape[0]),y.astype(int)]))

def accuracy(y,yp):
    return (y==np.argmax(yp,axis=1)).mean()

def softmax(score):
    score = np.asarray(score, dtype=float)
    score = np.exp(score-np.max(score))
    score = score/(np.sum(score, axis=1).reshape([score.shape[0],1]))#[:,np.newaxis]
    return score


# In[ ]:


get_ipython().run_cell_magic('time', '', "acc = accuracy(yr,yp)\nce = cross_entropy(yr,yp)\nprint('Valid ACC: %.4f Cross Entropy:%4f'%(acc,ce))")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'yps,_ = learn.get_preds(DatasetType.Test)\nyps = yps.numpy()')


# In[ ]:


sub = pd.DataFrame()
sub['ImageId'] = np.arange(yps.shape[0])+1
sub['Label'] = np.argmax(yps,axis=1)
sub.head()


# In[ ]:


from datetime import datetime
clock = "{}".format(datetime.now()).replace(' ','-').replace(':','-').split('.')[0]
out = 'fastai_%s_acc_%.4f_ce_%.4f.csv'%(clock,acc,ce)
print(out)
sub.to_csv(out,index=False)

