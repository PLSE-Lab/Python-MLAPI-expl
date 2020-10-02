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
from fastai.vision import *
from sklearn.model_selection import StratifiedShuffleSplit
# Any results you write to the current directory are saved as output.
import warnings
warnings.simplefilter("ignore")
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction =False):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction  = reduction 
    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce =None)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce =None)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction :
            return torch.mean(F_loss)
        else:
            return F_loss


# In[ ]:


from pathlib import Path
path=Path('../input')
df_trn=pd.read_csv(path/'X_train.csv')
df_label=pd.read_csv(path/'y_train.csv')
df_test=pd.read_csv(path/'X_test.csv')


# In[ ]:


df_all=pd.concat([df_trn,df_test])
df_all['train']=['train']*len(df_trn)+['test']*len(df_test)


# In[ ]:


df_all.columns


# In[ ]:


import seaborn as sns


# In[ ]:


sns.pairplot(df_all.sample(frac=0.05),hue='train',vars=['orientation_X','orientation_Y','orientation_Z','orientation_W'])


# In[ ]:


sns.pairplot(df_all.sample(frac=0.05),hue='train',vars=['angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z'])


# In[ ]:


sns.pairplot(df_all.sample(frac=0.05),hue='train',vars=['linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z'])


# In[ ]:


cols=['linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z']
for col in cols:

    df_trn[col]=(df_trn[col])/(85)
    df_test[col]=(df_test[col])/(85)
cols=['orientation_X','orientation_Y','orientation_Z','orientation_W','angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z','linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z']


# In[ ]:


test_list=df_test.groupby('series_id')
train_list=df_trn.groupby('series_id')


# In[ ]:


def open_image(self,i):
    mn=(np.hstack([self.items[i][1]['measurement_number'][:,None] for j in range(8)])-64)/256
    #mn=np.zeros_like(mn)
    feats=np.hstack([self.items[i][1][cols] for j in range(6)])
    feats_mean=(feats-feats.mean())/feats.std()
    img= np.append(np.append(feats,mn,axis=1),feats_mean,axis=1)
    img=np.stack([img,img.T,img[::-1,::-1]],axis=2)
    return pil2tensor(img,np.float32)


# In[ ]:


ImageList.get=open_image


# In[ ]:


src=(ImageList(df_trn.groupby('series_id'),inner_df=df_label).split_by_rand_pct(0.2).label_from_df(cols='surface'))
src.add_test(test_list,label='concrete');
data=src.databunch(bs=32)
stats=data.batch_stats()
data.normalize(stats);


# In[ ]:


data.show_batch()


# In[ ]:


data.show_batch(ds_type=DatasetType.Test)


# In[ ]:


X=list(df_trn.groupby('series_id').indices.keys())
label,group=df_label.set_index('series_id').loc[df_trn.groupby('series_id').indices.keys(),'surface'],df_label.set_index('series_id').loc[df_trn.groupby('series_id').indices.keys(),'group_id']


# In[ ]:


sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
sss.get_n_splits(X, label,group)
idx_train,idx_val = next(sss.split(X, label,group)
arch=models.resnet50


# In[ ]:


idx_train,idx_val = next(sss.split(X, label,group))
src=(ImageList(train_list,inner_df=df_label).split_by_idxs(idx_train,idx_val).label_from_df(cols='surface'))
src.add_test(test_list);
data=src.databunch(bs=32)
stats=data.batch_stats()
data.normalize(stats)
learn=cnn_learner(data,base_arch=arch,pretrained=False)#,loss_func=FocalLoss(logits=True,gamma=1))
learn.lr_find(num_it=200)
learn.recorder.plot()


# In[ ]:


df_sub=pd.read_csv(path/'sample_submission.csv')


# In[ ]:


src_list=ImageList(df_trn.groupby('series_id'),inner_df=df_label)
#for i,(idx_train,idx_val) in enumerate(sss.split(np.unique(df_trn.series_id), df_label.surface)):
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
sss.get_n_splits(X, label,group)


# In[ ]:


def accuracy_mult(input:Tensor, targs:Tensor)->Rank0Tensor:
    "Compute accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targs = targs.argmax(dim=-1).view(n,-1).long()
    return (input==targs).float().mean()


# In[ ]:


target_probs=[]


# In[ ]:


for i,(idx_train,idx_val) in enumerate(sss.split(X, label,group)):
    src=(src_list.split_by_idxs(idx_train,idx_val).label_from_df(cols='surface'))
    src.add_test(test_list);
    data=src.databunch(bs=32)
    stats=data.batch_stats()
    data.normalize(stats)
    learn=cnn_learner(data,base_arch=arch,pretrained=False,metrics=[accuracy])#,loss_func=FocalLoss(logits=True,gamma=1))
    learn.fit_one_cycle(15,max_lr=slice(5e-4,5e-3))
    learn.recorder.plot()
    learn.recorder.plot_losses()
    x,y=learn.get_preds(ds_type=DatasetType.Test)
    target_probs.append(x)
    


# In[ ]:


preds,y,losses = learn.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn, preds, y, losses)


# In[ ]:


interp.plot_confusion_matrix(figsize=(16,9))


# In[ ]:


np.unique(df_label.surface,return_counts=True)


# In[ ]:


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm
target_probs_stacked=np.stack(target_probs,axis=2)
target_probs_stacked


# # Mean

# In[ ]:


x=(target_probs_stacked).mean(axis=2)
df_sub['mean']=[learn.data.classes[idx] for idx in np.argmax(x,axis=1)]


# In[ ]:


x=np.median((target_probs_stacked),axis=2)
df_sub['median']=[learn.data.classes[idx] for idx in np.argmax(x,axis=1)]


# In[ ]:


x=(target_probs_stacked).max(axis=2)
df_sub['max']=[learn.data.classes[idx] for idx in np.argmax(x,axis=1)]


# In[ ]:


x=target_probs_stacked[(target_probs_stacked<target_probs_stacked.max(axis=2)[:,:,None])&(target_probs_stacked>target_probs_stacked.min(axis=2)[:,:,None])].reshape(list(target_probs_stacked.shape[0:2])+[3])


# In[ ]:


x=x.mean(axis=2)


# In[ ]:


df_sub['truncated']=[learn.data.classes[idx] for idx in np.argmax(x,axis=1)]


# In[ ]:


df_sub=df_sub.drop(columns='surface')


# In[ ]:


df_sub=df_sub.set_index('series_id')


# In[ ]:


df_sub['mean'].to_csv('mean.csv',header=['surface'])
df_sub['median'].to_csv('median.csv',header=['surface'])
df_sub['max'].to_csv('max.csv',header=['surface'])
df_sub['truncated'].to_csv('truncated.csv',header=['surface'])


# In[ ]:


for i in range(5):
    df_sub[f'sub_{i}']=[learn.data.classes[idx] for idx in np.argmax(target_probs_stacked[:,:,i],axis=1)]
    df_sub[f'sub_{i}'].to_csv(f'sub{i}.csv',header=['surface'])


# In[ ]:




