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
    #for filename in filenames:
        #print(os.path.join(dirname, filename))
    None

# Any results you write to the current directory are saved as output.


# **Here is the some route ahead after you are stuck between 0.04 and 0.05. I am not going to publish the kernel as it would be against the ethics of this competition at this stage.**
# * 1. I used Hicops Kernel only ,same Dataset ,image scaling ,did dataset cleaning using approach in point 7 
# 
# * 2. I used below version of Focal Loss
# * 3. I used effnetb2 in hicops model , You have to   replace 1280+1024 with 1408+1024
# * 4. I used higher image size 512,2048
# * 5. I used One Cycle LR scheduler max lr =1e-3 ,div_factor=8  and monitored mAP for saving best weights.This will help you prevernt overfitting.
# * 6. Stop the Training if you see Mask loss is rising more.
# * 7. I modified the Drop out rate for effnet as per https://www.kaggle.com/isakev/rb-s-centernet-baseline-pytorch-without-dropout

# **Dont Forget to Upvote if you get benefitted by the approach**
# I will keep updating if i find more approaches.

# In[ ]:


def _sigmoid(x):
    
    
    y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
    return y

class focal_loss(nn.Module):
  def __init__(self, gamma=2.0):
        super().__init__()
  def forward(self,pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pred=_sigmoid(pred)
    pos_inds = gt.eq(1).float()
    pos_inds=pos_inds.unsqueeze(1)
    #print(pos_inds.size())
    neg_inds = gt.lt(1).float().unsqueeze(1)

    neg_weights = torch.pow(1 - gt, 4).unsqueeze(1)

    loss = 0
    #print(neg_weights)
    pos_loss = torch.log(pred+1e-7) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred+1e-7) * torch.pow(pred, 2) * neg_weights * neg_inds

    
    #.float().sum()
    pos_loss = pos_loss.view(pred.size(0),-1).sum(-1)
    neg_loss = neg_loss.view(gt.size(0),-1).sum(-1)
    #neg_loss.sum(-1)
    num_pos  = pos_inds.sum()
    if num_pos == 0:
      loss = loss - neg_loss
    else:
      loss = loss - (pos_loss + neg_loss) #/ num_pos
    num_pos  = pos_inds.view(gt.size(0),-1).sum(-1)
    #print('loss',loss.size(),pos_loss.size(),loss.size(),'loss_sum',loss.sum(-1).mean(0),num_pos.size())
    return loss.mean(0)

