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


import csv
import pickle
from tqdm import tqdm_notebook as tqdm
from glob import glob


video2nframes = pickle.load(open('../input/youtube2019videolength/vid2nseg.pk','rb'))


# In[ ]:


df = pd.read_csv('../input/youtube2018bestsubs/yt2018_privLB0.86580_sub_jun24_4.csv')
sub = pd.read_csv('../input/youtube8m-2019/sample_submission.csv')
sub = sub.set_index(sub.Class).sort_index()
Labels = sub.Class.values
df.head(1)
df = df[df.VideoId.isin(set(video2nframes.keys()))]
Segments = df.VideoId.values


# In[ ]:


sub


# In[ ]:


NCLASS=4000
y_pred = np.zeros((len(Segments),NCLASS))
for i in tqdm(range(len(Segments))):
    pairs = df.iloc[i].LabelConfidencePairs.split(' ')
    labels = [int(x) for x in pairs[::2]]
    proba = [float(x) for x in pairs[1::2]]
    for j in range(len(labels)):
        y_pred[i,labels[j]] = proba[j]
        
sub.Segments = [' '.join(list(Segments[np.argsort(-y_pred[:,i])[:100000]])) for i in Labels]


# In[ ]:


sub


# In[ ]:


sub_ = []
for i in tqdm(sub.Class.values):
    segs = sub.loc[i].Segments.split(' ')
    S=[]
    for v in segs:
        nframes = video2nframes[v]
        S += [' '.join([v+':'+str(j) for j in range(30,(nframes//5)*5-30,15)])]
    sub_ += [ ' '.join(' '.join(S).split(' ')[:100000])]
    
sub.Segments = sub_
        


# In[ ]:


sub.to_csv('sub.csv',index=False)


# In[ ]:




