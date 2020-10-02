#!/usr/bin/env python
# coding: utf-8

# Use @Dieter 's post process to test  whether my submission can increase the lb score.
# 

# In[ ]:


'''
version 1: only change es language values, then my lb score jump from 0.950  to 0.9503
version 2: only change es and fr language values, then my lb score jump from 0.950  to 0.9504
version 3: change  5 language values, then my lb score jump from 0.950  to 0.9509

'''


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
os.listdir('../input/jigsaw-multilingual-toxic-comment-classification')

# Any results you write to the current directory are saved as output.


# In[ ]:


#0.9476
test0 = pd.read_csv('/kaggle/input/fastsub-test/submission.csv')
test0.head()


# In[ ]:


#lb 0.9494
test1 = pd.read_csv('/kaggle/input/kernel0531v4/submission.csv')
test1.head()


# In[ ]:


#lb 0.9478
test2 = pd.read_csv('/kaggle/input/jmtfstsub0407/submission.csv')
test2.head()


# In[ ]:


#This result should be lb 0.950
test3 = test1.copy()
test3['toxic'] = test1['toxic']*0.55 + 0.45*(test0['toxic']*0.5+test2['toxic']*0.5)


# # do the post process

# In[ ]:


test=pd.read_csv('../input/jigsaw-multilingual-toxic-comment-classification/test.csv')
test3=pd.merge(test3,test,on='id')
test3.head()


# ## We  change 5 language values

# In[ ]:


test3.loc[test3["lang"] == "es", "toxic"] *= 1.06
test3.loc[test3["lang"] == "fr", "toxic"] *= 1.04
test3.loc[test3["lang"] == "it", "toxic"] *= 0.97
test3.loc[test3["lang"] == "pt", "toxic"] *= 0.96
test3.loc[test3["lang"] == "tr", "toxic"] *= 0.98


# In[ ]:


test3.head()


# In[ ]:


test3[['id','toxic']].to_csv('submission.csv', index=False)


# In[ ]:


test3['l']=test1['toxic']
test3.corr()


# In[ ]:




