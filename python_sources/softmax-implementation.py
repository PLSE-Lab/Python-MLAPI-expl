#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#import matplotlib.plot

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[28]:


k = [1,2,7,0,0.01,0,0]

#map(np.exp, k)#/sum(map(np.exp, k))
exp = np.exp

#Creating anon func
def softArgMax(k : Vector) -> :
    '''
    Gives back the softmax values when supplied a vector.
    '''
    func = lambda x: np.exp(x)
    exps = func(k)
    softMaxOutputArray = exps/sum(exps)
    argMax = np.argmax(exps/sum(exps))
    return [softMaxOutputArray, argMax]

softArgMax(k)


# In[ ]:




