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
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv("/kaggle/input/hse-pml-2/train_resort.csv")
submission = pd.read_csv("/kaggle/input/hse-pml-2/resort_sample.csv")


# In[ ]:


train['amount_spent_per_room_night_scaled'].mean()


# Competition's metric is MAE, so if we submit 0 predictions we can get target mean of public part.
# 
# $MAE = \frac{\sum_{i=0}^{N}|y_{i} - pred_{i}|}{n}$
# 
# $\frac{\sum_{i=0}^{N}|y_{i} - 0|}{n} = \frac{\sum_{i=0}^{N}|y_{i}|}{n}$

# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




