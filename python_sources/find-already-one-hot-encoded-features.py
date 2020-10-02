#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import itertools
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')


# Get all binary columns.

# In[ ]:


bins = [col for col in train.columns if 'bin' in col]
print (bins)


# Let's see if we can find a combination of 2/3/4 binary columns where the rowwise sum is always 0/1.

# In[ ]:


# checking combination of 4 columns
for bin1, bin2, bin3, bin4 in list(itertools.combinations(bins, 4)):
        df = train[[bin1, bin2, bin3, bin4]]
        if len(df.sum(axis=1).unique())<=2:
            print (bin1, bin2, bin3, bin4, df.sum(axis=1).unique())           


# Well, it seems *ps_ind_06_bin, ps_ind_07_bin, ps_ind_08_bin, ps_ind_09_bin* are already one hot encoded features of a categoricat feature.
# 

# In[ ]:


# checking combination of 3 columns
for bin1, bin2, bin3 in list(itertools.combinations(bins, 3)):
        df = train[[bin1, bin2, bin3]]
        if len(df.sum(axis=1).unique())<=2:
            print (bin1, bin2, bin3, df.sum(axis=1).unique())           


# ps_ind_06_bin, ps_ind_07_bin, ps_ind_08_bin and ps_ind_09_bin are already available in previous 4 columns combination. They are part of another feature. For 3 columns we can say *ps_ind_16_bin, ps_ind_17_bin, ps_ind_18_bin * are OHE feature of a categorical value. for 0 sum, we can interpret that feature as missing.

# In[ ]:


for bin1, bin2 in list(itertools.combinations(bins, 2)):
        df = train[[bin1, bin2]]
        if len(df.sum(axis=1).unique())<=2:
            print (bin1, bin2, df.sum(axis=1).unique())   


# All these 2 columns combination are already available in 3/4 columns combination. These should not be considered as OHE of any categorical feature.

# 

# In[ ]:




