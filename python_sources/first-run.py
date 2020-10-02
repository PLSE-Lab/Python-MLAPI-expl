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

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import seaborn as sns


# In[ ]:


ts = pd.read_csv('../input/test.csv', index_col=0)
tr = pd.read_csv('../input/train.csv', index_col=0)
sa = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


tr.shape


# In[ ]:


dat = np.zeros(368)
for i in range(368):
    dat[i] = tr.iloc[:, [i,369]].corr().iloc[0,1]


# In[ ]:


s = pd.Series(dat)
s.sort_values(inplace=True)


# In[ ]:


s


# In[ ]:



