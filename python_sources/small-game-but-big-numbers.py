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


n = 200000


# In[ ]:


get_ipython().run_cell_magic('time', '', 'series_1 = np.random.randint(low = 1,high = 1000,size = n)\nseries_1_T = series_1.reshape(n,1)\nseries_2  = np.random.randint(low = 1,high = 1000,size = n)\nseries_2_T = series_2.reshape(n,1)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def differ(x):\n    count = 0\n    tabel_1 = series_1 + series_1_T[x:x+2000]\n    tabel_2 = series_2 + series_2_T[x:x+2000]\n    diff= tabel_1[tabel_1>tabel_2].shape[0]\n    count += diff\n    return count')


# In[ ]:


arr = pd.DataFrame(data = np.arange(0,n,2000),columns = ["numbers"])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'count_each_run = arr["numbers"].apply(differ)')


# In[ ]:


count_each_run.sum()


# This is just a small game, it doesn't relate to data science project so much. However, it works with really big numbers and big shape of arrays. I believe this small job would be a good practice for dealing with big things. Please help me to improve my code. Thanks a lot.
