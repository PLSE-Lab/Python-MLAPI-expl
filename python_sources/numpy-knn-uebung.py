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


get_ipython().system('dir ../input/male-daan-schnell-mal-klassifizieren')


# In[ ]:


fn = '../input/male-daan-schnell-mal-klassifizieren/train.csv'
import pandas as pd
pd.read_csv(fn).head(5)


# In[ ]:


df = pd.read_csv(fn,index_col='Id')
df.head()


# In[ ]:


df.dtypes


# In[ ]:


data = df


# In[ ]:


data


# In[ ]:


Xtrain = data[['X1','X2']]
Xtrain


# In[ ]:


y = data.y
y


# In[ ]:





# In[ ]:


import numpy as np
np.mean(Xtrain,axis=0)


# In[ ]:


Xtest = Xtrain.mean()
Xtest


# In[ ]:


k=7


# In[ ]:


Xtrain.values - Xtest.values
Xtrain.shape


# In[ ]:


Xtest.values.shape


# In[ ]:


7^2


# In[ ]:


distanz = (((Xtrain - Xtest)**2).sum(axis=1))**0.5


# In[ ]:


distanz


# In[ ]:


a= np.array([1,4,8,2])
sorted_indices = np.argsort(a)
a[sorted_indices]


# In[ ]:


sorted_indices = np.argsort(distanz)


# In[ ]:


distanz.sort_values('')

