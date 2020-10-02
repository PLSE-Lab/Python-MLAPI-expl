#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('../input/train.csv')


# In[ ]:


def find_nearest1(array,value):
    idx,val = min(enumerate(np.sort(array[array!=0])), key=lambda x: abs(x[1]-value))
    return val
targetsthatshouldnotbehere = [] #v3 changed the name to actually describe what it means!! ;)
for i in range(train.shape[0]):
    if(i%500==0):
        print(i)
    targetsthatshouldnotbehere.append(find_nearest1( train.iloc[i][train.columns[2:]].values, train.target[i] ))


# In[ ]:


plt.figure(figsize=(15,15))
_ = plt.scatter(np.log1p(targetsthatshouldnotbehere),np.log1p(train.target),s=1)


# In[ ]:


from sklearn.metrics import mean_squared_error
a = np.log1p(targetsthatshouldnotbehere)
b = np.log1p(train.target)
print(np.sqrt(mean_squared_error(b,a)))

