#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import os 
import pandas as pd

dat = pd.read_csv('../input/median-listing-price-1-bedroom/median_price.csv')


# In[ ]:


dat.head()


# In[ ]:


dat.describe()


# In[ ]:


plt.matshow(dat.corr())
plt.colorbar()
plt.show()


# In[ ]:


plt.scatter(dat['2015-12'], dat['2016-01'])


# In[ ]:


sns.regplot(x=dat['2015-12'], y=dat['2016-01'])


# In[ ]:


plt.style.use('fast')
sns.jointplot(x='2015-12', y='2016-01', data=dat)
plt.show()


# In[ ]:


sns.lineplot(x='2015-12', y='2016-01', data=dat)

