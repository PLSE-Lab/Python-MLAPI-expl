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

dat = pd.read_csv('../input/open-units/open_units.csv')


# In[ ]:


dat.head()


# In[ ]:


dat.describe()


# In[ ]:


plt.matshow(dat.corr())
plt.colorbar()
plt.show()


# In[ ]:


plt.scatter(dat['Units (4 Decimal Places)'], dat['Units per 100ml'])


# In[ ]:


sns.regplot(x=dat['Units (4 Decimal Places)'], y=dat['Units per 100ml'])


# In[ ]:


plt.style.use('fast')
sns.jointplot(x='Units (4 Decimal Places)', y='Units per 100ml', data=dat)
plt.show()


# In[ ]:


sns.lineplot(x='Units (4 Decimal Places)', y='Units per 100ml', data=dat)


# In[ ]:


p = dat.hist(figsize = (20,20))


# In[ ]:


ax = sns.violinplot(x=dat["Units per 100ml"])


# In[ ]:


ax = sns.violinplot(x=dat["Quantity"])


# In[ ]:


ax = sns.violinplot(x=dat["ABV"])


# In[ ]:


ax = sns.violinplot(y=dat["ABV"])

