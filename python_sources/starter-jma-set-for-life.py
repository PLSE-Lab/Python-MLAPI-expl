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

sol= pd.read_csv('../input/Set for Life.csv')


# In[ ]:


sol.head()


# In[ ]:


sol.describe()


# In[ ]:


plt.matshow(sol.corr())
plt.colorbar()
plt.show()


# In[ ]:


plt.scatter(sol['Ball 1'], sol['Life Ball'])


# In[ ]:


sns.regplot(x=sol['Ball 1'], y=sol['Life Ball'])


# In[ ]:


sns.lineplot(x='Ball 1', y='Life Ball', data=sol)


# In[ ]:


plt.style.use('fast')
sns.jointplot(x = 'Ball 1', y = 'Life Ball', data = sol)
plt.show()


# In[ ]:


q = sns.boxenplot(x = sol['Ball 1'], y = sol['Life Ball'], palette = 'rocket')

