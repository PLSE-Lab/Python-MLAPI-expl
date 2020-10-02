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


# In[ ]:


md = pd.read_csv('../input/meningitis_dataset.csv')


# In[ ]:


md.head()


# In[ ]:


md.describe()


# In[ ]:


plt.matshow(md.corr())
plt.colorbar()
plt.show()


# In[ ]:


sns.lineplot(x='gender_male', y='gender_female', data=md)


# In[ ]:


p = md.hist(figsize = (20,20))


# In[ ]:


plt.figure()
sns.distplot(md['gender_male'])
plt.show()
plt.close()


# In[ ]:


plt.figure()
sns.distplot(md['gender_female'])
plt.show()
plt.close()


# In[ ]:


sns.kdeplot(data=md['gender_male'],label='Male',shade=True)


# In[ ]:


sns.kdeplot(data=md['gender_female'],label='Female',shade=True)


# In[ ]:




