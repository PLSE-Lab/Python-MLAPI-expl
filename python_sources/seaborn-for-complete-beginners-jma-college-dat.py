#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set(style="whitegrid")
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import warnings
warnings.filterwarnings('ignore')

dat = pd.read_csv('../input/college-data/data.csv')


# In[ ]:


dat.head


# In[ ]:


dat.info()


# In[ ]:


dat['accept'].value_counts()


# In[ ]:


f, ax = plt.subplots(figsize=(8,6))
x = dat['apps']
ax = sns.distplot(x, bins=10)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(8,6))
x = dat['accept']
ax = sns.distplot(x, bins=10)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(8,6))
x = dat['accept']
ax = sns.distplot(x, bins=10, vertical = True)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(8,6))
x = dat['apps']
x = pd.Series(x, name="variable")
ax = sns.kdeplot(x)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(8,6))
x = dat['accept']
x = pd.Series(x, name="variable")
ax = sns.kdeplot(x)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(8,6))
x = dat['apps']
ax = sns.distplot(x, kde=False, rug=True, bins=10)
plt.show()


# In[ ]:


f, ax = plt.subplots(figsize=(8,6))
x = dat['accept']
ax = sns.distplot(x, kde=False, rug=True, bins=10)
plt.show()


# In[ ]:


g = sns.JointGrid(x="apps", y="accept", data=dat, space=0)
g = g.plot_joint(sns.kdeplot, cmap="Blues_d")
g = g.plot_marginals(sns.kdeplot, shade=True)


# In[ ]:


g = sns.JointGrid(x="apps", y="enroll", data=dat, space=0)
g = g.plot_joint(sns.kdeplot, cmap="Blues_d")
g = g.plot_marginals(sns.kdeplot, shade=True)

