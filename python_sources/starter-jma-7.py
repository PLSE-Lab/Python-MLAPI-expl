#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import cm
sns.set_style('ticks')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# In[ ]:


EMT = pd.read_csv("../input/pmsm_temperature_data.csv")
EMT.head()


# In[ ]:


EMT.describe()


# In[ ]:


corr_mat = EMT.corr(method='pearson')
plt.figure(figsize=(20,10))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')


# In[ ]:


plt.matshow(EMT.corr())
plt.colorbar()
plt.show()


# In[ ]:


sns.boxplot(EMT['coolant'])


# In[ ]:


plt.scatter(EMT['motor_speed'], EMT['torque'])


# In[ ]:


p = EMT.hist(figsize = (20,20))


# In[ ]:


ax = sns.violinplot(x=EMT["coolant"])


# In[ ]:


ax = sns.violinplot(x=EMT["motor_speed"])


# In[ ]:


ax = sns.violinplot(x=EMT["torque"])


# In[ ]:


sns.distplot(EMT["coolant"])


# In[ ]:


sns.distplot(EMT["torque"])

