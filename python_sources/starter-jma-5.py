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


flow = pd.read_csv("../input/flow_2017.csv")
humidity = pd.read_csv("../input/humidity_2017.csv")
temp = pd.read_csv("../input/temperature_2017.csv")
weight = pd.read_csv("../input/weight_2017.csv")


# In[ ]:


flow.head()


# In[ ]:


df = pd.concat([flow, humidity, temp, weight], axis=1, join='inner').sort_index()
corr_mat = df.corr(method='pearson')
plt.figure(figsize=(20,10))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')


# In[ ]:


df = pd.concat([flow, humidity, temp, weight], axis=1, join='inner').sort_index()
plt.matshow(df.corr())
plt.colorbar()
plt.show()


# In[ ]:


flow = pd.read_csv("../input/flow_wurzburg.csv")
humidity = pd.read_csv("../input/humidity_wurzburg.csv")
temp = pd.read_csv("../input/temperature_wurzburg.csv")
weight = pd.read_csv("../input/weight_wurzburg.csv")
df1 = pd.concat([flow, humidity, temp, weight], axis=1, join='inner').sort_index()


# In[ ]:


plt.matshow(df1.corr())
plt.colorbar()
plt.show()


# In[ ]:


corr_mat = df1.corr(method='pearson')
plt.figure(figsize=(20,10))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')


# In[ ]:


flow = pd.read_csv("../input/flow_schwartau.csv")
humidity = pd.read_csv("../input/humidity_schwartau.csv")
temp = pd.read_csv("../input/temperature_schwartau.csv")
weight = pd.read_csv("../input/weight_schwartau.csv")
df2 = pd.concat([flow, humidity, temp, weight], axis=1, join='inner').sort_index()


# In[ ]:


plt.matshow(df2.corr())
plt.colorbar()
plt.show()


# In[ ]:


corr_mat = df2.corr(method='pearson')
plt.figure(figsize=(20,10))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')

