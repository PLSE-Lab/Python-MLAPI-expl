#!/usr/bin/env python
# coding: utf-8

# ## [Seaborn](https://seaborn.pydata.org/) 
# 
# 

# ## Visualizing the Titanic Data
# 
# We will be working with the famous titanic data set for these exercise with a focus on the visualization of the data
# I find it enjoyable working with data visualizations as they provide insights into datasets that on occasions I don't anticipate on.
# 
# Here are a couple that are easy one-liners

# In[20]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


sns.set_style('whitegrid')


# ### Load Titanic dataset

# In[22]:


# titanic = sns.load_dataset('titanic')
# sns.load_dataset('titanic')

titanic = pd.read_csv('../input/train.csv')
titanic.head()


# ## Joint Plot
# 

# In[23]:


sns.jointplot(x='Fare',y='Age',data=titanic)


# ## Distribution Plot

# In[24]:


sns.distplot(titanic["Fare"],kde=False) #without the kde


# ## Box Plot

# In[25]:


sns.boxplot(x='Pclass',y='Age',data=titanic)


# ## Swarm plot

# In[26]:


sns.swarmplot(x='Pclass',y='Age',data=titanic)


# ## Count plot

# In[27]:


sns.countplot(x='Sex',data=titanic)


# ## Heat maps

# In[28]:


# for heat maps, indexing / correaltions needs to be established
tc = titanic.corr()
sns.heatmap(tc,cmap='coolwarm')
plt.title('titanic.corr()')


# ## Facet Grid

# In[29]:


g = sns.FacetGrid(data=titanic, col='Sex')
g.map(sns.distplot, 'Age',kde=False)


# In[ ]:




