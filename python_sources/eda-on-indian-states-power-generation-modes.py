#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Importing dataset

# In[ ]:


dataset = pd.read_csv('../input/MOP_installed_capacity_sector_mode_wise.csv')


# #### Checking head of the dataset

# In[ ]:


dataset.head()


# ## EDA

# In[ ]:


dataset['Mode'].unique()


# In[ ]:


dataset['State'].unique()


# ### Checking the count of modes ie.,in how many places all modes are used

# In[ ]:


sns.countplot(x = 'Mode',data = dataset)


# ### Maximum Installed capacity

# In[ ]:


dataset[dataset['Installed Capacity'] == max(dataset['Installed Capacity'])]


# In[ ]:


state = dataset.groupby(by = 'State').sum().reset_index()


# In[ ]:


state


# In[ ]:


state.max()


# In[ ]:


fig_size = plt.figure(figsize = (50,20))
plt.rcParams.update({'font.size':35})
sns.barplot(x = 'State',y = 'Installed Capacity',data=state,palette='coolwarm')
plt.xlabel('State')
plt.ylabel('Installed Capacity')


# In[ ]:


state_type = dataset.groupby(by = ['State','Mode']).sum().reset_index()


# In[ ]:


state_type.head()


# ### Top 3 cities using Renewable Energy Sources

# In[ ]:


state_type[state_type['Mode'] == 'RES'].sort_values(by = 'Installed Capacity',ascending = False)[:3]


# ### Top 3 cities using Hydro Power

# In[ ]:


state_type[state_type['Mode'] == 'Hydro'].sort_values(by = 'Installed Capacity',ascending = False)[:3]


# ### Top 3 cities using Thermal Power

# In[ ]:


state_type[state_type['Mode'] == 'Thermal'].sort_values(by = 'Installed Capacity',ascending = False)[:3]


# ### Top 3 cities using Nuclear Power

# In[ ]:


state_type[state_type['Mode'] == 'Nuclear'].sort_values(by = 'Installed Capacity',ascending = False)[:3]


# In[ ]:


mode_based = dataset.groupby(by = 'Mode').sum().reset_index()


# In[ ]:


mode_based


# In[ ]:


plt.rcParams.update({'font.size':14})
plt.title('Power Generation Modes Distribution')
plt.pie(x = mode_based['Installed Capacity'],labels = mode_based['Mode'],
        colors = ['#65c6c4','#cf3030','#c9f658','#ff5a00'],explode = (0.0,0.0,0.1,0.0),shadow = True,autopct='%1d%%',
        startangle = 90)
plt.axis('equal')


# In[ ]:




