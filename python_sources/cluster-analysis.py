#!/usr/bin/env python
# coding: utf-8

# ### Import relevant libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Load the Data

# In[ ]:


# Load the standardized data
# index_col is an argument we can set to one of the columns
# this will cause one of the Series to become the index
data = pd.read_csv('/kaggle/input/Country clusters standardized.csv', index_col='Country')


# In[ ]:


data.head()


# In[ ]:


# Create a new data frame for the inputs, so we can clean it
x_scaled = data.copy()
# Drop the variables that are unnecessary for this solution
x_scaled.drop('Language', axis=1, inplace=True)


# In[ ]:


x_scaled


# In[ ]:


# Using the Seaborn method 'clustermap' we can get a heatmap and dendrograms for both the observations and the features
sns.clustermap(x_scaled, cmap='mako')


# In[ ]:




