#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


raw_data = pd.read_csv('/kaggle/input/chicago-crime-for-da-train-question/Crimes_-_2001_to_present.csv', dtype={'ID': int, 'Case Number': object,'Date': object, 'Block': object, 'IUCR': object,'Primary Type': object,'Description': object,'Location, Description':object, 'Arrest': bool,'Domestic': bool,'Beat': int,'District': float,'Ward': float,'Community Area': float,'FBI Code': object,'X Coordinate': float,'Y Coordinate': float,'Year': int,'Updated On': object,'Latitude': float, 'Longitude': float,'Location': object})


# In[ ]:


raw_data.head()


# In[ ]:


raw_data.info()


# Dropping columns with no bearing to crime and beat:

# In[ ]:


raw_data = raw_data.drop(['ID', 'Case Number', 'Date', 'IUCR', 'X Coordinate', 'Y Coordinate', 'Updated On','Latitude', 'Longitude', 'Location'], axis = 1)


# In[ ]:


raw_data.head()


# In[ ]:


raw_data.describe()


# In[ ]:


raw_data.describe(include=[np.object])


# In[ ]:


raw_data.describe(include=[np.bool])


# We need to ennumerate the data for Arrest and Domestic numerically now. We will let 0 represent False and 1 represent True

# In[ ]:


raw_data['Arrest'] = raw_data['Arrest'].map( {True: 1, False: 0} ).astype(int)
raw_data['Domestic'] = raw_data['Domestic'].map( {True: 1, False: 0} ).astype(int)


# In[ ]:


raw_data.head()


# Primary Type is the type of crime committed...let's see what different types of crimes there are:

# In[ ]:


raw_data['Primary Type'].unique()


# We must make a delineation between what is a "relevant" crime to use and what isn't...

# Now, let's look at Location Description and ennumerate it:

# In[ ]:


raw_data['Location Description'].unique()


# 
