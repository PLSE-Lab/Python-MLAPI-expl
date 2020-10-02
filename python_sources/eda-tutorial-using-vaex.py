#!/usr/bin/env python
# coding: utf-8

# # Vaex? 
# 
# Well, this stuff could handle big stock of data!
# 
# It also has big use in my working field, astrohysics, as we handle HDF datasets.
# 
# Here is some Data Exloratory using Vaex instead of Pandas. 
# 
# 1. Data Loading
# 2. Quicklook
# 3. Describing data
# 4. Data Subsetting
# 5. Plotting
# 6. Vaex Datasets Plot

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Tips: install tensorflow 2 to enable vaex
# 
# otherwise it couldnt run on TF 1

# In[ ]:


get_ipython().system('pip install vaex==2.5.0 ')
get_ipython().system('pip install tensorflow==2  #For enabling vaex to operates ')


# ## Lets import some data

# In[ ]:


import vaex
df = vaex.read_csv('/kaggle/input/ngc-628-7793-krumholz-2015/opencluster.tsv', delimiter = ';')


# In[ ]:


df.head()


# ## Reading Columns
# 
# 

# Checking column names

# In[ ]:


df.columns


# Those 3 commands are equivalent

# In[ ]:


df.col.AV_84


# In[ ]:


df.AV_84


# In[ ]:


df['AV_84']


# ### Raw NumPy array

# In[ ]:


df.data.AV_84


# NumPy array

# In[ ]:


df.evaluate(df.AV_84)


# You could apply NumPy operation on these data.

# ## Describe data
# 
# It will show you the statistics for each columns

# In[ ]:


df.describe()


# ## Selection

# Lets see the Field column

# In[ ]:


df['Field'].unique()


# We only took the desired `NGC_7793e_l` column   

# In[ ]:


select = df[df['Field'] == 'NGC_7793e_l']


# In[ ]:


select.head(5)


# ## Plotting
# 
# ### We will use matlotlib and seaborn to plot

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# ### Preparing data

# In[ ]:


index = select.evaluate(select['AV_84'])


# Distribute to 15 interval

# In[ ]:


plt.hist(index, bins = 15)
plt.title('AV_84')
plt.xlabel('AV_84')
plt.ylabel('n')


# #### With seaborn

# In[ ]:


sns.distplot(index, bins = 15)
plt.title('AV_84')
plt.xlabel('AV_84')
plt.ylabel('n')


# In[ ]:


select.plot1d(select['AV_84'])


# In[ ]:


x = select.evaluate(select['logT_84'])
y = select.evaluate(select['AV_84'])


# In[ ]:


plt.scatter(x,y, s = 2, color = 'r')
plt.title('logT vs AV')
plt.xlabel('logT')
plt.ylabel('AV')


# # More Example From Vaex Dataset
# 
# You could visit more vaex use here https://github.com/vaexio/vaex
# 
# And here is the documentations https://vaex.readthedocs.io/en/latest/

# In[ ]:




