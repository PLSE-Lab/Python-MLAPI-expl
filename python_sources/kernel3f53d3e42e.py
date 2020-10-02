#!/usr/bin/env python
# coding: utf-8

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


# ### Load data

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('/kaggle/input/us-breweries/breweries_us.csv')
df.head()


# ### See data types for each columns

# In[ ]:


df.info()


# ### Double check if there is not NULL values in each cols

# In[ ]:


df.isnull().sum()


# The above is telling us there is 8 null values in 'website' col but it does not matter for this data analysis.

# ### Drop cols including irrelevant info

# In[ ]:


df1 = df.drop(['brewery_name', 'address', 'website', 'state_breweries'], axis=1)
df1.head()


# ### Group the date by 'type' and see the number of breweries by each types

# In[ ]:


dataframe = df1.groupby('type').count()
dataframe


# This doesn't look orgarnized. Let's rename the col and sort the date by desending order.

# In[ ]:


dataframe.columns = ['Count of brewery types']
sorted_value_df = dataframe.sort_values('Count of brewery types', ascending=False)

sorted_value_df.head()


# ### Ready for box plot

# In[ ]:


ax = sorted_value_df.plot(kind='bar', facecolor='coral', figsize=(15,5))

ax.grid()
ax.set_title('Type of US breweries', fontsize=20, fontweight='bold', pad=30)
ax.set_xlabel('Types of US brewery', fontsize=15, fontweight='bold')
ax.set_ylabel('Number of brewery types', fontsize=15, fontweight='bold')
ax.legend(fontsize=15)


# Result: it shows us 'Brewpub' is the most popluar type of breweries in the US

# ### Additional data analysis

# Let's analyze data one more time to see which 'state' does have breweies the most

# In[ ]:


df_state = df1.groupby('state').count()
df_state.columns = ['Number of breweries']
sorted_value_state = df_state.sort_values('Number of breweries', ascending=False)

ax_state = sorted_value_state.plot(kind='bar', facecolor='violet', figsize=(20,10))

ax_state.grid()
ax_state.set_title('Number of breweries by US states', fontsize=20, fontweight='bold', pad=30)
ax_state.set_xlabel('US States', fontsize=15, fontweight='bold')
ax_state.set_ylabel('Number of breweries', fontsize=15, fontweight='bold')
ax_state.legend(fontsize=20)


# Result: California is the state that has the most breweries in the US
