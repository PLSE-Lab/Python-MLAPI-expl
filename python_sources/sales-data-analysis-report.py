#!/usr/bin/env python
# coding: utf-8

# # Sales analysis
# 
# ![](https://www.growthaspire.com/wp-content/uploads/2018/11/Sales-data-analysis-for-sales-managers.png)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Import necessary libraries

# In[ ]:


import pandas as pd
import matplotlib as plt


# ## Import data

# In[ ]:


all_data = pd.read_csv('/kaggle/input/sales-analysis/SalesAnalysis.csv')


# ### Explore data

# In[ ]:


all_data.head()


# ## Clean up the data!
# The first step in this is figuring out what we need to clean. I have found in practice, that you find things you need to clean as you perform operations and get errors. Based on the error, you decide how you should go about cleaning the data

# In[ ]:


all_data.isnull().values.any()


# ### Drop rows of NAN

# In[ ]:


all_data = all_data.dropna(how='all')
all_data.head()


# ### Get rid of text in order date column

# In[ ]:


all_data = all_data[all_data['Order Date'].str[0:2]!='Or']


# ## Make columns correct type

# In[ ]:


all_data['Quantity Ordered'] = pd.to_numeric(all_data['Quantity Ordered'])
all_data['Price Each'] = pd.to_numeric(all_data['Price Each'])


# ### Augment data with additional columns
# Add month column

# In[ ]:


all_data['Month'] = all_data['Order Date'].str[0:2]
all_data['Month'] = all_data['Month'].astype('int32')
all_data.head()


# ### Add city column

# In[ ]:


def get_city(address):
    return address.split(",")[1].strip(" ")

def get_state(address):
    return address.split(",")[2].split(" ")[1]

all_data['City'] = all_data['Purchase Address'].apply(lambda x: f"{get_city(x)}  ({get_state(x)})")
all_data.head()


# # Data Exploration!
# ## Question 1: What was the best month for sales? How much was earned that month?
# 

# In[ ]:


all_data['Sales'] = all_data['Quantity Ordered'].astype('int') * all_data['Price Each'].astype('float')


# In[ ]:


re=all_data.groupby(['Month']).sum()
re


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')
months = range(1,13)

plt.figure(figsize=(12,8))
plt.bar(months,re['Sales'])
plt.xticks(months)
plt.title('Best month for sales',fontsize=18)
plt.ylabel('Sales in USD ($)',fontsize=13)
plt.xlabel('Month number',fontsize=13)
plt.show()


# ## Answer : December the best month for sales. 4.613443e+06 was earned that month.

# ## Question 2: What city sold the most product?

# In[ ]:


all_data.groupby(['City']).sum()


# In[ ]:


import matplotlib.pyplot as plt

keys = [city for city,df in all_data.groupby(['City'])]
plt.figure(figsize=(12,8))
plt.bar(keys,all_data.groupby(['City']).sum()['Sales'])
plt.ylabel('Sales in USD ($)')
plt.xlabel('Month number')
plt.xticks(keys, rotation='vertical', size=8)
plt.show()


# ## Answer : San Francisco (CA) sold the most product.

# ## Question 3: What time should we display advertisements to maximize likelihood of customer's buying product?

# In[ ]:


# Add hour column
all_data['Hour'] = pd.to_datetime(all_data['Order Date']).dt.hour
all_data['Minute'] = pd.to_datetime(all_data['Order Date']).dt.minute
all_data['Count'] = 1
all_data.head()


# In[ ]:


keys = [pair for pair, df in all_data.groupby(['Hour'])]
plt.figure(figsize=(12,8))
plt.plot(keys, all_data.groupby(['Hour']).count()['Count'], 'go--', linewidth=2, markersize=12)
plt.xticks(keys)
plt.xlabel('Hour',fontsize=13)
plt.ylabel('counts',fontsize=13)
plt.grid()
plt.show()


# ## Answer : My recommendation is slightly before 11am or 7pm we display advertisements to maximize likelihood of customer's buying product.

# ## Question 4: What product sold the most? Why do you think it sold the most?

# In[ ]:


product_group = all_data.groupby('Product')
quantity_ordered = product_group.sum()['Quantity Ordered']
plt.figure(figsize=(12,8))
keys = [pair for pair, df in product_group]
plt.bar(keys, quantity_ordered)
plt.xticks(keys, rotation='vertical', size=8)
plt.show()


# In[ ]:


prices = all_data.groupby('Product').mean()['Price Each']

fig, ax1 = plt.subplots(figsize=(12,8))

ax2 = ax1.twinx()
ax1.bar(keys, quantity_ordered, color='g')
ax2.plot(keys, prices, color='b')

ax1.set_xlabel('Product Name')
ax1.set_ylabel('Quantity Ordered', color='g')
ax2.set_ylabel('Price ($)', color='b')
ax1.set_xticklabels(keys, rotation='vertical', size=8)

fig.show()


# ## Answer : Batteries sold the most, because it is cheap.
# 

# If you have reached till here, So i hope you liked my Analysis.
# 
# Don't forget to upvote if you like it!.
# 
# I'm a beginner and any suggestion in the comment box is highly appreciated.
# 
# If you have any doubt reagrding any part of the notebook, feel free to comment your doubt in the comment box.
# 
# Thank you!!

# In[ ]:




