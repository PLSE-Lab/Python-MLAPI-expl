#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


df = pd.read_csv('../input/fertilizers-by-product-fao/FertilizersProduct.csv', encoding='latin-1')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(10,6))
plt.tight_layout()
sns.set_style('whitegrid')
sns.countplot(x = 'Year', data=df)


# In[ ]:


India = df[df['Area'] == "India"]


# In[ ]:


India.head()


# In[ ]:


len(India.Item.value_counts())


# In[ ]:


India.Year.hist(bins=20)


# In[ ]:


plt.figure(figsize=(10,6))
plt.tight_layout()
sns.countplot(x = 'Year', data=India)


# In[ ]:


plt.figure(figsize=(10,6))
plt.tight_layout()
sns.countplot(x = 'Year', data=India, hue = 'Element')


# In[ ]:


plt.figure(figsize=(10,6))
plt.tight_layout()
plot = sns.countplot(x = 'Item', data=India, hue = 'Element')
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()


# In[ ]:


plt.figure(figsize=(10,8))
plot = sns.countplot(x = "Item", data = India[India['Element'] == "Agricultural Use"])
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()


# In[ ]:


plt.figure(figsize=(10,8))
plot = sns.countplot(x = "Item",  data = India[India['Element'] == "Export Value"])
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()


# In[ ]:


plt.figure(figsize=(10,8))
plot = sns.countplot(x = "Item",  data = India[India['Element'] == "Import Value"])
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()


# In[ ]:


plt.figure(figsize=(10,8))
plot = sns.countplot(x = "Item",  data = India[India['Element'] == "Production"])
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()


# In[ ]:


data = India[India['Element'] == "Export Value"]


# In[ ]:


data.head()


# In[ ]:


plt.figure(figsize=(8,8))
sns.lineplot(x='Year', y='Value', data=data)


# **Exports seem to decrease after 2015**

# In[ ]:


plt.figure(figsize=(8,8))
plot = sns.barplot(x="Item", y="Value", data=data)
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()


# In[ ]:


importdata = India[India['Element'] == "Import Value"]


# In[ ]:


sum(importdata['Value'])


# In[ ]:


sum(exportdata['Value'])


# In[ ]:


plt.figure(figsize=(8,8))
sns.lineplot(x='Year', y='Value', data=importdata)


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(x='Year', y='Value', data=importdata, hue="Item")


# In[ ]:


plt.figure(figsize=(8,8))
plot = sns.barplot(x="Item", y="Value", data=importdata)
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()


# In[ ]:


agridata = India[India['Element'] == "Agricultural Use"]


# In[ ]:


agridata.head()


# In[ ]:


plt.figure(figsize=(8,8))
plot = sns.barplot(x="Item", y="Value", data=agridata)
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()


# In[ ]:


plt.figure(figsize=(8,8))
sns.lineplot(x='Year', y='Value', data=agridata)


# In[ ]:


df['Element'].unique()


# In[ ]:


exportdata = India[India['Element'] == "Export Quantity"]


# In[ ]:


plt.figure(figsize=(8,8))
plot = sns.barplot(x="Item", y="Value", data=exportdata)
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()


# In[ ]:


plt.figure(figsize=(8,8))
sns.lineplot(x='Year', y='Value', data=exportdata)


# In[ ]:


proddata = India[India['Element'] == "Production"]
proddata.head()


# In[ ]:


plt.figure(figsize=(8,8))
sns.lineplot(x='Year', y='Value', data=proddata)


# In[ ]:


plt.figure(figsize=(8,8))
plot = sns.barplot(x="Item", y="Value", data=proddata)
plot.set_xticklabels(plot.get_xticklabels(), rotation=90, horizontalalignment='center')
plot.plot()


# In[ ]:


plt.figure(figsize=(10,10))
sns.lineplot(x='Year', y='Value', data=proddata, hue=proddata['Item'])


# In[ ]:


sum(proddata['Value'])


# In[ ]:


df['Item'].unique()


# **CONCLUSIONS**

# 1. India deals with 22 fertilizers of the 23 fertilizers used worldwide.
# 2. The sum of total of imports of fertilizers is much more than exports over the years, hence India is dependent on imported fertilizers.
# 3. NPK Fertilizers, Nitrogeneous fertilizers are the most exported fertilizers.
# 4. Imports of fertilizers have decreased consistently in last 5 years. 
# 5. 'Diammonium phosphate (DAP)', 'Urea' are the most imported Fertilizers in India.
# 6. Use of fertilizers have been constant around the last decade except in 2015.
# 7. Urea remains the most produced fertilizer in India.
# 8. 'Urea' and 'Diammonium phosphate (DAP)' are the most fertilizers in India.
# 9. India is also steadily increasing production of 'Diammonium phosphate (DAP)' given its high use and import value.

# **Kudos to https://www.kaggle.com/komalbarge45 for her notebook https://www.kaggle.com/komalbarge45/fertilizers-import-export-in-india which proved to be of great help.**

# Please Upvote if you Like.
# Thanks.
