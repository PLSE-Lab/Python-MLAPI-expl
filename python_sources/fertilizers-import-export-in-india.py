#!/usr/bin/env python
# coding: utf-8

# Hello all,
# This is my first notebook. I am practicing this dataset for data visualization.
# Any suggestions would be appreciated. Thank you!

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/fertilizers-by-product-fao/FertilizersProduct.csv', encoding='latin-1')


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(10,6))
plt.tight_layout()
sns.set_style('whitegrid')
sns.countplot(x='Year',data=df)


# Highest import-export of fertilizers happened in year 2011.

# In[ ]:


IndiaData = df[df['Area']=='India']


# In[ ]:


plt.figure(figsize=(10,6))
IndiaData['Year'].hist(bins=30)


# In last 15 years, India has always been importing and exporting around 80-90 fertilizers.

# In[ ]:


plt.figure(figsize=(10,6))
sns.countplot(x='Year',data = IndiaData,hue='Element')


# You can see from above graph that, India has exported more number of fertilizers rather than importing them most of the times.

# In[ ]:


plt.figure(figsize=(15,8))
chart = sns.countplot(x = 'Item', data=IndiaData)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')
chart.plot()


# Urea and Ammonium sulphate are two mostly used fertilizers in India.

# In[ ]:


plt.figure(figsize=(15,8))
chart = sns.countplot(x = 'Item', data=IndiaData[IndiaData['Element']=='Agricultural Use'])
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')
chart.plot()


# India has been using Urea, Ammonium Sulphate, CAN, SOP fertilizers in higher amounts for agricultural use.

# In[ ]:


plt.figure(figsize=(10,6))
dat = IndiaData[IndiaData['Element']=='Production']
chart = sns.countplot(x = 'Item', data=dat)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')
chart.plot()


# India has been producing only 11 fertilizers.

# In[ ]:


plt.figure(figsize=(10,6))
export_data = IndiaData[IndiaData['Element']=='Export Value']
sns.lineplot(x = 'Year', y='Value', data=export_data)


# India generated largest revenue around 8000 x 1000 US$ in 2009 year through exportation.

# In[ ]:


chart = sns.barplot(x='Item',y='Value',data=export_data)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90, horizontalalignment='center')
chart.plot()


# 'NPK fertilizers' has generated most of the revenues in India.

# In[ ]:




