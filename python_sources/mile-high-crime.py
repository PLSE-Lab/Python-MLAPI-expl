#!/usr/bin/env python
# coding: utf-8

# ## Mile High Crime
# 
# ![](https://www.dea.gov/sites/default/files/styles/crop_paragraph_hero/public/2018-08/denver_copy.jpg?h=a9ac1a00&itok=uXjl0udh)

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[ ]:


DATA_FILE='../input/crime.csv'
df = pd.read_csv(DATA_FILE)
df.head(4)


# In[ ]:


df.OFFENSE_CATEGORY_ID.value_counts().plot(kind='barh')


# In[ ]:


df.NEIGHBORHOOD_ID.value_counts()[:10].plot(kind='barh', title='High Crime Occurance')


# In[ ]:


df.NEIGHBORHOOD_ID.value_counts()[-10:].plot(kind='barh', title='Low Crime Occurance')


# In[ ]:




