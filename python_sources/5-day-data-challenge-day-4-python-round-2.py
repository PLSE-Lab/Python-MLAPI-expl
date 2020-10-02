#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Followining:
# http://mailchi.mp/5f0a34899a89/data-challenge-day-1-read-in-and-summarize-a-csv-file-2576429


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


dataset = pd.read_csv('../input/anonymous-survey-responses.csv')


# In[ ]:


dataset.head()


# In[ ]:


frequency_table = dataset['Just for fun, do you prefer dogs or cat?'].value_counts()


# In[ ]:


plt.bar(frequency_table.index, frequency_table.values)


# In[ ]:


sns.barplot(frequency_table.index, frequency_table.values)


# In[ ]:


sns.countplot(dataset['Just for fun, do you prefer dogs or cat?'])


# In[ ]:




