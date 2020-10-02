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


# In[ ]:


import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df = pd.read_csv('/kaggle/input/startup-investments-crunchbase/investments_VC.csv',encoding = "ISO-8859-1")
df.head()


# ### Checking shape of the dataset

# In[ ]:


df.shape


# ### Now checking ofr NULL values

# In[ ]:


fig = plt.subplots(figsize=(20,10))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
plt.show()


# #### Here we can observe the yellow lines are the null values and if we keenly observe at the end of the dataset all the rows are having null values in order to get the count of null values we can go for its sum of

# In[ ]:


df.isnull().sum()


# #### From the above we can observe the count 4856 which is null value count of many columns. So we can infer that 4856 rows have NULL values

# In[ ]:


#Removing the rows with NULL values
startups = df.dropna(how='all')


# In[ ]:


startups.shape


# #### After removing the NULL values wee can find rows got reduced to 49438 rows

# In[ ]:


startups.tail()


# In[ ]:


startups.columns


# #### we can observe there are some spaces in column names hence we have to remove it

# In[ ]:


startups.columns = startups.columns.str.strip()
startups.columns


# #### Now we have removed the extra spaces in all column names

# ### Now we can analyse 'market' column

# In[ ]:


startups['market'].value_counts()[:20]


# #### Let's plot it

# In[ ]:


fig = plt.subplots(figsize=(20,10))
ax = sns.countplot(startups['market'],order=startups['market'].value_counts()[:20].index)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()


# #### So from the above countplot we can observe most of the startups has been started on SOFTWARE market followed by the BIOTECHNOLOGY and so on.

# ### In funding_total_usd column we can find -'s converting them into 0's

# In[ ]:


def clear_str(x):
    x = x.replace(',','').strip()
    x = x.replace('-','0')
    return int(x)


# In[ ]:


startups['funding_total_usd'] = startups['funding_total_usd'].apply(lambda x: clear_str(x))


# ### Now we can find for which market more funding has been done

# In[ ]:


startups['funding_total_usd'].isnull().sum()


# In[ ]:


fig = plt.subplots(figsize=(20,10))
market_fund = startups.groupby('market').sum()['funding_total_usd'].sort_values(ascending=False)[:10]
ax=sns.barplot(data = startups,x = market_fund.index, y= market_fund.values)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()


# #### Biotechnology has the highest funding among all other market

# ### Now we can analyse in which country has more number of startups has been started

# In[ ]:


fig = plt.subplots(figsize=(20,10))
ax = sns.countplot(startups['country_code'],order=startups['country_code'].value_counts()[:15].index)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()


# #### From the above plot we can observe the count of startup companies in USA is high compared to other countries

# ### Now we can find the count of the status

# In[ ]:


fig = plt.subplots(figsize=(20,10))
ax = sns.countplot(startups['status'],order=startups['status'].value_counts()[:15].index)
plt.xticks(rotation = 50)
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.2f'), (p.get_x() + p.get_width() / 2., p.get_height()), 
               ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.show()


# #### Status of 'operating' is highest followed by 'acquired'

# ### For the same status find the percentage

# In[ ]:


plt.figure(figsize = (10,10))
startups.status.value_counts().plot(kind='pie', explode=(0, 0.05, 0.1),autopct='%1.1f%%',startangle=45)
plt.title('Status')
plt.show()


# In[ ]:


startups.columns


# ### Now we can check the famous companies funding details like Facebook,Alibaba and Uber

# In[ ]:


sns.factorplot('name',data=startups[(startups['name']=='Facebook')|(startups['name']=='Alibaba')|(startups['name']=='Uber')],kind='count',hue='funding_total_usd')
plt.show()


# #### So Facebook has total funding compared to Alibaba and Uber

# In[ ]:




