#!/usr/bin/env python
# coding: utf-8

# # **Analyzing StartUp Investments**

# ### Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Reading Data 

# In[ ]:


df = pd.read_csv('/kaggle/input/startup-investments-crunchbase/investments_VC.csv',encoding = 'unicode_escape')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


#Correcting columns name
df.columns


# In[ ]:


df.rename(columns={' market ': 'market', ' funding_total_usd ': 'funding_total_usd'}, inplace = True)


# In[ ]:


df.dropna(inplace = True)


# In[ ]:


#Checking if are duplicated rows
df.duplicated().sum()


# ### Exploring Data

# In[ ]:


#Market
market_count  = df['market'].value_counts()
market_count = market_count[:10,]
plt.figure(figsize=(16,12))
sns.barplot(market_count.index, market_count.values, alpha=0.8)
plt.title('Top 10 quantity of startups by market')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Market', fontsize=12)
plt.show()


# In[ ]:


#Country
country_count  = df['country_code'].value_counts()
country_count = country_count[:10,]
plt.figure(figsize=(16,12))
sns.barplot(country_count.index, country_count.values, alpha=0.8)
plt.title('Top 10 quantity of startups by country')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Country', fontsize=12)
plt.show()


# As we can see, this dataset is about American's as Canadian's StartUps, with majority of American ones

# In[ ]:


#Status
status_count  = df['status'].value_counts()
status_count = status_count[:10,]
plt.figure(figsize=(16,12))
sns.barplot(status_count.index, status_count.values, alpha=0.8)
plt.title('Top 10 quantity of startups by Status')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Status', fontsize=12)
plt.show()


# ### Trying understand and pre processing the column 'funding_total_usd'

# In[ ]:


df.funding_total_usd.replace(' -   ',0, inplace = True)


# In[ ]:


df.funding_total_usd


# In[ ]:


df.funding_total_usd.head().replace


# #### Waiting for dataset owner answer about that column

# ## Profile about which all 3 status

# In[ ]:


operating = df[df.status == 'operating']
acquired = df[df.status == 'acquired']
closed = df[df.status == 'closed']


# In[ ]:


operating_count  = operating['market'].value_counts(normalize=True)
operating_count = operating_count[:10,]

acquired_count  = acquired['market'].value_counts(normalize=True)
acquired_count = acquired_count[:10,]

closed_count  = closed['market'].value_counts(normalize=True)
closed_count = closed_count[:10,]

print('Operating')
print(operating_count)
print('-------------------------------------------')
print('Acquired')
print(acquired_count)
print('-------------------------------------------')
print('Closed')
print(closed_count)


# In[ ]:


#Market
operating_count  = operating['market'].value_counts(normalize=True)
operating_count = operating_count[:10,]
plt.figure(figsize=(24,8))
sns.barplot(operating_count.index, operating_count.values, alpha=0.8)
plt.title('Top 10 quantity of operating startups by market')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Market', fontsize=12)
plt.show()


# In[ ]:


#Market
acquired_count  = acquired['market'].value_counts(normalize=True)
acquired_count = acquired_count[:10,]
plt.figure(figsize=(24,8))
sns.barplot(acquired_count.index, acquired_count.values, alpha=0.8)
plt.title('Top 10 quantity of acquired startups by market')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Market', fontsize=12)
plt.show()


# In[ ]:


#Market
closed_count  = closed['market'].value_counts(normalize=True)
closed_count = closed_count[:10,]
plt.figure(figsize=(24,8))
sns.barplot(closed_count.index, closed_count.values, alpha=0.8)
plt.title('Top 10 quantity of closed startups by market')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Market', fontsize=12)
plt.show()


# In[ ]:




