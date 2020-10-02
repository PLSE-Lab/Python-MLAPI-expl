#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="white", color_codes=True)
from wordcloud import WordCloud
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read data
df = pd.read_csv('../input/startup_funding.csv')


# In[ ]:


df.head()


# ### Check for null values

# In[ ]:


df.isnull().sum(axis=0)


# ### Check for duplicates

# In[ ]:


df.duplicated().sum()


# ### Check Datatypes

# In[ ]:


df.dtypes


# ### Check for unique values

# In[ ]:


df['Amount in USD'].unique()


# ### Replace Null Values 

# In[ ]:


df["Amount in USD"] = df['Amount in USD'].str.replace(',','').str.extract('(^\d*)')
df["Amount in USD"] = df['Amount in USD'].replace('',np.nan)
df['Amount in USD'] = df['Amount in USD'].astype(float)
# Replace with mean
df['Amount in USD'].fillna(df['Amount in USD'].mean(),inplace=True)


# ### Correlation Matrix

# In[ ]:


import seaborn as sns
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True);


# ### Distribution of a variable

# In[ ]:


import seaborn as sns
sns.distplot(df['Amount in USD'])


# In[ ]:


plt.scatter(range(len(df["Amount in USD"])),np.sort(df["Amount in USD"].values))
plt.xlabel("index")
plt.ylabel("Funding in USD")
plt.show()


# There are few outliers in the data.

# In[ ]:


df['Industry Vertical'].value_counts().head(10).plot.barh()


# Companies related to *Consumer Internet* got more easily funding than comes *technology* and *eCommerce*.

# In[ ]:


df['SubVertical'].value_counts().head(10).plot.barh()


# In sub categories *Online Pharmacy* is most popular among investors.

# In[ ]:


df['Date ddmmyyyy'].value_counts().head(10).plot.barh()


# In[ ]:


sns.violinplot(x=df['Industry Vertical'][df['Industry Vertical'] == 'Consumer Internet'], y="Amount in USD", data=df);


# # Types of Investment

# In[ ]:


# Frequency of each category
df['InvestmentnType'].value_counts()


# In[ ]:


df['InvestmentnType'][df['InvestmentnType'] == 'SeedFunding'] = 'Seed Funding'
df['InvestmentnType'][df['InvestmentnType'] == 'Crowd funding'] = 'Crowd Funding'
df['InvestmentnType'][df['InvestmentnType'] == 'PrivateEquity'] = 'Private Equity'


# In[ ]:


df['InvestmentnType'].value_counts().head().plot.bar()


# Most funding falls under the category of *Seed Funding*.

# # Location

# In[ ]:


df['City  Location'].value_counts().head(10).plot.bar()


# It shows that *Bangalore* is the heart of startup industry.

# # Investors

# In[ ]:


names = df["Investorsxe2x80x99 Name"][~pd.isnull(df["Investorsxe2x80x99 Name"])]


# In[ ]:


wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(names))


# In[ ]:


plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")


# In[ ]:




