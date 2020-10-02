#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/loan.csv', low_memory=False)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df.columns


# In[ ]:


plt.figure(figsize=(8, 5))
sns.barplot(y=df.term.value_counts(), x=df.term.value_counts().index, palette='spring')
plt.xticks(rotation=0)
plt.title("Loan's Term Distribution")
plt.ylabel("Count")


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(16,5))
sns.distplot(df['loan_amnt'], ax=ax[0])
ax[0].set_title("Loan Amount Distribution")
sns.distplot(df['funded_amnt'], ax=ax[1])
ax[1].set_title("Funded Amount Distribution")


# #### Fine, Let's Checkout the ```Grades```
# * ##### It appears that ```B & C grade``` are the dominant ones

# In[ ]:


# df.title.value_counts()[:10]
# sns.countplot(x='title', data=df)


# In[ ]:


fig, ax = plt.subplots(2, 1, figsize=(12, 8))
plt.subplots_adjust(wspace=1.0, hspace=0.50)
df.grade.value_counts().plot(kind="bar", ax=ax[0])
ax[0].set_title("Grade Count")
df.purpose.value_counts().plot(kind="bar", ax=ax[1])
ax[1].set_title("Loan Purposes")
plt.xticks(rotation=60)


# #### It appears that most of the loans are for ```debt_consolidation```

# In[ ]:


df.title = df.title.replace({'Debt Consolidation': 'Debt consolidation'})
df.title = df.title.replace({'debt consolidation': 'Debt consolidation'})


# In[ ]:


print(df.title.value_counts()[:10])
plt.figure(figsize=(16,5))
sns.barplot(x=df.title.value_counts()[:10].index, y=df.title.value_counts()[:10], data=df, palette='inferno')
plt.xticks(rotation=30)


# In[ ]:


print("Loan Amount Distribution BoxPlot")
plt.figure(figsize=(8,5))
sns.boxplot(x=df.term, y=df.loan_amnt)


# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(x=df.verification_status, y=df.loan_amnt)
plt.xlabel("Verification Status")
plt.ylabel("Loan Amount")


# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(x=df.verification_status, y=df.loan_amnt, hue=df.term, palette="terrain")
plt.xlabel("Verification Status")
plt.ylabel("Loan Amount")
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)


# In[ ]:


plt.figure(figsize=(16,8))
sns.boxplot(x=df.grade, y=df.loan_amnt, hue=df.term, palette="inferno")
plt.xlabel("Verification Status")
plt.ylabel("Loan Amount")
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)


# In[ ]:


def getMonth(x):
    return x.split('-')[0]
def getYear(x):
    return x.split('-')[1]

df['Month'] = df.issue_d.apply(getMonth)
df['Year'] = df.issue_d.apply(getYear)


# ### Loan Amount Distribution by ```Month```

# In[ ]:


plt.figure(figsize = (14,6))

g = sns.pointplot(x='Month', y='loan_amnt', 
              data=df)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Duration Distribuition", fontsize=15)
g.set_ylabel("Mean amount", fontsize=15)
g.set_title("Loan Amount by Months", fontsize=20)
plt.legend(loc=1)
plt.show()


# In[ ]:


plt.figure(figsize = (14,6))

g = sns.pointplot(x='Month', y='loan_amnt', 
              data=df, hue='grade')
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Duration Distribuition", fontsize=15)
g.set_ylabel("Mean amount", fontsize=15)
g.set_title("Loan Amount by Months", fontsize=20)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
plt.show()


# In[ ]:


plt.figure(figsize = (14,6))

g = sns.pointplot(x='Month', y='loan_amnt', 
              data=df, hue='term')
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_xlabel("Duration Distribuition", fontsize=15)
g.set_ylabel("Mean amount", fontsize=15)
g.set_title("Loan Amount by Months", fontsize=20)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.5)
plt.show()


# #### From the above graph a new insight has been obtained i.e. Mean Amount for 60 Months Tenure is very much greater as compared to 36 months Tenure

# In[ ]:


df.Month.value_counts().index


# In[ ]:


orderby = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.figure(figsize=(12, 7))
sns.countplot(x="Month", data=df,
             facecolor=(0, 0, 0, 0),
             linewidth=5,
             edgecolor=sns.color_palette("dark", 6),
             order=orderby)


# #### Seems that October and July have the highest number of applications

# In[ ]:


plt.figure(figsize=(12, 7))
sns.countplot(x="Year", data=df,
             facecolor=(0, 0, 0, 0),
             linewidth=5,
             edgecolor=sns.color_palette("dark", 6))


# #### Wow an exponential rise in the number of applications over a period of years
# ### That's Interesting

# ## More Coming Soon
