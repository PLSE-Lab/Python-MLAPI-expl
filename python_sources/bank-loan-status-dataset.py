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


# Loan Status Classification Here in this notebook we take a look at the data from a bank/financial organization of all their loans. We explore various features about the borrowers like credit score, mortgage, annual income, years of employment and try to train our classifier to predict if the loan would be paid or not.

# Exploratory data analysis

# In[ ]:


credit_df=pd.read_csv("/kaggle/input/my-dataset/credit_train.csv")
credit_df.head()


# In[ ]:


credit_df.shape


# In[ ]:


credit_df.info()


# In[ ]:


credit_df.isnull().sum()


# This are the blanks or N/A columns in the data 

# In[ ]:


credit_df.describe()


# **Average credit score from above data is 1076.45.**

# In[ ]:


credit_df['Number of Credit Problems'].plot()

credit_df['Current Credit Balance'].plot()
# In[ ]:


credit_df.plot()


# In[ ]:


credit_df


# In[ ]:


credit_df.df = credit_df[credit_df['Credit Score']>800]
credit_df.head()


# *I noticed that credit score are within the range of* **300-850.** 
# > But in this dataset credit score are beyond **850.**
# I can try to find sense in this, the credit score in between **5958 to till 7290** is **charged off**.And also
# I got to know that those who have taken loan amount RS **9,99,99,999** are **Fully Paid up**.
# 

# In[ ]:


print("Value counts for each term: \n",credit_df['Term'].value_counts())
print("Missing data in loan term:",credit_df['Term'].isna().sum())


# In[ ]:


credit_df['Term'].replace(("Short Term","Long Term"),(0,1), inplace=True)
credit_df.head()


# In[ ]:


import pandas as pd


# In[ ]:


scount = credit_df[credit_df['Term'] == 0]['Term'].count()
lcount = credit_df[credit_df['Term'] ==1]['Term'].count()
data = {"Counts":[scount, lcount]}
credit_df = pd.DataFrame(data, index=["Short Term", "Long Term"])
credit_df.head()


# In[ ]:


credit_df.plot(kind="barh", title="Term of Loans")


# As per the above data we can see that demad for short term loan is much more than long term. 

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(font_scale = 2)
credit = pd.read_csv('/kaggle/input/my-dataset/credit_test.csv')


# In[ ]:


plt.figure(figsize=(20,8))

sns.countplot(credit['Years in current job'])


# **From the above diagram we can see that the person who are having more than 10 years of experience have taken maximum numbers of loan.**

# In[ ]:


dataframe = pd.read_csv('/kaggle/input/my-dataset/credit_test.csv')
dataframe = dataframe.drop(['Credit Score'], axis=1)


# In[ ]:


dataframe['Purpose'].value_counts().sort_values(ascending=True).plot(kind='barh', title="Purpose for Loans", figsize=(15,10))


# **Here we can say that maximum numbers of loan is taken for Debt Consolidation.** 
