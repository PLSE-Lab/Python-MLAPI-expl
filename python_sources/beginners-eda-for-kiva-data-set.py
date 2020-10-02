#!/usr/bin/env python
# coding: utf-8

# **KIVA Data Set**
# 
# More EDA on the way!!
# 
# The approach I am using is to perform EDA on the all the inputs separately and them add them in every possible way and perform EDA on the merged data
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import time
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


loans = pd.read_csv("../input/kiva_loans.csv")


# In[ ]:


loans.columns


# In[ ]:


loans.head(2)


# In[ ]:


#Missing Value Identification 
pd.concat([loans.isnull().sum().sort_values(ascending=False)])


# In[ ]:


plt.subplots(figsize=(15,8))
plt.xticks(rotation='vertical')
sns.countplot(x="sector",data=loans,orient="h")


# In[ ]:


# Top Activity for which loans used for 
agg1 = pd.DataFrame(loans.activity.value_counts().head(15))
agg1= agg1.reset_index()
agg1.columns = ['Activity','Count']
plt.figure(figsize=(15,6))
plt.xticks(rotation="vertical")
sns.barplot(x="Activity",y="Count",data=agg1)


# In[ ]:


#Top Countries with most loans
agg1 = pd.DataFrame(loans.country.value_counts().head(15))
agg1= agg1.reset_index()
agg1.columns = ['Country','Count']
plt.figure(figsize=(15,6))
sns.barplot(x="Country",y="Count",data=agg1)


# In[ ]:


loans.repayment_interval.value_counts().plot(kind="pie",figsize=(6,6),autopct='%1.1f%%')
print("More than 50% of loans opted for Monthly repayment mode")


# In[ ]:


def gender(x):
    lis = x.split(",")
    lis = list(set(lis))
    lis = [x.strip() for x in lis]
    if len(lis)==2:
        return "Both"
    elif lis[0]=="female":
        return "female"
    else:
        return "male"

top_g = loans.borrower_genders.value_counts().reset_index().head(1)['index'][0]
loans.borrower_genders[loans.borrower_genders.isnull()]= top_g
loans['gender1'] = loans.borrower_genders.apply(gender)
loans.gender1.value_counts().plot(kind="pie",autopct="%1.1f%%",figsize=(6,6))


# * 64% groups had female borrowers only
# * 25% of groups had only male borrowers and 10% had both male and female borrowers
# 

# In[ ]:


def gender_cnt(x):
    lis = x.split(",")
    return len(lis)

loans['borrower_count'] = loans.borrower_genders.apply(gender_cnt)
plt.figure(figsize=(15,6))
sns.distplot(loans.borrower_count,bins=20)


# In[ ]:


plt.figure(figsize=(15,6))
plt.xlim([0,100])
plt.ylabel("Count")
sns.distplot(loans.term_in_months,kde=False)


# In[ ]:


sns.boxplot(data=loans[['funded_amount','loan_amount']],orient="h")


# In[ ]:


loans['funded_date'] = pd.to_datetime(loans.date,errors='coerce')
loans['Year']  = loans.funded_date.apply(lambda x : x.year)
loans['Month'] = loans.funded_date.apply(lambda x : x.month)


# In[ ]:


#Heat map to visualize the loan application count 
time_agg = loans.groupby(['Year','Month'],as_index=False).size().reset_index()
time_agg.columns = ['Year','Month','loan_count']
time_agg = time_agg.pivot(index='Month',columns='Year',values='loan_count')
time_agg = time_agg.fillna(0)
plt.figure(figsize=(12,6))
sns.heatmap(time_agg,cmap="YlGnBu")


# * In the first 6 months of 2017 we can see more loans have been disbursed than it was done in the past
# * We don't have data for second half of 2017 which is represented by white here

# In[ ]:




