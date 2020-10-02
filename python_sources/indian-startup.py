#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df="../input/startup_funding.csv"
data=pd.read_csv(df)
print(data)


# In[ ]:


print(data.head())


# In[ ]:


print(data.columns)


# In[ ]:


print(data.info())


# In[ ]:


print("Identifying null values based on ascending order")
data.isnull().sum().sort_values(ascending =False)


# In[ ]:


data.drop(['Remarks'],axis=1,inplace=True)


# In[ ]:


print("After dropping Remarks column")
print(data.columns)


# In[ ]:


data['AmountInUSD'] = data['AmountInUSD'].apply(lambda x:float(str(x).replace(",","")))


# In[ ]:


print(data['AmountInUSD'].values)


# In[ ]:


data['AmountInUSD']=data['AmountInUSD'].astype(float)
print(data.info())


# In[ ]:


print(data['AmountInUSD'].values)


# In[ ]:


s=data['AmountInUSD'].mean()


# In[ ]:


data['AmountInUSD'].replace(np.NAN,s)


# In[ ]:


data['AmountInUSD'].sum()


# In[ ]:


data['Date']=data['Date'].replace({"12/05.2015":"12/05/2015"})
data['Date']=data['Date'].replace({"13/04.2015":"13/04/2015"})
data['Date']=data['Date'].replace({"15/01.2015":"15/01/2015"})
data['Date']=data['Date'].replace({"22/01//2015":"22/01/2015"})


# In[ ]:


print(data['Date'].values)


# In[ ]:


data["yearmonth"] = (pd.to_datetime(data['Date'],format='%d/%m/%Y').dt.year*100)+(pd.to_datetime(data['Date'],format='%d/%m/%Y').dt.month)
temp = data['yearmonth'].value_counts().sort_values(ascending = False)
print("Number of funding per month in decreasing order (Funding Wise)\n\n",temp)
year_month = data['yearmonth'].value_counts()


# In[ ]:


data['AmountInUSD'].min()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import dateutil
import squarify


# In[ ]:


plt.figure(figsize=(20,15))
sns.barplot(year_month.index, year_month.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Year-Month of transaction', fontsize=15,color='red')
plt.ylabel('Number of fundings made', fontsize=15,color='red')
plt.title("Year-Month - Number of Funding Distribution", fontsize=18)
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(data['yearmonth'],data['AmountInUSD'], alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('YearMonth', fontsize=12)
plt.ylabel('Amonut Of Investments', fontsize=12)
plt.title("YearMonth - Number of fundings distribution", fontsize=16)
plt.show()


# In[ ]:


print("Total number of startups")
len(data['StartupName'])


# In[ ]:


print("Unique startups")
len(data['StartupName'].unique())


# In[ ]:


tot = (data['StartupName'].value_counts())
c=0
for i in tot:
    if i > 1:
        c=c+1
print("Startups that got funding more than 1 times = ",c)


# In[ ]:


funt_count  = data['StartupName'].value_counts()
fund_count = funt_count.head(20)
print(fund_count)


# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(fund_count.index, fund_count.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Startups', fontsize=15)
plt.ylabel('Number of fundings made', fontsize=15)
plt.title("Startups-Number of fundings distribution", fontsize=16)
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(fund_count.index, fund_count.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Startups', fontsize=15)
plt.ylabel('Number of fundings made', fontsize=15)
plt.title("Startups-Number of fundings distribution", fontsize=16)
plt.show()


# In[ ]:


print("Unique Industry verticals")
len(data['IndustryVertical'].unique())


# In[ ]:


IndustryVert = data['IndustryVertical'].value_counts().head(20)
print(IndustryVert)


# In[ ]:


plt.figure(figsize=(15,8))
sns.barplot(year_month.index, year_month.values, alpha=0.9)
plt.xticks(rotation='vertical')
plt.xlabel('Year-Month of transaction', fontsize=12)
plt.ylabel('Number of fundings made', fontsize=12)
plt.title("Year-Month - Number of Funding Distribution", fontsize=16)
plt.show()


# In[ ]:




