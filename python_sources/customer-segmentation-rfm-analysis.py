#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Customer segmentation is a method of dividing customers into groups or clusters on the basis of common characteristics. The market researcher can segment customers into the B2C model using various customer's demographic characteristics such as occupation, gender, age, location, and marital status. Here we predict customer Segment using RFM analysis.
# 

# In[ ]:


#importing requied libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from mlxtend.frequent_patterns import apriori #, association_rule


# In[ ]:


#Loading the data
#may be there is non standerd null values, replace all as NaN
df=pd.read_csv("/kaggle/input/transaction-data/transaction_data.csv",na_values=(" ","?","_","-1"))
df.head()


# In[ ]:


#disply the bottom 5 rows
df.tail()


# In[ ]:


#explore total number of rows and columns
df.shape


# **Data Preprocessing**

# In[ ]:



df=df[pd.notnull(df["UserId"])]


# In[ ]:


#dropingduplicates record because a huge data set as in thes case contains 1083818 recodes
#often have some duplicate data which might be disturbing.
df=df.drop_duplicates()
df.head()


# In[ ]:


#checking the shape after removing duplicate values
df.shape


# In[ ]:


#explore columns of the data for forther use
df.columns


# In[ ]:


#removing spaces from item discription
df['ItemDescription']=df['ItemDescription'].str.strip()


# In[ ]:


#renaming the columns
df=df.rename(columns={"NumberOfItemsPurchased":"Quantity"})


# In[ ]:


#customers ordered negative quantity,which is not possible.so we filter quantity grater then zero
df=df[(df["Quantity"]>0)]


# In[ ]:


#finding the null values
df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


#droping the missing values
df=df.dropna()
df.count()


# Feature engineering

# In[ ]:


#adding a new column total Cost
df['Total_Cost']=df['Quantity']*df['CostPerItem']
df.head()


# In[ ]:


#finding those coustomers who done max shoping.
s_data=df.query('Country=="United Kingdom"').sort_values("Total_Cost",ascending=False)
s_data.head()


# In[ ]:


#finding most expensive items
most_exp_items=df.sort_values("CostPerItem",ascending=False)
most_exp_items.head()


# In[ ]:


#finding max and min for each column
df.agg(["max","min"])


# In[ ]:


df['Country'].value_counts()


# In[ ]:


df.Country.value_counts().nlargest(15).plot(kind="bar",figsize=(10,5))
plt.title("Number of Country")
plt.ylabel("number of Country")
plt.xlabel("country")


# #In this figer we can see most of thr customers are from "United Kingdom".so we filter data for UK customer

# In[ ]:


uk_data=df[df.Country== "United Kingdom"]


# In[ ]:


uk_data.head()


# In[ ]:


uk_data.describe()


# In[ ]:


#finding time when min and max transaction done
uk_data['TransactionTime'].min(),uk_data['TransactionTime'].max()


# In[ ]:


PRESENT=dt.datetime(2018,7,26)
uk_data['TransactionTime'] = pd.to_datetime(uk_data['TransactionTime'])


# In[ ]:


data=uk_data[["UserId","TransactionId","TransactionTime","Quantity","CostPerItem","Total_Cost"]]
data.head()


# **RFM Analysis**
# 
# RFM (Recency, Frequency, Monetary) analysis is a behavior-based approach grouping customers into segments. It groups the customers on the basis of their previous purchase transactions. How recently, how often, and how much did a customer buy. RFM filters customers into various groups for the purpose of better service. It helps managers to identify potential customers to do more profitable business. There is a segment of customer who is the big spender but what if they purchased only once or how recently they purchased? Do they often purchase our product? Also, It helps managers to run an effective promotional campaign for personalized service.

# Here, I am going to perform following opertaions:
# 
# For Recency, Calculate the number of days between present date and date of last purchase each customer.
# For Frequency, Calculate the number of orders for each customer. 
# For Monetary, Calculate sum of purchase price for each customer.

# In[ ]:


rfm= uk_data.groupby('UserId').agg({'TransactionTime': lambda date: (PRESENT - date.max()).days,
                                    'TransactionId': lambda num: len(num),
                                    'Total_Cost': lambda price: price.sum()})


# In[ ]:


rfm.columns


# In[ ]:


# Change the name of columns 
rfm.columns=['monetary','recency','frequency']


# In[ ]:


rfm['recency'] = rfm['recency'].astype(int)


# In[ ]:


rfm.head()


# Computing Quantile of RFM values
# 
# Customers with the lowest recency, highest frequency and monetary amounts considered as top customers.
# 
# qcut() is Quantile-based discretization function. qcut bins the data based on sample quantiles. For example, 1000 values for 4 quantiles would produce a categorical object indicating quantile membership for each customer.

# In[ ]:


rfm['r_quartile'] = pd.qcut(rfm['recency'], 4, ['1','2','3','4']) 
rfm['f_quartile'] = pd.qcut(rfm['frequency'], 4, ['4','3','2','1']) 
rfm['m_quartile'] = pd.qcut(rfm['monetary'], 4, ['4','3','2','1'])


# In[ ]:


rfm.head()


# RFM Result Interpretation
# 
# Combine all three quartiles(r_quartile,f_quartile,m_quartile) in a single column, this rank will help you to segment the customers well group

# In[ ]:


rfm['RFM_Score'] = rfm.r_quartile.astype(str)+ rfm.f_quartile.astype(str) + rfm.m_quartile.astype(str)


# In[ ]:


rfm.head()


# In[ ]:


rfm.tail()


# In[ ]:





# In[ ]:





# **pls upvote if you found helpfil.**
