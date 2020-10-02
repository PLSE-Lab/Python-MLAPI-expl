#!/usr/bin/env python
# coding: utf-8

# In[3]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

transactions = pd.read_csv('../input/transactions.csv')


# In[4]:


#Summary of transaction data set
transactions.info()


# In[5]:


#Numbers of Columns
transactions.shape[1]


# In[6]:


# Numbers of Records
transactions.shape[0]


# In[7]:


# Get the row names
transactions.index.values


# In[8]:


#Get the column names
transactions.columns.values


# In[10]:


#view top 10 records
transactions.head(10)


# In[11]:


#Change the name of Column "Quantity" to "Quant"
transactions.rename(columns={'Quantity' :'Quant'})


# In[12]:


#Change the name of columns ProductID and UserID to PID and UID respectively 
transactions.rename(columns ={"ProductID":"PID","UserID": "UID"})


# In[14]:


#Order the rows of transactions by TransactionId decending
# if ascending then ascending = True,
transactions.sort_values('TransactionID', ascending=False)


# In[15]:


# Order the rows of transactions by Quantity ascending, TransactionDate descending

transactions.sort_values(['Quantity','TransactionDate'],ascending=[True,False])


# In[16]:


# Set the column order of transactions as ProductID, Quantity, TransactionDate, TransactionID, UserID
transactions[['ProductID', 'Quantity', 'TransactionDate', 'TransactionID', 'UserID']]


# In[30]:


# Make UserID the first column of transactions
transactions[pd.unique(['UserID'] + transactions.columns.values.tolist()).tolist()]


# In[36]:


#Extracting arrays from a DataFrame
# Get the 2nd column
transactions[:2]


# In[37]:


transactions.iloc[1]


# In[38]:


# Get the ProductID array
transactions.ProductID.values


# In[39]:


#Get the productId array using a variable 
col= "ProductID"
transactions[[col]].values[:,0]


# In[43]:


#Row Subsetting


# In[44]:


#Subset rows 1,3 and 6
#transactions.iloc[[1-1,3-1,6-1]]
transactions.iloc[[0,2,5]]


# In[45]:


#subset rows excluding 1,3, and 6
transactions.drop([0,2,5],axis=0)


# In[46]:


#Subset the fist three rows
transactions[:3]
transactions.head(3)


# In[47]:


# Subset the last 2 rows
transactions.tail(2)


# In[50]:


#subset rows excluding the last 2 rows
transactions.head(-2)


# In[51]:


# Subset rows excluding the first 3 rows
transactions[3:]
transactions.tail(-3)


# In[52]:


#Subset rows where Quantity > 1
transactions[(transactions.Quantity >1 )]


# In[53]:


#Subset rows where UserID =2
transactions[transactions.UserID ==2]


# In[54]:


# Subset rows where Quantity > 1 and UserID = 2
transactions[(transactions.Quantity >0) & (transactions.UserID == 2)]


# In[55]:


# Subset rows where Quantity + UserID is > 3
transactions[(transactions.Quantity + transactions.UserID )> 3]


# In[56]:


#Subset rows where an external array,foo, is True
foo = np.array([True,False,True,False,True,False,True,False,True,False])

transactions[foo]


# In[57]:


# Subset rows where an external array, bar, is positive
bar = np.array([1, -3, 2, 2, 0, -4, -4, 0, 0, 2])
bar
transactions[bar > 0]


# In[58]:


# Subset rows where foo is TRUE or bar is negative
transactions[foo | (bar <0)]


# In[59]:


# Subset the rows where foo is not TRUE and bar is not negative

transactions[~foo | bar >= 0]


# In[ ]:


#Column excercises


# In[60]:


#Subset by columns 1 and 3
transactions.iloc[:,[0,2]]


# In[61]:


## Subset by columns TransactionID and TransactionDate
transactions[['TransactionID','TransactionDate']]


# In[63]:


# Subset by columns TransactionID and TransactionDate wtih logical operator
transactions.loc[transactions.TransactionID >5,['TransactionID','TransactionDate']]


# In[64]:


# Subset columns by a variable list of columm names
cols = ["TransactionID","UserID","Quantity"]
transactions[cols]


# In[65]:


# Subset columns excluding a variable list of column names
cols = ["TransactionID", "UserID", "Quantity"]
transactions.drop(cols,axis =1)


# In[ ]:


#Inserting and updating values


# In[66]:


# Convert the TransactionDate column to type Date
transactions['TransactionDate'] = pd.to_datetime(transactions.TransactionDate)
transactions['TransactionDate']


# In[67]:


# Insert a new column, Foo = UserID + ProductID
transactions['Foo'] = transactions.UserID + transactions.ProductID
transactions['Foo']


# In[68]:


#post your query and comment.

