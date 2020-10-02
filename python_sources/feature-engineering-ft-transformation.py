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


# pandas version:

# In[ ]:


df=pd.read_csv("../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv")
df


# In[ ]:


df.info()


# In[ ]:


df.columns


# # *********Get CATEGORICAL data which is represented by int64:*************

# sources: 
# * https://pbpython.com/categorical-encoding.html=
# 
# * https://www.shanelynn.ie/using-pandas-dataframe-creating-editing-viewing-data-in-python/
# 
# * https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd

# In[ ]:


cat_df = df.select_dtypes(include=['int64']).copy()
cat_df = cat_df.drop(columns="ID")#delete ID from categorical data -> not useful
cat_df.columns


# In[ ]:


cat_df.shape


# 1. ONE-HOT-ENCODE certain categorical data
# 
# 
# * replaces i column with multiple columns that will be "hot" when rows with a certain status
# * will not one-hot-encode AGE

# ONE-HOT-ENCODE: "SEX","MARRIAGE","EDUCATION",

# TOO MANY COLUMNS FOR "EDUCATION"
# 
# ->replace certain education statuses due to too many cols ->put all other options into 4 

# In[ ]:


cat_df['EDUCATION'].replace({0: 4, 5: 4, 6: 4}, inplace=True)


# In[ ]:


encode_columns=['SEX','MARRIAGE','EDUCATION']
for i in encode_columns:
    cat_df=pd.get_dummies(cat_df, columns=[i])


# In[ ]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
cat_df.columns


# ONE-HOT-ENCODE "PAY_i":

# In[ ]:


unique_status = np.unique(cat_df[['PAY_0']])
print("total unique statuses:", len(unique_status))
print(unique_status)


# * will get 10-11 new columns per PAY_i with one-hot-encoding because some monthes might have 0 frequency of a payment status/es

# In[ ]:


monthes=['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
for i in monthes:
    cat_df=pd.get_dummies(cat_df, columns=[i])


# BIN the AGE feature
# 
# 5 groups : 21-30 , 31-40 , 40-50 , 50-60 , 60-75
# 
# sources:
# 
# https://medium.com/vickdata/four-feature-types-and-how-to-transform-them-for-machine-learning-8693e1c24e80

# In[ ]:


bins = [21, 30, 40, 50, 60, 76]
group_names = ['21-30', '31-40', '41-50', '51-60', '61-76']
age_cats = pd.cut(cat_df['AGE'], bins, labels=group_names)
cat_df['age_cats'] = pd.cut(cat_df['AGE'], bins, labels=group_names)


# ONE-HOT-ENCODE the age categories :

# In[ ]:


cat_df=pd.get_dummies(cat_df, columns=['age_cats'])


# In[ ]:


cat_df.columns


# In[ ]:


len(cat_df.columns)


# In[ ]:


cat_df.dtypes


# In[ ]:


len(cat_df.columns)


# # *********Get NUMERICAL data which is represented by float64:************* 

# In[ ]:


num_df = df.select_dtypes(include=['float64']).copy()
num_df.columns


# 1. ADAPTIVE BINNING: the BILL_AMT cols
# 
# 
# * we use the data distribution itself to decide our bin ranges
# 
# * bill_amts will be put into quantiles
# 
# source: https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b

# In[ ]:


bills=['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4','BILL_AMT5','BILL_AMT6']
col_names=['Q_BILL_AMT1', 'Q_BILL_AMT2', 'Q_BILL_AMT3', 'Q_BILL_AMT4','Q_BILL_AMT5', 'Q_BILL_AMT6']
i=0#counter 

for col in bills:
    quantile_list = [0, 0.25, 0.5, 0.75, 1.0]
    quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']
    num_df[col_names[i]] = pd.qcut(num_df[col],q=quantile_list,labels=quantile_labels)
    i+=1
    
num_df.columns


# In[ ]:


num_df.head()


# 2. ADAPTIVE BINNING: the PAY_AMT cols AND LIMIT_BAL
# we use the data distribution itself to decide our bin ranges
# 
# PAY_AMT(s) and LIMIT_BAL will be put into quantiles
# 
# source: https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b

# In[ ]:


pays=['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5','PAY_AMT6','LIMIT_BAL']
col_names=['Q_PAY_AMT1', 'Q_PAY_AMT2', 'Q_PAY_AMT3','Q_PAY_AMT4','Q_PAY_AMT5','Q_PAY_AMT6','Q_LIMIT_BAL']
i=0#counter 

for col in pays:
    quantile_list = [0, 0.25, 0.5, 0.75, 1.0]
    quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']
    num_df[col_names[i]] = pd.qcut(num_df[col],q=quantile_list,labels=quantile_labels)
    i+=1
    
num_df.columns


# now the originally numerical columns are categorical columns 

# ONE-HOT-ENCODE the Q_PAY_AMTs , Q_BILL_AMTs, and Q_LIM_BAL

# In[ ]:


encode_columns=['Q_BILL_AMT1', 'Q_BILL_AMT2','Q_BILL_AMT3', 'Q_BILL_AMT4', 'Q_BILL_AMT5', 'Q_BILL_AMT6','Q_PAY_AMT1', 'Q_PAY_AMT2', 'Q_PAY_AMT3','Q_PAY_AMT4','Q_PAY_AMT5','Q_PAY_AMT6','Q_LIMIT_BAL']
for i in encode_columns:
    num_df=pd.get_dummies(num_df, columns=[i])


# In[ ]:


num_df.head()


# In[ ]:


num_df.columns


# In[ ]:


len(num_df.columns)


# NEW COLUMN: create column that indicates if tuple has payment status >1 in first month and last month
# 
# 
# * make loop to go thru each PAY_0
# * make col with 0 or 1 that will indicate if condtion in true 
# * add to num_df 
# 

# source:https://stackoverflow.com/questions/32984462/setting-1-or-0-to-new-pandas-column-conditionally

# In[ ]:


num_df['late_payer']=df['PAY_0'].apply(lambda x: 1 if x > 1 else 0)

num_df['late_payer'].head()


# NEW COLUMN: create column that indicates if tuple has payed more than BILL_AMT (meaning they have a negative balance)

# In[ ]:


bill_mons=['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
cols=['OVER_BILL_AMT1','OVER_BILL_AMT2','OVER_BILL_AMT3','OVER_BILL_AMT4','OVER_BILL_AMT5','OVER_BILL_AMT6']
i=0#counter

for mon in bill_mons:
    num_df[cols[i]]=df[mon].apply(lambda x: 1 if x < 0 else 0)
    i+=1
    
num_df['OVER_BILL_AMT1'].head()    


# CONCAT ALL DATAFRAMES MADE:

# In[ ]:


data = pd.concat([cat_df, num_df], axis=1)


# In[ ]:


target=data['default.payment.next.month']
data = data.drop(columns='default.payment.next.month')#delete target from dataframe


# In[ ]:


data.head()


# In[ ]:


len(data.columns)


# 152 columns in all: next step is creating model with this df 

# In[ ]:


#data.to_csv('mycsvfile.csv',index=False)

