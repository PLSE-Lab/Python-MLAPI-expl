#!/usr/bin/env python
# coding: utf-8

# This is my first competition so I am pretty excited. All the best to everyone reading my kernel. Hope you get to learn something about this dataset through my kernel. Please upvote if you like the kernel. 

# **Importing basic libraries and modules**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.plotly as py
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
seed=5
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import gc
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Loading the data**

# In[ ]:


train_identity=pd.read_csv('../input/train_identity.csv',index_col='TransactionID')
test_identity=pd.read_csv('../input/test_identity.csv',index_col='TransactionID')
train_transaction=pd.read_csv('../input/train_transaction.csv',index_col='TransactionID')
test_transaction=pd.read_csv('../input/test_transaction.csv',index_col='TransactionID')
sub=pd.read_csv('../input/sample_submission.csv',index_col='TransactionID')


# In[ ]:


print('Shape of train identity :',train_identity.shape)
print('Shape of test identity :',test_identity.shape)
print('Shape of train transaction :',train_transaction.shape)
print('Shape of test transaction :',test_transaction.shape)


# In[ ]:


train_identity.head()


# In[ ]:


train_transaction.head()


# In[ ]:


train_df=pd.merge(train_identity,train_transaction,how='right',on='TransactionID')
test_df=pd.merge(test_identity,test_transaction,how='right',on='TransactionID')


# **Important step not go out of RAM**

# In[ ]:


del train_identity,test_identity,train_transaction,test_transaction


# In[ ]:


train_df.info()


# **Downcast the dataframes to free up upto 1 GB RAM. **

# In[ ]:


def downcast(df):
    float_cols=[col for col in df.columns if df[col].dtype=='float']
    df[float_cols]=df[float_cols].astype('float32')
    int_cols=[col for col in df.columns if df[col].dtype=='int']
    df[int_cols]=df[int_cols].astype('int16')
    return df    


# In[ ]:


train_df=downcast(train_df)
test_df=downcast(test_df)


# **Save these downcasted dataframes becausing downcasting was time consuming**

# In[ ]:


train_df.to_csv('IEEE_train.csv')
test_df.to_csv('IEEE_test.csv')


# In[ ]:


print(train_df.shape)
print(test_df.shape)


# In[ ]:


train_df.info()


# In[ ]:


def null_values(df):
    series_1=pd.Series(df.isnull().sum(),name='Total null values')
    series_2=pd.Series(series_1*100/df.shape[0],name='Percentage of null values')
    null_df=pd.concat([series_1,series_2],axis=1).sort_values(by='Percentage of null values',ascending=False)
    return null_df


# In[ ]:


null_df=null_values(train_df)


# In[ ]:


null_df.head()


# **Fill missing values of numerical features with their mean and that of categorical feature with their mode.**

# In[ ]:


def fill_missing(df):
    num_cols=[col for col in df.columns if df[col].dtype=='float32' or df[col].dtype=='int16']
    for col in num_cols:
        df[col]=df[col].fillna(df[col].mean())
    obj_cols=[col for col in df.columns if df[col].dtype=='object']
    for col in obj_cols:
        df[col]=df[col].fillna(df[col].mode()[0])
        
    return df


# In[ ]:


train_df=fill_missing(train_df)
test_df=fill_missing(test_df)


# **Generate some  new features**

# In[ ]:


num_cols=[col for col in train_df.columns if train_df[col].dtype=='int16' or train_df[col].dtype=='float32']
num_cols.remove('isFraud')
train_df['mean']=train_df[num_cols].mean(axis=1)
test_df['mean']=test_df[num_cols].mean(axis=1)
train_df['max']=train_df[num_cols].max(axis=1)
test_df['max']=test_df[num_cols].max(axis=1)
train_df['min']=train_df[num_cols].min(axis=1)
test_df['min']=test_df[num_cols].min(axis=1)
train_df['median']=train_df[num_cols].median(axis=1)
test_df['median']=test_df[num_cols].median(axis=1)
train_df['skew']=train_df[num_cols].skew(axis=1)
test_df['skew']=test_df[num_cols].skew(axis=1)
train_df['kurt']=train_df[num_cols].kurt(axis=1)
test_df['kurt']=test_df[num_cols].kurt(axis=1)


# **Visualizing all numerical features for Train and Test Data **

# In[ ]:


#blue represents train while orange represents test
#I had tried setting labels but the legends of adjacent plots used to overlap.
fig,ax=plt.subplots(10,10,figsize=(16,16))
plt.tight_layout()
for i in range(10):
    for j in range(10):
        sns.distplot(train_df[num_cols[i*10+j]],ax=ax[i,j],hist=False)
        sns.distplot(test_df[num_cols[i*10+j]],ax=ax[i,j],hist=False)
        ax[i,j].set_xlabel(num_cols[i*10+j])
        plt.legend()
        

plt.show()


# Most of the train and test feature distributions look similar. Some distributions like that of TransactionDT, card1,card2,addr1,addr2,etc. almost exactly overlap, that's a bit strange I guess. Some distributions like that of C9,id13,id21(peaks at different value) are significantly different.

# In[ ]:


fig,ax=plt.subplots(10,10,figsize=(16,16))
plt.tight_layout()
for i in range(10):
    for j in range(10):
        sns.distplot(train_df[num_cols[i*10+j+100]],ax=ax[i,j],hist=False)
        sns.distplot(test_df[num_cols[i*10+j+100]],ax=ax[i,j],hist=False)
        ax[i,j].set_xlabel(num_cols[i*10+j+100])
        plt.legend()
        

plt.show()


# Some distributions like that of V77,V78,V86,V87 are seemingly different. Here again, many look similar, some are just shifted versions while some are completely different. We do similar analysis for the rest of the numerical features.

# In[ ]:


fig,ax=plt.subplots(10,10,figsize=(16,16))
plt.tight_layout()
for i in range(10):
    for j in range(10):
        sns.distplot(train_df[num_cols[i*10+j+200]],ax=ax[i,j],hist=False)
        sns.distplot(test_df[num_cols[i*10+j+200]],ax=ax[i,j],hist=False)
        ax[i,j].set_xlabel(num_cols[i*10+j+200])
        plt.legend()
        

plt.show()


# In[ ]:


fig,ax=plt.subplots(10,10,figsize=(16,16))
plt.tight_layout()
for i in range(10):
    for j in range(10):
        sns.distplot(train_df[num_cols[i*10+j+300]],ax=ax[i,j],hist=False)
        sns.distplot(test_df[num_cols[i*10+j+300]],ax=ax[i,j],hist=False)
        ax[i,j].set_xlabel(num_cols[i*10+j+300])
        plt.legend()
        

plt.show()


# In[ ]:


sns.distplot(train_df['V339'],label='train',hist=False)
sns.distplot(test_df['V339'],label='test',hist=False)
plt.legend()


# **Visualizing all categorical features of Train and Test set**

# In[ ]:


obj_cols=[col for col in train_df.columns if train_df[col].dtype=='object']

fig,ax=plt.subplots(3,2,figsize=(10,10))

for i in range(3):
    j=0
    sns.countplot(train_df[obj_cols[i*2+j]],ax=ax[i,j])
    sns.countplot(test_df[obj_cols[i*2+j]],ax=ax[i,j+1])
    ax[i,j].set_title('Train')
    ax[i,j+1].set_title('Test')
    ax[i,j].set_xlabel(obj_cols[i*2+j])
    ax[i,j+1].set_xlabel(obj_cols[i*2+j])
    
plt.tight_layout()    
plt.show()


# All the distributions are similar in train and test

# In[ ]:


fig,ax=plt.subplots(3,2,figsize=(10,10))

for i in range(3):
    j=0
    sns.countplot(train_df[obj_cols[i*2+j+6]],ax=ax[i,j])
    sns.countplot(test_df[obj_cols[i*2+j+6]],ax=ax[i,j+1])
    ax[i,j].set_title('Train')
    ax[i,j+1].set_title('Test')
    ax[i,j].set_xlabel(obj_cols[i*2+j+6])
    ax[i,j+1].set_xlabel(obj_cols[i*2+j+6])
    
plt.tight_layout()    
plt.show()


# id_31 has too many unique values. id_34 has more unique values in train than in test. We do similar analysis for rest of the categorical features.

# In[ ]:


fig,ax=plt.subplots(3,2,figsize=(10,10))

for i in range(3):
    j=0
    sns.countplot(train_df[obj_cols[i*2+j+12]],ax=ax[i,j])
    sns.countplot(test_df[obj_cols[i*2+j+12]],ax=ax[i,j+1])
    ax[i,j].set_title('Train')
    ax[i,j+1].set_title('Test')
    ax[i,j].set_xlabel(obj_cols[i*2+j+12])
    ax[i,j+1].set_xlabel(obj_cols[i*2+j+12])
    
plt.tight_layout()    
plt.show()


# In[ ]:


fig,ax=plt.subplots(3,2,figsize=(10,10))

for i in range(3):
    j=0
    sns.countplot(train_df[obj_cols[i*2+j+18]],ax=ax[i,j])
    sns.countplot(test_df[obj_cols[i*2+j+18]],ax=ax[i,j+1])
    ax[i,j].set_title('Train')
    ax[i,j+1].set_title('Test')
    ax[i,j].set_xlabel(obj_cols[i*2+j+18])
    ax[i,j+1].set_xlabel(obj_cols[i*2+j+18])
    
plt.tight_layout()    
plt.show()


# In[ ]:


fig,ax=plt.subplots(3,2,figsize=(10,10))

for i in range(3):
    j=0
    sns.countplot(train_df[obj_cols[i*2+j+24]],ax=ax[i,j])
    sns.countplot(test_df[obj_cols[i*2+j+24]],ax=ax[i,j+1])
    ax[i,j].set_title('Train')
    ax[i,j+1].set_title('Test')
    ax[i,j].set_xlabel(obj_cols[i*2+j+24])
    ax[i,j+1].set_xlabel(obj_cols[i*2+j+24])
    
plt.tight_layout()    
plt.show()


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(10,10))

sns.countplot(train_df[obj_cols[30]],ax=ax[0])
sns.countplot(test_df[obj_cols[30]],ax=ax[1])
ax[0].set_title('Train')
ax[1].set_title('Test')
ax[0].set_xlabel(obj_cols[30])
ax[1].set_xlabel(obj_cols[30])
    
plt.tight_layout()    
plt.show()


# **Visualizing all numerical features for Fraudulent and Non-Fraudulent transactions.**

# In[ ]:


#Blue represents Non Fraud while orange represents Fraud
fig,ax=plt.subplots(10,10,figsize=(16,16))
plt.tight_layout()
for i in range(10):
    for j in range(10):
        sns.distplot(train_df[train_df['isFraud']==0][num_cols[i*10+j]],ax=ax[i,j],hist=False)
        sns.distplot(train_df[train_df['isFraud']==1][num_cols[i*10+j]],ax=ax[i,j],hist=False)
        ax[i,j].set_xlabel(num_cols[i*10+j])
        
        

plt.show()


# Here again, we try to find features which have similar and features which have different distributions in case of Fraudulent and Non-Fraudulent data. In many cases, Fraudulent data(orange) features have sharply peaked distributions. 

# In[ ]:


fig,ax=plt.subplots(10,10,figsize=(16,16))
plt.tight_layout()
for i in range(10):
    for j in range(10):
        sns.distplot(train_df[train_df['isFraud']==0][num_cols[i*10+j+100]],ax=ax[i,j],hist=False)
        sns.distplot(train_df[train_df['isFraud']==1][num_cols[i*10+j+100]],ax=ax[i,j],hist=False)
        ax[i,j].set_xlabel(num_cols[i*10+j+100])
        plt.legend()
        

plt.show()


# In[ ]:


fig,ax=plt.subplots(10,10,figsize=(16,16))
plt.tight_layout()
for i in range(10):
    for j in range(10):
        sns.distplot(train_df[train_df['isFraud']==0][num_cols[i*10+j+200]],ax=ax[i,j],hist=False)
        sns.distplot(train_df[train_df['isFraud']==1][num_cols[i*10+j+200]],ax=ax[i,j],hist=False)
        ax[i,j].set_xlabel(num_cols[i*10+j+200])
        plt.legend()
        

plt.show()


# In[ ]:


fig,ax=plt.subplots(10,10,figsize=(16,16))
plt.tight_layout()
for i in range(10):
    for j in range(10):
        sns.distplot(train_df[train_df['isFraud']==0][num_cols[i*10+j+300]],ax=ax[i,j],hist=False)
        sns.distplot(train_df[train_df['isFraud']==1][num_cols[i*10+j+300]],ax=ax[i,j],hist=False)
        ax[i,j].set_xlabel(num_cols[i*10+j+300])
        plt.legend()
        

plt.show()


# In[ ]:


sns.distplot(train_df[train_df['isFraud']==0]['V339'],hist=False,label='Non Fraud')
sns.distplot(train_df[train_df['isFraud']==1]['V339'],hist=False,label='Fraud')
plt.legend()
plt.xlabel('V339')
plt.show()


# **Visualizing all categorical features for Fraud and Non-Fraudulent transactions.**

# In[ ]:


fig,ax=plt.subplots(3,2,figsize=(10,10))

for i in range(3):
    j=0
    sns.countplot(train_df[train_df['isFraud']==0][obj_cols[i*2+j]],ax=ax[i,j])
    sns.countplot(train_df[train_df['isFraud']==1][obj_cols[i*2+j]],ax=ax[i,j+1])
    ax[i,j].set_title('Non_Fraud')
    ax[i,j+1].set_title('Fraud')
    ax[i,j].set_xlabel(obj_cols[i*2+j])
    ax[i,j+1].set_xlabel(obj_cols[i*2+j])
    
plt.tight_layout()    
plt.show()


# Distribution of unique values of id_16 is different in Fraudulent and Non-Fraudulent transactions. id 27 has only 1 unique in case of Fraudulent transactions. We do similar analysis for further categorical features.

# In[ ]:


fig,ax=plt.subplots(3,2,figsize=(10,10))

for i in range(3):
    j=0
    sns.countplot(train_df[train_df['isFraud']==0][obj_cols[i*2+j+6]],ax=ax[i,j])
    sns.countplot(train_df[train_df['isFraud']==1][obj_cols[i*2+j+6]],ax=ax[i,j+1])
    ax[i,j].set_title('Non_Fraud')
    ax[i,j+1].set_title('Fraud')
    ax[i,j].set_xlabel(obj_cols[i*2+j+6])
    ax[i,j+1].set_xlabel(obj_cols[i*2+j+6])
    
plt.tight_layout()    
plt.show()


# In[ ]:


fig,ax=plt.subplots(3,2,figsize=(10,10))

for i in range(3):
    j=0
    sns.countplot(train_df[train_df['isFraud']==0][obj_cols[i*2+j+12]],ax=ax[i,j])
    sns.countplot(train_df[train_df['isFraud']==1][obj_cols[i*2+j+12]],ax=ax[i,j+1])
    ax[i,j].set_title('Non_Fraud')
    ax[i,j+1].set_title('Fraud')
    ax[i,j].set_xlabel(obj_cols[i*2+j+12])
    ax[i,j+1].set_xlabel(obj_cols[i*2+j+12])
    
plt.tight_layout()    
plt.show()


# In[ ]:


fig,ax=plt.subplots(3,2,figsize=(10,10))

for i in range(3):
    j=0
    sns.countplot(train_df[train_df['isFraud']==0][obj_cols[i*2+j+18]],ax=ax[i,j])
    sns.countplot(train_df[train_df['isFraud']==1][obj_cols[i*2+j+18]],ax=ax[i,j+1])
    ax[i,j].set_title('Non_Fraud')
    ax[i,j+1].set_title('Fraud')
    ax[i,j].set_xlabel(obj_cols[i*2+j+18])
    ax[i,j+1].set_xlabel(obj_cols[i*2+j+18])
    
plt.tight_layout()    
plt.show()


# In[ ]:


fig,ax=plt.subplots(3,2,figsize=(10,10))

for i in range(3):
    j=0
    sns.countplot(train_df[train_df['isFraud']==0][obj_cols[i*2+j+24]],ax=ax[i,j])
    sns.countplot(train_df[train_df['isFraud']==1][obj_cols[i*2+j+24]],ax=ax[i,j+1])
    ax[i,j].set_title('Non_Fraud')
    ax[i,j+1].set_title('Fraud')
    ax[i,j].set_xlabel(obj_cols[i*2+j+24])
    ax[i,j+1].set_xlabel(obj_cols[i*2+j+24])
    
plt.tight_layout()    
plt.show()


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(10,10))

sns.countplot(train_df[train_df['isFraud']==0][obj_cols[30]],ax=ax[0])
sns.countplot(train_df[train_df['isFraud']==1][obj_cols[30]],ax=ax[1])
ax[0].set_title('Non_Fraud')
ax[1].set_title('Fraud')
ax[0].set_xlabel(obj_cols[30])
ax[1].set_xlabel(obj_cols[30])
    
plt.tight_layout()    
plt.show()


# **Visualization of new engineered features**

# In[ ]:


fig,ax=plt.subplots(2,1,figsize=(10,10))
sns.distplot(train_df['mean'],ax=ax[0],label='train')
sns.distplot(test_df['mean'],ax=ax[0],label='test')
ax[0].set_title('Train and Test Mean Column')
plt.legend()


sns.distplot(train_df[train_df['isFraud']==0]['mean'],ax=ax[1],label='Non Fraud')
sns.distplot(train_df[train_df['isFraud']==1]['mean'],ax=ax[1],label='Fraud')
ax[1].set_title('Fraud and Non Fraud Mean Column')
plt.legend()

plt.tight_layout()
plt.show()


# Similar distributions for train-test as well as for fraud-nonfraud combo.

# In[ ]:


fig,ax=plt.subplots(2,1,figsize=(10,10))
sns.distplot(train_df['max'],ax=ax[0],label='train')
sns.distplot(test_df['max'],ax=ax[0],label='test')
ax[0].set_title('Train and Test Max Column')
plt.legend()

sns.distplot(train_df[train_df['isFraud']==0]['max'],ax=ax[1],label='Non Fraud')
sns.distplot(train_df[train_df['isFraud']==1]['max'],ax=ax[1],label='Fraud')
ax[1].set_title('Fraud and Non Fraud Max Column')
plt.legend()

plt.tight_layout()
plt.show()


# Similar .

# In[ ]:


fig,ax=plt.subplots(2,1,figsize=(10,10))
sns.distplot(train_df['min'],ax=ax[0],label='train')
sns.distplot(test_df['min'],ax=ax[0],label='test')
ax[0].set_title('Train and Test Min Column')
plt.legend()

sns.distplot(train_df[train_df['isFraud']==0]['min'],ax=ax[1],label='Non Fraud')
sns.distplot(train_df[train_df['isFraud']==1]['min'],ax=ax[1],label='Fraud')
ax[1].set_title('Fraud and Non Fraud Min Column')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:


fig,ax=plt.subplots(2,1,figsize=(10,10))
sns.distplot(train_df['median'],ax=ax[0],label='train')
sns.distplot(test_df['median'],ax=ax[0],label='test')
ax[0].set_title('Train and Test Median Column')
plt.legend()

sns.distplot(train_df[train_df['isFraud']==0]['median'],ax=ax[1],label='Non Fraud')
sns.distplot(train_df[train_df['isFraud']==1]['median'],ax=ax[1],label='Fraud')
ax[1].set_title('Fraud and Non Fraud Median Column')
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:


fig,ax=plt.subplots(2,1,figsize=(10,10))
sns.distplot(train_df['skew'],ax=ax[0],label='train')
sns.distplot(test_df['skew'],ax=ax[0],label='test')
ax[0].set_title('Train and Test Skew Column')
plt.legend()

sns.distplot(train_df[train_df['isFraud']==0]['skew'],ax=ax[1],label='Non Fraud')
sns.distplot(train_df[train_df['isFraud']==1]['skew'],ax=ax[1],label='Fraud')
ax[1].set_title('Fraud and Non Fraud Skew Column')
plt.legend()

plt.tight_layout()
plt.show()


# Train and test skewness is a bit shifted. 

# In[ ]:


fig,ax=plt.subplots(2,1,figsize=(10,10))
sns.distplot(train_df['kurt'],ax=ax[0],label='train')
sns.distplot(test_df['kurt'],ax=ax[0],label='test')
ax[0].set_title('Train and Test Kurt Column')
plt.legend()

sns.distplot(train_df[train_df['isFraud']==0]['kurt'],ax=ax[1],label='Non Fraud')
sns.distplot(train_df[train_df['isFraud']==1]['kurt'],ax=ax[1],label='Fraud')
ax[1].set_title('Fraud and Non Fraud Kurt Column')
plt.legend()

plt.tight_layout()
plt.show()


# The kurtosis is shifted as well.
