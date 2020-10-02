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


# # Foreword:
# This is a really lazy but effective and easy-to-understand method to see how features affect targeted labels' distributions. It's also widely applicable to other competitions / real-life problems

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


train_identity = pd.read_csv('../input/train_identity.csv')
train_transaction = pd.read_csv('../input/train_transaction.csv')
test_identity = pd.read_csv('../input/test_identity.csv')
test_transaction = pd.read_csv('../input/test_transaction.csv')


# In[ ]:


train_identity.head()


# Add isFraud value to test data, value 2 means 'unknown'

# In[ ]:


test_transaction['isFraud'] = 2


# # EDA Transaction Dataset

# In[ ]:


transaction = pd.concat([train_transaction,test_transaction],axis=0)
print (transaction.shape)


# ## For Object features, I use barplot to see the distribution

# In[ ]:


def eda_object(df,feature):
    a = len(df[feature].unique())
    plt.figure(figsize = [20,min(max(8,a),12)])

    plt.subplot(1,3,1)
    x_ = df.groupby([feature])[feature].count()
    x_.plot(kind='pie')
    plt.title(feature)

    plt.subplot(1,3,2)
    cross_tab = pd.crosstab(df['isFraud'],df[feature],normalize=0).reset_index()
    x_ = cross_tab.melt(id_vars=['isFraud'])
    x_['value'] = x_['value']*100

    sns.barplot(x=feature,y='value',hue ='isFraud',data=x_,palette = ['b','r','g'],alpha =0.7)
    plt.xticks(rotation='vertical')
    plt.title(feature + " - Normalized by isFraud")

    plt.subplot(1,3,3)
    cross_tab = pd.crosstab(df[feature],df['isFraud'],normalize=1).reset_index()
    cross_tab['Difference'] = cross_tab[0] - cross_tab[1]
    cross_tab['Difference'] = cross_tab['Difference']*100

    sns.barplot(x=feature,y='Difference',data=cross_tab,alpha =0.7)
    plt.ylim(-100,100)
    plt.xticks(rotation='vertical')
    plt.title(feature + " - Difference")

    plt.tight_layout()
    plt.legend()
    plt.show()
       


# To save space, I just draw 10 available features

# In[ ]:


rm_list = ['TransactionID','TransactionDT','isFraud']
type_list = ['object']
feature_list = []

for feature in transaction.columns:
    if (feature not in rm_list) & (transaction[feature].dtypes in type_list):
        feature_list.append(feature)
for feature in feature_list[:10]:
    eda_object(transaction,feature)


# ## For Numeric features with more than 10 unique values, I use boxplot and density plot to see the distribution

# In[ ]:


def eda_numeric(df,feature):
    x_ = df[feature]+0.01
    y_ = df['isFraud']
    data = pd.concat([x_,y_],1)
    plt.figure(figsize=[20,5])

    ax1 = plt.subplot(1,2,1)
    sns.boxplot(x='isFraud',y=feature,data=data)
    plt.title(feature+ " - Boxplot")
    upper_0 = data[data['isFraud']==0][feature].quantile(q=0.80)
    upper_1 = data[data['isFraud']==1][feature].quantile(q=0.80)
    lower_0 = data[data['isFraud']==0][feature].quantile(q=0.20)
    lower_1 = data[data['isFraud']==1][feature].quantile(q=0.20)

    ax1.set(ylim=(min(lower_0,lower_1),max(upper_0,upper_1)))

    ax2 = plt.subplot(1,2,2)
    plt.title(feature+ " - Density with Log")

    p1=sns.kdeplot(data[data['isFraud']==0][feature].apply(np.log), color="b",legend=False)
    p2=sns.kdeplot(data[data['isFraud']==1][feature].apply(np.log), color="r",legend=False)
    p3=sns.kdeplot(data[data['isFraud']==2][feature].apply(np.log), color="g",shade = True,linestyle = '--',alpha =0.3,legend=False)
    plt.legend(loc='upper right', labels=['0', '1', '2'])

    plt.tight_layout()
    plt.show()


# To save space, I just draw 10 available features

# In[ ]:


rm_list = ['TransactionID','TransactionDT','isFraud']
type_list = ['int64','float']
feature_list = []

for feature in transaction.columns:
    if (feature not in rm_list) & (transaction[feature].dtypes in type_list) & (len(transaction[feature].unique()) > 10):

        feature_list.append(feature)
for feature in feature_list[:10]:
    eda_numeric(transaction,feature)


# # EDA Identity Dataset

# I will break down into 2 types: Distributions of identity by PEOPLE and Distribution of identity by TRANSACTION

# ### By PEOPLE

# In[ ]:


isFraud_trans_id_0 = transaction[transaction['isFraud'] == 0]['TransactionID']
isFraud_trans_id_1 = transaction[transaction['isFraud'] == 1]['TransactionID']
isFraud_trans_id_2 = transaction[transaction['isFraud'] == 2]['TransactionID']


# In[ ]:


identity = pd.concat([train_identity,test_identity],axis = 0)
identity['isFraud'] = np.where(identity.TransactionID.isin(isFraud_trans_id_0),0,
                               np.where(identity.TransactionID.isin(isFraud_trans_id_1),1,2))


# In[ ]:


rm_list = ['TransactionID','TransactionDT','isFraud']
type_list = ['object']
feature_list = []

for feature in identity.columns:
    if (feature not in rm_list) & (identity[feature].dtypes in type_list):
        feature_list.append(feature)
for feature in feature_list[:10]:
    eda_object(identity,feature)


# In[ ]:


rm_list = ['TransactionID','TransactionDT','isFraud']
type_list = ['int64','float']
feature_list = []

for feature in identity.columns:
    if (feature not in rm_list) & (identity[feature].dtypes in type_list) & (len(identity[feature].unique()) > 10):

        feature_list.append(feature)
for feature in feature_list[:10]:
    eda_numeric(identity,feature)


# ### BY TRANSACTION

# In[ ]:


transaction_identity = transaction[['TransactionID','isFraud']]
transaction_identity = transaction_identity.merge(identity,on='TransactionID',how='left')


# In[ ]:


rm_list = ['TransactionID','TransactionDT','isFraud']
type_list = ['object']
feature_list = []

for feature in identity.columns:
    if (feature not in rm_list) & (identity[feature].dtypes in type_list):
        feature_list.append(feature)
for feature in feature_list[:10]:
    eda_object(identity,feature)


# In[ ]:


rm_list = ['TransactionID','TransactionDT','isFraud']
type_list = ['int64','float']
feature_list = []

for feature in identity.columns:
    if (feature not in rm_list) & (identity[feature].dtypes in type_list) & (len(identity[feature].unique()) > 10):

        feature_list.append(feature)
for feature in feature_list[:10]:
    eda_numeric(identity,feature)


# It's kinda messy now, don't worry it helps us to have a general overview about the distributions according Fraud/not Fraud types, I will have a compact version soon
