#!/usr/bin/env python
# coding: utf-8

# # Hi,
# 
# ## We start this Kernel with the basic steps towards the Data Science technology of Exploring the Data Set, and going onto wrangle around with it and build a model as required for the Competition.
# ## This is my First ever Kaggle Competition excluding the Titanic Competition. Your support,suggestions and upvotes will get me closer to the right approach to be used.
# 
# ### Let's Start!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as pl
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_identity = pd.read_csv("../input/train_identity.csv")
train_transaction = pd.read_csv("../input/train_transaction.csv")
test_identity = pd.read_csv("../input/test_identity.csv")
test_transaction = pd.read_csv("../input/test_transaction.csv")


# In[ ]:


train_transaction.head()


# In[ ]:


print(train_transaction.info())


# # Let's Try our Data Cleaning!

# In[ ]:


train_transaction.isnull().sum()


# In[ ]:


#Let's merge our data sets for Future!
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')


# In[ ]:


clean_df = train_transaction.dropna(axis=1)
clean_df.head()


# In[ ]:


print(clean_df.shape)
print(train_transaction.shape)


# ## So we see that the number of rows are intact, whereas the attributes are filtered on their own with null values.
# ## Next we can do that is check the missing values again, and try finding the balance in our data set!

# In[ ]:


clean_df.isnull().sum()


# In[ ]:


# Wow, the data looks good to go!
# let's check "isFraud" data balance in respect to our target
target_balancing = clean_df['isFraud'].value_counts().values
sns.barplot([0,1],target_balancing);
pl.title("How much is our target balanced??");


# In[ ]:


#Let's try seeing the Fraud and non fraud transactions based on Products
sns.barplot(x = clean_df.index,y = 'ProductCD',hue='isFraud',data = clean_df);


# ### We have highest number of Fraud Transactions for the Product **S** and **C**.
# ## Next we check for outliers in our data frame!

# In[ ]:


pl.figure(figsize=(16,7))
sns.boxplot(data = clean_df.drop(columns = ['TransactionID','isFraud']))
pl.yscale('log')
pl.xticks(rotation=90);


# In[ ]:


pl.hist(train['TransactionDT'], label='train');
pl.hist(test['TransactionDT'], label='test');
pl.legend();
pl.title('Distribution of transaction dates');


# In[ ]:


train.head()


# In[ ]:


clean_df.head()


# ## Next is to deal with the outliers and see the correlation.
# ## Moving on them we will try to build a model to get the scores of prediction and accuracy.

# In[ ]:


pl.figure(figsize=(16,7))
cor  =clean_df.corr()
sns.heatmap(cor,annot=True);


# ## The top 5 pair of attributes with High Correlation! for the whole data set are:
# * C8 and C6
# * C1 and C6
# * C1 and C8
# * C8 and C7
# * C6 and C11
# 
# ## Let's now split the data frame based on Fraudlent and Non Fraudlent and then check for their correlation.
# ### This gives us as how many of those attributes still hold true!

# In[ ]:


fraudlent = clean_df[clean_df['isFraud']==1]
non_fraudlent = clean_df[clean_df['isFraud']==0]
fraudlent.head()


# In[ ]:


#Let's see the Transaction AMount for the Fraudlent and Non Fraudlent Transactions!
pl.figure(figsize=(16,7))
pl.subplot(1,2,1)
sns.distplot(fraudlent['TransactionAmt'],rug=True);
pl.title('Transaction Amount for Fraudlent Transactions!');
pl.subplot(1,2,2)
sns.distplot(non_fraudlent['TransactionAmt'],rug=True);
pl.title('Transaction Amount for Non Fraudlent Transactions!');


# ## So we can say from the graph that *fraudlent transactions can be brought in light if the transaction is within the limit of 5000* whereas the *non fraudlents can be considered if the transactions are reasonably above 5.5k*
#  
# ### Let's chec the mean value to support our hypothesis!

# In[ ]:


mean_fraud = fraudlent['TransactionAmt'].mean()
mean_non_fraud = non_fraudlent['TransactionAmt'].mean()
print(mean_fraud)
print(mean_non_fraud)


# ### So we see that on Average a *Fradlent ID can perform 149 Rs. Transaction* but a *Non Fraudlent perform approx 135Rs. transaction!*
# 
# ## We can use this figure to compare on respective Attribute!

# In[ ]:


# Correlation for each
cor_fraud = fraudlent.corr()
cor_non_fraud = non_fraudlent.corr()
pl.figure(figsize=(20,10))
pl.subplot(2,1,1)
sns.heatmap(cor_fraud,annot=True)
pl.subplot(2,1,2)
sns.heatmap(cor_non_fraud,annot=True);


# In[ ]:


# We will go on to find the Top 5 Attributes that are highly Correlated fro each of the Category!
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations for Fraudlent Transactions!")
print(get_top_abs_correlations(fraudlent.drop(columns = ['ProductCD','isFraud']), 5))

# I love this peice of code as it makes the Corelation much easier to look!


# In[ ]:


# We will go on to find the Top 5 Attributes that are highly Correlated fro each of the Category!
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=5):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

print("Top Absolute Correlations for Non Fraudlent Transactions!")
print(get_top_abs_correlations(non_fraudlent.drop(columns = ['ProductCD','isFraud']), 5))


# In[ ]:


train_identity.head()


# In[ ]:


# Let's get rid of Null values on AXIS = 1
train_identity_clean_df  = train_identity.dropna(axis=0).reset_index(drop=True)
train_identity_clean_df.head()


# In[ ]:


train_identity_clean_df.isnull().sum()


# In[ ]:


# Well I'm thinking of mergign the cleaned Transaction and Identity dataframes as this gives us better view
merged_train = pd.merge(clean_df,train_identity_clean_df,on = 'TransactionID',how='left')
merged_train.head()


# In[ ]:


merged_train = merged_train.dropna(axis = 0).reset_index(drop=True)
merged_train.head()


# In[ ]:


merged_train.shape


# ## Now we see that the sahpe is reduced and has been reduced drastically when we tried to merge based on the Valid ID's for which in both the dataframes, Entry was present.
# ## We can still fill out the missing values, but for 60 rows, it will be explicitally hard coding.
# ### Do let me know if there is any better way to keep the shape intact and be free of Null values or suggest me a better approach!

# In[ ]:


balancing_merge_df = merged_train['isFraud'].value_counts().values
sns.barplot([0,1],balancing_merge_df);
pl.title('Checking balance of our target in the merged Data Set!');


# In[ ]:


merged_train['TransactionAmt'].describe()


# In[ ]:


pl.figure(figsize=(16,7))
pl.subplot(1,2,1)
pl.plot('TransactionAmt','ProductCD', data=merged_train, marker='o', alpha=0.4)
pl.subplot(1,2,2)
pl.plot('TransactionAmt','ProductCD', data=merged_train, linestyle='none', marker='o', color="orange", alpha=0.3);
pl.title('Different types of Products on Transaction Amount!');


# ## Next we can go onto create metrics for the Group of Amounts and based on that, we an find the Fraudlent and Non Fraudlent in that Amount Group!
# 
# ## Till then think and contribute some methods/explorations and help me in making this kernel better!

# In[ ]:


sns.set_style("white")
sns.kdeplot(merged_train.TransactionAmt, merged_train.card1);


# In[ ]:


# top correlating Attributes for the merged Data Frame!
pl.figure(figsize=(20,10))
corr_merge = merged_train.corr()
sns.heatmap(corr_merge,annot=True)
print("Top Absolute Correlations for Merged Data frame!!")
numeric_df_merged = merged_train.select_dtypes(include=['int'])
print(get_top_abs_correlations(numeric_df_merged, 5))

