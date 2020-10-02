#!/usr/bin/env python
# coding: utf-8

# ### Purpose : Predicting the probability that an online transaction is fraudulent
# ### Target : isFraud(binary target)**
# 
# The data is broken into two files "identity" and "transaction", which are joined by "TransactionID". 
# 
# Not all transactions have corresponding identity information.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
sns.set(font_scale=2)
print(os.listdir("../input"))


# # Load data

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')\ntest_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')\ntrain_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')\ntest_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')\nsubmission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')")


# To save memory, you can use reduce_mem_usage function. 

# In[ ]:


#reference : https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
# def reduce_mem_usage(props):
#     start_mem_usg = props.memory_usage().sum() / 1024**2 
#     print("Memory usage of properties dataframe is :",start_mem_usg," MB")
#     NAlist = [] # Keeps track of columns that have missing values filled in. 
#     for col in props.columns:
#         if props[col].dtype != object:  # Exclude strings
#             IsInt = False
#             mx = props[col].max()
#             mn = props[col].min()
#             if not np.isfinite(props[col]).all(): 
#                 NAlist.append(col)
#                 props[col].fillna(mn-1,inplace=True)                     
#             asint = props[col].fillna(0).astype(np.int64)
#             result = (props[col] - asint)
#             result = result.sum()
#             if result > -0.01 and result < 0.01:
#                 IsInt = True
#             if IsInt:
#                 if mn >= 0:
#                     if mx < 255:
#                         props[col] = props[col].astype(np.uint8)
#                     elif mx < 65535:
#                         props[col] = props[col].astype(np.uint16)
#                     elif mx < 4294967295:
#                         props[col] = props[col].astype(np.uint32)
#                     else:
#                         props[col] = props[col].astype(np.uint64)
#                 else:
#                     if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
#                         props[col] = props[col].astype(np.int8)
#                     elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
#                         props[col] = props[col].astype(np.int16)
#                     elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
#                         props[col] = props[col].astype(np.int32)
#                     elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
#                         props[col] = props[col].astype(np.int64) 
#             else:
#                 props[col] = props[col].astype(np.float32)            

#     print("___MEMORY USAGE AFTER COMPLETION:___")
#     mem_usg = props.memory_usage().sum() / 1024**2 
#     print("Memory usage is: ",mem_usg," MB")
#     print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
#     return props, NAlist


# # Exploratoy Data Analysis

# In[ ]:


print(train_transaction.shape, test_transaction.shape)
print(train_identity.shape, test_identity.shape)


# The results show that the number of columns in the train_identity and test_identity is the same.

# In[ ]:


train_identity.head()


# In[ ]:


train_transaction.head()


# In[ ]:


train_transaction.describe()


# - imbalanced data set (more than 75% of isFraud is filled with 0).
# - a lot of NaN values.
# 

# # Examine the Distribution of the Target Colum

# In[ ]:


print(round(train_transaction['isFraud'].value_counts(normalize=True) * 100,2))
train_transaction['isFraud'].astype(int).plot.hist();


# From this information, we see this is an imbalanced class problem (Only have 3.5% of positive values). So, we can weight the classes by their representation in the data to reflect this imbalance.

# # Examine Missing Values

# In[ ]:


# Reference : https://www.kaggle.com/willkoehrsen/start-here-a-gentle-introduction
def missing_values_table(df):    
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    return mis_val_table_ren_columns


# In[ ]:


missing_values_iden = missing_values_table(train_identity)
missing_values_iden.head(10)


# In[ ]:


missing_values_trans = missing_values_table(train_transaction)
missing_values_trans.head(10)


# There are many columns with over 90% missing values.
# It is important to deal with missing values.

# # Column Types

# In[ ]:


print('\n', 'Number of each type of column')
print(train_identity.dtypes.value_counts())
print('\n', 'Number of unique classes in each object column')
print(train_identity.select_dtypes('object').apply(pd.Series.nunique, axis = 0).sort_values(ascending = False))


# Some categorical variables have a relatively large number of unique entries. 
# We will need to find a way to deal with these categorical variables.
# Because of the large number of columns, I think it would be better to do frequency encoding or mean encoding than one-hot encoding.

# # features - transaction
# 
# * emaildomain
# * card1 - card6
# * addr1, addr2
# * P_emaildomain
# * R_emaildomain
# * M1 - M9
# The TransactionDT feature is a timedelta from a given reference datetime (not an actual timestamp).

# In[ ]:


print('\n', 'Number of each type of column')
print(train_transaction.dtypes.value_counts())
print('\n', 'Number of unique classes in each int column')
print(train_transaction.select_dtypes('int').apply(pd.Series.nunique, axis = 0).sort_values(ascending = False))
print('\n', 'Number of unique classes in each object column')
print(train_transaction.select_dtypes('object').apply(pd.Series.nunique, axis = 0).sort_values(ascending = False))


# In[ ]:


#Visualize except float type features.
def bar_plot(col, data, hue=None):
    f, ax = plt.subplots(figsize = (30, 5))
    sns.countplot(x=col, hue=hue, data=data, alpha=0.5)

#Visualize float type features.
def dist_plot(col, data):
    f, ax = plt.subplots(figsize = (30, 5))
    sns.distplot(data[col].dropna(), kde=False, bins=10)


# ### emaildomain

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(30,15))

sns.countplot(y="P_emaildomain", ax=ax[0], data=train_transaction)
ax[0].set_title('P_emaildomain')
sns.countplot(y="R_emaildomain", ax=ax[1], data=train_transaction)
ax[1].set_title('R_emaildomain')


# ### Product

# In[ ]:


bar_plot('ProductCD', train_transaction, hue='isFraud')


# ### card1 - card6

# In[ ]:


card_float = ['card1', 'card2', 'card3', 'card5']
for col in card_float:
    print(col, train_transaction[col].nunique())


# In[ ]:


card_n_float = ['card4', 'card6']    

for col in card_n_float:
    bar_plot(col, train_transaction, hue='isFraud')


# ### addr1, addr2
# 

# In[ ]:


card_float = ['addr1', 'addr2']
for col in card_float:
    dist_plot(col, train_transaction)


# ### C1 - C14

# In[ ]:


C_col = [c for c in train_transaction if c[0] == 'C']
corr = train_transaction[C_col].corr()

cmap = sns.color_palette("Blues")
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, cmap=cmap)


# There are many variables with a correlation of 1. Therefore, it is necessary to process the correlated variables.

# ### D1 - D9

# In[ ]:


D_col = [c for c in train_transaction if c[0] == 'D']
corr = train_transaction[D_col].corr()

cmap = sns.color_palette("Blues")
f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(corr, cmap=cmap)


# - D1, D2
# - D4, D12, D6
# - D5, D7

# ### M1 - M9

# In[ ]:


M_col = [c for c in train_transaction if c[0] == 'M']
for col in M_col:
    bar_plot(col, train_transaction, hue='isFraud')


# ### V1 - V339

# In[ ]:


V_col = [c for c in train_transaction if c[0] == 'V']
train_transaction[V_col].describe()


# Most of the columns are filled with zero

# ### TransactionDT

# The following hist shows that train_transaction and test_transaction were split by TransactionDT. So it would be prudent to use time-based split for validation.

# In[ ]:


train_transaction['TransactionDT'].plot(kind='hist', figsize=(15, 5), label='train_transaction', bins=100)
test_transaction['TransactionDT'].plot(kind='hist', label='test_transaction', bins=100)
plt.legend()
plt.show()


# ### TransactionAMT

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(18,4))

time_val = train_transaction['TransactionAmt'].values

sns.distplot(train_transaction['TransactionAmt'].values, ax=ax[0], color='r')
ax[0].set_title('Distribution of TransactionAmt', fontsize=14)
ax[0].set_xlim([min(train_transaction['TransactionAmt'].values), max(train_transaction['TransactionAmt'].values)])

sns.distplot(np.log(train_transaction['TransactionAmt'].values), ax=ax[1], color='b')
ax[1].set_title('Distribution of LOG TransactionAmt', fontsize=14)
ax[1].set_xlim([min(np.log(train_transaction['TransactionAmt'].values)), max(np.log(train_transaction['TransactionAmt'].values))])

plt.show()


# ## Categorical features - identity
# 
# * DeviceType
# * DeviceInfo
# * id_12 - id_38
# 

# In[ ]:


print('\n', 'Number of each type of column')
print(train_identity.dtypes.value_counts())
print('\n', 'Number of unique classes in each object column')
print(train_identity.select_dtypes('object').apply(pd.Series.nunique, axis = 0).sort_values(ascending = False))


# To be continue...
