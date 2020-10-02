#!/usr/bin/env python
# coding: utf-8

# # Overview
# The relationship between several sample and feature can be seen constant multiplication.  
# In this kernel we will look at some of those data.  

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt
import seaborn as sns
from tqdm import tqdm_notebook as tqdm

import os
print(os.listdir("../input"))

pd.set_option('display.max_columns', 500)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def load_data(path, num_row=None):
    df = pd.read_csv(path, nrows=num_row)
    return df


# In[ ]:


is_debug = False
num_row = 10000 if is_debug else None

train_path = "../input/train_identity.csv"
train_transaction_path = "../input/train_transaction.csv"
test_path = "../input/test_identity.csv"
test_transaction_path = "../input/test_transaction.csv"


# In[ ]:


train = load_data(train_path)
train_transaction = load_data(train_transaction_path)
test = load_data(test_path)
test_transaction = load_data(test_transaction_path)

print(train.shape)
print(train_transaction.shape)
print(test.shape)
print(test_transaction.shape)


# In[ ]:


adjusted_train = pd.merge(train_transaction, train, on="TransactionID", how='left')
adjusted_test = pd.merge(test_transaction, test, on="TransactionID", how='left')

print("adjusted_train: {}".format(adjusted_train.shape))
print("adjusted_test: {}".format(adjusted_test.shape))


# In[ ]:


adjusted_train.head()


# # Extract number

# Some anonymization features seem to be a constant multiple of a certain number. Extract them about V202.  
# These trends can be seen in some V features such as ['V202', 'V203', 'V204'] and ['V211', 'V212', 'V213'].  

# In[ ]:


num_dict = {}
unique_list = []
for num_1 in tqdm(adjusted_train['V202'].unique()):
    tmp_list = []
    if num_1.is_integer():
        continue
        
    for num_2 in adjusted_train['V202'].unique():
        if num_2.is_integer():
            continue
        if num_1 == num_2:
            continue
        if num_2%num_1 == 0:
            tmp_list.append(num_2)
            num_dict[num_1] = tmp_list
    if num_1 in num_dict:
        pass
    else:
        unique_list.append(num_1)


# In[ ]:


print(len(num_dict))
print(len(unique_list))


# In[ ]:


for idx, (k, v) in enumerate(num_dict.items()):
    if idx >= 10:
        break
    print(k, v)


# # Check Dataset

# In[ ]:


df_list = []
for idx, (k, v) in enumerate(num_dict.items()):
    if idx >= 10:
        break
    tmp_all = pd.DataFrame(columns=adjusted_train.columns)
    for idx_2, v_val in enumerate(v):
        if idx_2 == 0:
            tmp = adjusted_train[adjusted_train['V202']==k]
            tmp_all = pd.concat([tmp_all, tmp], sort=False)            
            tmp = adjusted_train[adjusted_train['V202']==v_val]
            tmp_all = pd.concat([tmp_all, tmp], sort=False)
        else:
            tmp = adjusted_train[adjusted_train['V202']==v_val]
            tmp_all = pd.concat([tmp_all, tmp], sort=False)
    tmp_all = tmp_all.sort_values(by='TransactionDT')
    df_list.append(tmp_all)


# In[ ]:


print(df_list[0]['V202'].values)
df_list[0]


# In[ ]:


df_list[0][['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'V202', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']]


# In[ ]:


print(df_list[1]['V202'].values)
df_list[1]
# The relationship of TransactionAmt can also be seen from this data set.


# In[ ]:


df_list[1][['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'V202', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']]


# # Other type
# It seems that there is also diagonal data. Is it a number assigned by the operation of the system??  
# Or it looks like the user's account steal by fraud user.  

# In[ ]:


adjusted_train[adjusted_train.index.isin([388104, 388121, 418229])][['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'V202','V211', 'V273', 'V306', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']]


# In[ ]:


print(143.255798 - 69.301697)
print((143.255798 - 69.301697)*8)


# # Next Step
# * Are these the same user?  
# * Some V features may be mechanically calculated??
# * TransactionAmt also seems to be continuous. What does this mean?
# * Some features also have a diagonally offset relationship.
# * Find patterns more efficiently.
# 
# I think that they represent the same user or contain artificial data. Also, I think that the anonymization feature has already been calculated between features.  
# I look forward to your advice.  
