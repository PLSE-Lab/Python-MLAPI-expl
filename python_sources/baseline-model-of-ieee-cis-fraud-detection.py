#!/usr/bin/env python
# coding: utf-8

# Hello, This Kernel is the first kernel I write in Kaggle.
# 
# This competetion is the unbalance binary classification competition, which is common in Kaggle.
# 
# What is unique is that Train / Test Data is provided in two separated files, transaction and identify, both are linked by 'TransactionID' value.

# ### OK. Let's start the kernel by loading the basic packages.

# In[ ]:


import numpy as np
import pandas as pd
import os

os.listdir("../input")

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# #### I will check if the necessary data set files are located correctly.
# #### 2 train data sets, 2 test data sets and lastly data for submission.
# #### All data sets are OK.

# * As mentioned earlier, this competition is divided into train and test data set into transaction and identity data sets.
# * Therefore, we need to combine these two data sets into one.
# * The 'merge_train_data' is a function that loads two files of Train Data Set and merges identity based on transaction data set.

# In[ ]:


def merge_train_data():    
    print( "Loading Train Data...")
    train_transaction = pd.read_csv('../input/train_transaction.csv')
    train_identity = pd.read_csv('../input/train_identity.csv')
    
    print( "Shape of train_transaction" , train_transaction.shape )
    train_transaction.head()
    
    print("Shape of train_identity" , train_identity.shape )
    train_identity.head()
    
    print( "Merging..." )
    train_merged = pd.merge( train_transaction , train_identity , on='TransactionID' , how='left' )
    
    del train_transaction 
    del train_identity
    
    return train_merged


# In[ ]:


train_merged = merge_train_data()
train_merged.shape
train_merged.head()


# In[ ]:





# #### Transaction data has about 590,000 data, 394 features, while identity data has about 140,000 data, and 41 features.
# 
# #### As we compare the shape of these two data sets, when we merge identity Data based on transaction data, merged data will include many NaN data.

# In[ ]:


chk_NaN = train_merged.isnull().sum()
chk_NaN


# #### Similar to train data, let's load test data and merge as same ways of train data

# In[ ]:


def merge_test_data():
    print( "Loading Test Data...")
    test_transaction = pd.read_csv('../input/test_transaction.csv')
    test_identity = pd.read_csv('../input/test_identity.csv')
    
    print( "Shape of test_transaction" , test_transaction.shape )
    test_transaction.head()
    
    print("Shape of test_identity" , test_identity.shape )
    test_identity.head()
    
    print( "Merging..." )
    test_merged = pd.merge( test_transaction , test_identity , on='TransactionID' , how='left' )

    del test_transaction
    del test_identity
    return test_merged


# In[ ]:


test_merged = merge_test_data()
test_merged.shape
test_merged.head()


# In[ ]:


chk_NaN = test_merged.isnull().sum()
chk_NaN


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot( x = 'isFraud', data = train_merged )
plt.xticks(rotation=45)
plt.title( 'Target' )
plt.show()


# #### The distribution of target is very unbalanced.

# #### In the train dats set, the target column is removed from the features.

# In[ ]:


target = train_merged['isFraud']
train_merged = train_merged.drop('isFraud' , axis='columns')


# In[ ]:


train_merged.shape


# #### Currently, the total number of features is 433 and I think it seems there are too many features.
# #### So, I decided to remove some features that has little information

# #### First, let's remove the feature with a large number of NaN.

# In[ ]:


rows_count = train_merged.shape[0]

missing_value = pd.DataFrame(columns=['Missing Rate'])
missing_value['Missing Rate'] = train_merged.isnull().sum() / rows_count

high_miss_rate = missing_value[ missing_value['Missing Rate'] >= 0.85 ].index.values.tolist()

print("High Missing Rate(85%) Feature Count : ",len(high_miss_rate))

train_merged = train_merged.drop(high_miss_rate , axis='columns')
train_merged.shape
del high_miss_rate


# #### Total 74 features are consisted of NaN over 85% and will be removed from train data set.
# #### However, since we can do feature engineering later using the ratio of NaN or the meaning of NaN, let's try to decide whether to use it

# #### Let's take a look at each of the remaining features by their data type.
# #### Let's break it down into Int, Float, and Object

# In[ ]:


cols_int = []
cols_float = []
cols_object = []

all_cols_list = train_merged.columns.values.tolist()

for col in all_cols_list:
    if train_merged[col].dtypes == 'int64':
        cols_int.append(col)
    elif train_merged[col].dtypes == 'float64':
        cols_float.append(col)
    elif train_merged[col].dtypes == 'object':
        cols_object.append(col)
    else:
        print('Exception')

print( 'Total train_merged Feature Numbers : ', len(all_cols_list))
print( 'int64 Feature Numbers : ', len(cols_int))
print( 'float64 Feature Numbers : ', len(cols_float))
print( 'object Feature Numbers : ', len(cols_object))


# #### Let's measure the feature importance of the remaining features with XGBoost.

# In[ ]:


import xgboost as xgb
from xgboost import XGBClassifier


# In[ ]:


# fit model no training data
model = XGBClassifier( nthread = 4 )
model.fit( train_merged[cols_int + cols_float] , target)


# In[ ]:


xgb_fea_imp = pd.DataFrame(list(model.get_booster().get_fscore().items()),columns=['feature','importance']).sort_values('importance', ascending=False)
xgb_fea_imp.shape
xgb_fea_imp.head(10)


# #### XGBoost has selected 132 features with Importance.
# #### Now let's train using these features.

# In[ ]:


import lightgbm as lgb

train_ds = lgb.Dataset( train_merged[ xgb_fea_imp['feature'].tolist() ] , label = target )

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_threads' : 4,
    'num_leaves': 31,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 1
}

model = lgb.train(parameters, train_ds ,num_boost_round=5000)


# In[ ]:


prob = model.predict( test_merged[ xgb_fea_imp['feature'].tolist() ] )
prob


# In[ ]:


submission = pd.DataFrame()
submission['TransactionID'] = test_merged['TransactionID']
submission['isFraud'] = prob


# In[ ]:


submission.head()


# #### Not a high score, but it seems appropriate for use as a Baseline Model.

# #### The features I used are all features without special feature engineering
# #### We only used numerical features, and we did not use any other categorical features at all.

# ### In addition, the following applies to the following kernel.
# - Using categorical features, additional Feature Engineering
# - Application of Cross Validation
# - Stacking

# #### I will share with you the next kernel on the methods I will use.
# #### Please let me know if I made a mistake or have a better suggestion.
# 
# ### Thank you for reading.

# In[ ]:




