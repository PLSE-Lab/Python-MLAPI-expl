#!/usr/bin/env python
# coding: utf-8

# In[26]:


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


# # Objective of the competition is to "predict their clients' repayment abilities." There are quite a number of files provided. For this notebook, let's start with the main file and explore the variables in that
# Summary of this notebook:
# 1. Load the main data (train) file
# 2. Observe the input variables (Features) and output variable available in the dataset
# 3. Do Univariate and Bivaraite analysis on these features
# 4. Create Simple baseline model
# 5. Create a template for submitting predictions
# 

# # 1. Load the main data (train) file

# In[27]:


train_df = pd.read_csv('../input/application_train.csv')


# In[28]:


train_df.shape


# In[29]:


#there are 307K rows and 122 columns; the number of cols is very high


# In[30]:


train_df.head()


# In[31]:


# we see that TARGET is the output variable and also that we have a number of demographic variables


# In[32]:


#let's explore the output variable


# In[33]:


train_df.TARGET.value_counts()


# In[34]:


# low event rate (as we would expect)
print("event rate is : {} %".format(round((train_df.TARGET.value_counts()[1]/train_df.shape[0]) * 100)))


# In[35]:


#the file homecredit_columns_description has details 


# In[36]:


train_df.NAME_CONTRACT_TYPE.value_counts()


# In[37]:


#there are a number of numeric and categorical features; for the baseline model in this notebook, let's take only a few
#also, for the categorical features, we will try to do mean encoding instead of the regular one hot encoding / label encoding


# #numeric features we are interested in:
# AMT_INCOME_TOTAL
# AMT_CREDIT
# AMT_ANNUITY
# AMT_GOODS_PRICE
# DAYS_EMPLOYED
# 

# In[38]:


num_features_list = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_EMPLOYED']


# # categorical features:
# NAME_CONTRACT_TYPE
# CODE_GENDER
# FLAG_OWN_CAR
# FLAG_OWN_REALTY
# NAME_INCOME_TYPE
# NAME_EDUCATION_TYPE
# NAME_FAMILY_STATUS
# OCCUPATION_TYPE
# 
# 

# In[39]:


cat_features_list = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                     'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'OCCUPATION_TYPE']


# In[40]:


#mean encoding for cat features


# In[41]:


num_rows = train_df.shape[0] #get number of records in train

for cat_feature in cat_features_list: #iterate over all the cat features
        encoder_series = train_df[cat_feature].value_counts() / num_rows #create a series that would have the mean for each value in the cat feature
        train_df[cat_feature+'_mean_enc'] = train_df[cat_feature].map(encoder_series) #map that to the specific cat feature and create a new col


# In[43]:


# we have created a set of cols that are mean encoded from categorical cols
#lets move on to create a baseline model with the selected numeric features and the mean encoded cols


# In[44]:


#create a list with numeric cols


# In[45]:


train_df.columns


# In[47]:


features = num_features_list + ['NAME_CONTRACT_TYPE_mean_enc', 'CODE_GENDER_mean_enc', 'FLAG_OWN_CAR_mean_enc', 'FLAG_OWN_REALTY_mean_enc',
                               'NAME_INCOME_TYPE_mean_enc', 'NAME_EDUCATION_TYPE_mean_enc',
                               'NAME_FAMILY_STATUS_mean_enc', 'OCCUPATION_TYPE_mean_enc']


# In[48]:


X_train = train_df[features]


# In[50]:


y_train = train_df.TARGET


# In[51]:


from xgboost import XGBClassifier


# In[52]:


seed = 111


# In[69]:


#without scale pos weight, we had no 1 preds; with scale pos weight as 12, we had a 127K 1s with accuracy of 60%


# In[70]:


model_xgb = XGBClassifier(scale_pos_weight=6)


# In[71]:


model_xgb.fit(X=X_train, y=y_train)


# In[72]:


np.sum(model_xgb.predict(X_train))


# In[55]:


from sklearn.metrics import accuracy_score


# In[73]:


accuracy_score(y_true=y_train, y_pred=model_xgb.predict(data=X_train))


# In[74]:


from sklearn.metrics import confusion_matrix


# In[75]:


confusion_matrix(y_true=y_train, y_pred=model_xgb.predict(data=X_train))


# In[57]:


#now to make predictions on the test set
#first, we need to do the mean encoding for the test data as well


# In[58]:


test_df = pd.read_csv('../input/application_test.csv')


# In[59]:


for cat_feature in cat_features_list: #iterate over all the cat features
        test_df[cat_feature+'_mean_enc'] = test_df[cat_feature].map(encoder_series) #map that to the specific cat feature and create a new col


# In[60]:


X_test = test_df[features]


# In[76]:


y_pred_test = model_xgb.predict(X_test)


# In[77]:


np.sum(y_pred_test)


# In[78]:


#no 1s are predicted; FAIL


# In[79]:


#do submission


# In[80]:


y_pred_test_prob = model_xgb.predict_proba(X_test)[:, 1]


Submission = pd.DataFrame({ 'SK_ID_CURR': test_df.SK_ID_CURR,'TARGET': y_pred_test_prob })
Submission.to_csv("sample_submission_baseline_23May18.csv", index=False)


# In[ ]:




