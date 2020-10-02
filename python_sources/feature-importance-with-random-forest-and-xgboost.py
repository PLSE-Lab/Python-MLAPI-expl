#!/usr/bin/env python
# coding: utf-8

# # Feature impotance testing with Random Forest and XGBoost
# Goal of notebook is glimspe about feature importance.
# ### Reference Notebook
# Inspriration from other kernel
# * https://www.kaggle.com/codename007/home-credit-complete-eda-feature-importance
# * https://www.kaggle.com/nanomathias/feature-engineering-importance-testing

# In[3]:


import numpy as np 
import pandas as pd 
import os

print(os.listdir("../input"))


# ## Retreived the data
# 

# In[6]:


application_train = pd.read_csv('../input/application_train.csv')
application_test = pd.read_csv('../input/application_test.csv')
POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
previous_application = pd.read_csv('../input/previous_application.csv')
installments_payments =  pd.read_csv('../input/installments_payments.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
bureau  = pd.read_csv('../input/bureau.csv')

print('Size of application_train : ' + str(application_train.shape))
print('Size of application_test : ' + str(application_test.shape))
print('Size of POS_CASH_balance : ' + str(POS_CASH_balance.shape))
print('Size of bureau_balance : ' + str(bureau_balance.shape))
print('Size of previous_application : ' + str(previous_application.shape))
print('Size of installments_payments : ' + str(installments_payments.shape))
print('Size of credit_card_balance : ' + str(credit_card_balance.shape))
print('Size of bureau : ' + str(bureau.shape))


# ## Glimpse the data

# In[17]:


print('Size of application_train : ' + str(application_train.shape))
application_train.head()


# In[125]:


print('Size of application_test : ' + str(application_test.shape))
application_test.head()


# In[19]:


print('Size of POS_CASH_balance : ' + str(POS_CASH_balance.shape))
POS_CASH_balance.head()


# In[20]:


print('Size of bureau_balance : ' + str(bureau_balance.shape))

bureau_balance.head()


# In[21]:


print('Size of previous_application : ' + str(previous_application.shape))

previous_application.head()


# In[22]:


print('Size of installments_payments : ' + str(installments_payments.shape))

installments_payments.head()


# In[23]:


print('Size of credit_card_balance : ' + str(credit_card_balance.shape))

credit_card_balance.head()


# In[24]:


print('Size of bureau : ' + str(bureau.shape))
bureau.head()


# ## Checking missing data
# 

# In[31]:


# checking missing data in application_train

number = application_train.isnull().sum().sort_values(ascending = False)
percent = (application_train.isnull().sum() / application_train.isnull().count() * 100).sort_values(ascending = False)

missing_application_train = pd.concat([number , percent] , axis = 1 , keys = ['Total' , 'Percent'])
print(missing_application_train.shape)
print(missing_application_train.head(10))


# In[32]:


# checking missing data in application_test

number = application_test.isnull().sum().sort_values(ascending = False)
percent = (application_test.isnull().sum() / application_test.isnull().count() * 100).sort_values(ascending = False)

missing_application_test = pd.concat([number , percent] , axis = 1 , keys = ['Total' , 'Percent'])
print(missing_application_test.shape)
print(missing_application_test.head(10))


# ## Feature important using Random Forest

# In[36]:


# Preprocessing data

from sklearn import preprocessing
categorical_features = [f for f in application_train.columns if application_train[f].dtype == 'object']

for col in categorical_features:
    lb = preprocessing.LabelEncoder()
    lb.fit(list(application_train[col].values.astype('str')) + list(application_test[col].values.astype('str')))
    
    application_train[col] = lb.transform(list(application_train[col].values.astype('str')))
    application_test[col] = lb.transform(list(application_test[col].values.astype('str')))
    
    


# In[37]:


application_train.fillna(-1 , inplace = True)


# In[42]:


from sklearn.ensemble import RandomForestClassifier

X_train = application_train.drop(['SK_ID_CURR' , 'TARGET'] , axis = 1)
Y_train = application_train['TARGET']

rdf = RandomForestClassifier(n_estimators = 100 , max_depth = 5, min_samples_leaf = 4, max_features = 0.5)
rdf.fit(X_train , Y_train)


# In[71]:


features = application_train.drop(['SK_ID_CURR' , 'TARGET'] , axis = 1).columns
features_impt_rdf = pd.DataFrame(rdf.feature_importances_, columns = ['SCORE'])
features_impt_rdf['FEATURE'] = features 
print(features_impt_rdf.sort_values('SCORE',ascending = False).head(20))


# In[92]:


# Training with XGboost
import xgboost as xgb

X_train_xgb = application_train.drop(['SK_ID_CURR' , 'TARGET'] , axis = 1).select_dtypes(include=[np.number])

clf_xgBoost = xgb.XGBClassifier(
    max_depth = 4,
    subsample = 0.8,
    colsample_bytree = 0.7,
    colsample_bylevel = 0.7,
    scale_pos_weight = 9,
    min_child_weight = 0,
    reg_alpha = 4,
    n_jobs = 4, 
    objective = 'binary:logistic'
)
# Fit the models
clf_xgBoost.fit(X_train_xgb, Y_train)


# In[124]:


from sklearn import preprocessing

importance_df = clf_xgBoost.get_booster().get_score(importance_type='weight')

features_impt_xgb = pd.DataFrame(list(importance_df.items()), columns = ['FEATURE' , 'SCORE'])

print(features_impt_xgb.sort_values('SCORE', ascending = False).head(20))


# In[ ]:




