#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


user_list            = pd.read_csv("../input/purchase-coupons-data/user_list.csv")
coupon_area_train    = pd.read_csv("../input/purchase-coupons-data/coupon_area_train.csv")
coupon_detail_train  = pd.read_csv("../input/purchase-coupons-data/coupon_detail_train.csv")
coupon_list_train    = pd.read_csv("../input/purchase-coupons-data/coupon_list_train.csv")
coupon_visit_train   = pd.read_csv("../input/purchase-coupons-visit/coupon_visit_train.csv")
#prefecture_locations = pd.read_csv("../input/purchase-coupons-data/prefecture_locations.csv")


# In[ ]:


import json

# read file
with open('../input/purchase-coupons-data/translations.json', 'r') as myfile:
    data=myfile.read()

# parse file
translations = json.loads(data)


# In[ ]:


user_list['PREF_NAME'] = user_list['PREF_NAME'].map(translations)
coupon_area_train['PREF_NAME'] = coupon_area_train['PREF_NAME'].map(translations)
coupon_area_train['SMALL_AREA_NAME'] = coupon_area_train['SMALL_AREA_NAME'].map(translations)
coupon_detail_train['SMALL_AREA_NAME'] = coupon_detail_train['SMALL_AREA_NAME'].map(translations)
#prefecture_locations['PREF_NAME'] = prefecture_locations['PREF_NAME'].map(translations)
#prefecture_locations['PREFECTUAL_OFFICE'] = prefecture_locations['PREFECTUAL_OFFICE'].map(translations)


# In[ ]:


coupon_list_english = ['CAPSULE_TEXT','GENRE_NAME','large_area_name','ken_name','ken_name','small_area_name']
coupon_list_train[coupon_list_english] = coupon_list_train[coupon_list_english].apply(lambda x: x.map(translations))


# In[ ]:


user_list.head()


# In[ ]:


coupon_area_train.head()


# In[ ]:


coupon_detail_train.head()


# In[ ]:


coupon_visit_train.count()


# ### Exploratory Analysis

# In[ ]:


coupon_visit_train.count()


# In[ ]:


visit_coupon = pd.merge(coupon_visit_train, coupon_list_train, how = 'inner',left_on="VIEW_COUPON_ID_hash", right_on="COUPON_ID_hash")


# In[ ]:


visit_coupon.count().head()


# In[ ]:


user_visit_coupon = pd.merge(visit_coupon, user_list, how = 'left', left_on="USER_ID_hash", right_on="USER_ID_hash")


# In[ ]:


user_visit_coupon.describe()


# In[ ]:


# Serching null Values 
user_visit_coupon.isnull().sum()


# In[ ]:


#Converting dates to datetime
user_visit_coupon['DISPFROM'] = pd.to_datetime(user_visit_coupon['DISPFROM'], infer_datetime_format=True)
user_visit_coupon['DISPEND'] = pd.to_datetime(user_visit_coupon['DISPEND'], infer_datetime_format=True)
user_visit_coupon['VALIDFROM'] = pd.to_datetime(user_visit_coupon['VALIDFROM'], infer_datetime_format=True)
user_visit_coupon['VALIDEND'] = pd.to_datetime(user_visit_coupon['VALIDEND'], infer_datetime_format=True)
user_visit_coupon['WITHDRAW_DATE'] = pd.to_datetime(user_visit_coupon['WITHDRAW_DATE'], infer_datetime_format=True)
user_visit_coupon['REG_DATE'] = pd.to_datetime(user_visit_coupon['REG_DATE'], infer_datetime_format=True)


# In[ ]:


#Extracting year, month and days in number from dates
user_visit_coupon_DISPFROM_year      = user_visit_coupon['DISPFROM'].dt.year
user_visit_coupon_DISPFROM_month     = user_visit_coupon['DISPFROM'].dt.month
user_visit_coupon_DISPFROM_day       = user_visit_coupon['DISPFROM'].dt.day
user_visit_coupon_DISPEND_year       = user_visit_coupon['DISPEND'].dt.year
user_visit_coupon_DISPEND_month      = user_visit_coupon['DISPEND'].dt.month
user_visit_coupon_DISPEND_day        = user_visit_coupon['DISPEND'].dt.day
user_visit_coupon_VALIDFROM_year     = user_visit_coupon['VALIDFROM'].dt.year
user_visit_coupon_VALIDFROM_month    = user_visit_coupon['VALIDFROM'].dt.month
user_visit_coupon_VALIDFROM_day      = user_visit_coupon['VALIDFROM'].dt.day
user_visit_coupon_VALIDEND_year      = user_visit_coupon['VALIDEND'].dt.year
user_visit_coupon_VALIDEND_month     = user_visit_coupon['VALIDEND'].dt.month
user_visit_coupon_VALIDEND_day       = user_visit_coupon['VALIDEND'].dt.day
user_visit_coupon_REG_DATE_DATE_year   = user_visit_coupon['REG_DATE'].dt.year
user_visit_coupon_REG_DATE_DATE_month  = user_visit_coupon['REG_DATE'].dt.month
user_visit_coupon_REG_DATE_DATE_day    = user_visit_coupon['REG_DATE'].dt.day
user_visit_coupon_WITHDRAW_DATE_year   = user_visit_coupon['WITHDRAW_DATE'].dt.year
user_visit_coupon_WITHDRAW_DATE_month  = user_visit_coupon['WITHDRAW_DATE'].dt.month
user_visit_coupon_WITHDRAW_DATE_day    = user_visit_coupon['WITHDRAW_DATE'].dt.day


# In[ ]:


user_visit_coupon = user_visit_coupon.drop(['DISPFROM', 'DISPEND', 'VALIDFROM', 'VALIDEND', 'WITHDRAW_DATE','REG_DATE'],axis = 1)


# In[ ]:


#Replacing null values 
numerical_columns = user_visit_coupon.select_dtypes(np.number).columns
user_visit_coupon[numerical_columns] = user_visit_coupon[numerical_columns].fillna(0)
user_visit_coupon['PREF_NAME'] =  user_visit_coupon['PREF_NAME'].fillna('unknown')
user_visit_coupon.isnull().sum()


# In[ ]:


user_visit_coupon.AGE.unique()


# In[ ]:


user_visit_coupon.AGE.min(),user_visit_coupon.AGE.max()


# In[ ]:


#We put the age column in buckets to better visualize the ranges 
bins = [15, 30, 40, 50, 60,70, 80]
labels = ['15-29', '30-39', '40-49', '50-59', '60-69', '70-80']
user_visit_coupon['agelabel'] = pd.cut(user_visit_coupon.AGE , bins, labels = labels,include_lowest = True)


# In[ ]:


#Chart to identify the age ranges where coupons are used
PURCHASE_POSITIVE = user_visit_coupon[user_visit_coupon['PURCHASE_FLG']==1]
gender_age = PURCHASE_POSITIVE.groupby(['SEX_ID','agelabel']).size()
gender_age.plot.bar()


# In[ ]:


#Identify in which category of coupons there is more use of coupons
pd.crosstab(user_visit_coupon['GENRE_NAME'],user_visit_coupon['PURCHASE_FLG']).plot.bar()


# In[ ]:


#Trying to see the largest number of purchases according to the PREF_NAME
pd.crosstab(user_visit_coupon['PREF_NAME'],user_visit_coupon['PURCHASE_FLG']).plot.bar()


# In[ ]:


#Percentage of Actual Discount on Price
user_visit_coupon['PERCENTAGE_DISCOUNT'] = 100 - user_visit_coupon['PRICE_RATE']


# In[ ]:


user_visit_coupon[['PRICE_RATE','PERCENTAGE_DISCOUNT','CATALOG_PRICE','DISCOUNT_PRICE']].head()


# In[ ]:


user_visit_coupon  = user_visit_coupon.drop(['I_DATE', 'PAGE_SERIAL', 'REFERRER_hash',
       'VIEW_COUPON_ID_hash', 'USER_ID_hash', 'SESSION_ID_hash',
       'PURCHASEID_hash','CAPSULE_TEXT','PRICE_RATE','AGE'],axis=1)


# In[ ]:


#I don't know exactly what 2 means in those kind of features...
#Let's take the value 2 like - Available with restrictions related with the location according to kaggle discussions
user_visit_coupon.USABLE_DATE_FRI.unique()


# **Feature engineering**

# In[ ]:


### convert categoricals to OneHotEncoder form
categoricals = ['GENRE_NAME', 'large_area_name', 'ken_name', 'small_area_name', 'agelabel','SEX_ID','PREF_NAME']
combined_categoricals = user_visit_coupon[categoricals]
combined_categoricals = pd.get_dummies(combined_categoricals,dummy_na=False)


# In[ ]:


user_visit_coupon = user_visit_coupon.drop(['GENRE_NAME', 'large_area_name', 'ken_name', 'small_area_name', 'agelabel','SEX_ID','PREF_NAME','COUPON_ID_hash'],axis = 1)


# In[ ]:


#Putting together the categorical columns
user_visit_coupon = pd.concat([user_visit_coupon,combined_categoricals], axis=1)


# In[ ]:


user_visit_coupon.head()


# In[ ]:


#Know if binary values are balanced
user_visit_coupon.groupby(['PURCHASE_FLG']).size()


# In[ ]:


#With this graph we can see we have an imbalanced dataset  
user_visit_coupon.groupby(['PURCHASE_FLG']).size().plot.bar()


# In[ ]:


# Separate input features and target
y = user_visit_coupon.PURCHASE_FLG
X = user_visit_coupon.drop('PURCHASE_FLG', axis=1)


# In[ ]:


user_visit_coupon.shape


# In[ ]:


# Testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=27)


# **Downsampling the data because the imbalance of the data**

# In[ ]:


# Data back together
X = pd.concat([X_train, y_train], axis=1)


# In[ ]:


# Separate minority and majority 
not_purchase = X[X.PURCHASE_FLG==0]
purchase = X[X.PURCHASE_FLG==1]


# In[ ]:


# Downsampled majority
purchase_downsampled = resample(not_purchase,
                          replace=False, # sample with replacement
                          n_samples=len(purchase), # match number in majority class
                          random_state=48) # reproducible results


# In[ ]:


downsampled = pd.concat([purchase, purchase_downsampled])
downsampled.PURCHASE_FLG.value_counts()


# In[ ]:


downsampled.columns


# In[ ]:


y_train = downsampled.PURCHASE_FLG
X_train = downsampled.drop('PURCHASE_FLG', axis=1)


# **Training the models **

# In[ ]:


#Logistic Regression model
model_LR = LogisticRegression(solver='lbfgs')
model_LR.fit(X_train, y_train)


# In[ ]:


pred_LR = model_LR.predict(X_test)


# In[ ]:


#Evaluating the Algorithm
# Checking accuracy
from sklearn.metrics import accuracy_score,f1_score, recall_score
accuracy_score(y_test, pred_LR)


# In[ ]:


# f1 score
f1_score(y_test, pred_LR)


# In[ ]:


#Recall
recall_score(y_test, pred_LR)


# In[ ]:


pip install joblib


# In[ ]:


##RandomForest Classifier  model
import joblib
RF = RandomForestClassifier(n_estimators=60, max_depth=2, random_state=27)
RF.fit(X_train, y_train)
# save model
joblib.dump(RF, 'model.pkl')


# In[ ]:


y_pred_rf = RF.predict(X_test)


# In[ ]:


#Evaluating the Algorithm
# Checking accuracy
accuracy_score(y_test, y_pred_rf)


# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred_rf))
print(classification_report(y_test,y_pred_rf))


# In[ ]:


y_test.value_counts()


# In[ ]:




