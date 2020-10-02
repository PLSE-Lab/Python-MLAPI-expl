#!/usr/bin/env python
# coding: utf-8

# installing fastai

# In[ ]:


pip install fastai==0.7.0


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

from fastai.imports import *
from fastai.structured import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv("../input/predict-the-churn-for-customer-dataset/Train File.csv")
test=pd.read_csv("../input/predict-the-churn-for-customer-dataset/Test File.csv")


# In[ ]:


train.head()


# In[ ]:


train.dtypes


# In[ ]:


for i in train.columns:
    print(i)
    print(train[i].unique())


# In[ ]:


train['TotalCharges'].fillna((train['TotalCharges'].median()), inplace=True)


# In[ ]:


test['TotalCharges'].fillna((test['TotalCharges'].median()), inplace=True)


# In[ ]:


#reducing the categories
train[['tenure']]=train[['tenure']]/10
train[['tenure']]=train['tenure'].astype(int)
test[['tenure']]=test[['tenure']]/10
test[['tenure']]=test['tenure'].astype(int)

train[['MonthlyCharges']]=train[['MonthlyCharges']]/10
train[['MonthlyCharges']]=train['MonthlyCharges'].astype(int)
test[['MonthlyCharges']]=test[['MonthlyCharges']]/10
test[['MonthlyCharges']]=test['MonthlyCharges'].astype(int)

train[['TotalCharges']]=train[['TotalCharges']]/1000
train[['TotalCharges']]=train['TotalCharges'].astype(int)
test[['TotalCharges']]=test[['TotalCharges']]/1000
test[['TotalCharges']]=test['TotalCharges'].astype(int)


# In[ ]:


train


# In[ ]:


test2=pd.DataFrame()
test2['customerID']=test['customerID']


# In[ ]:


#dropping id column
train=train.drop(['customerID'],axis=1)
test=test.drop(['customerID'],axis=1)


# In[ ]:


test2


# In[ ]:


#converting pandas dataframe category
train_cats(train)


# In[ ]:


train.dtypes


# In[ ]:


#converting to numerical values and imputing missing values with mean ## "proc_df" does things in fastai
df, y, nas = proc_df(train, 'Churn')


# In[ ]:


for i in df.columns:
    print(i)
    print(df[i].unique())


# In[ ]:


df


# In[ ]:


from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score
from xgboost import XGBClassifier
import xgboost as xgb


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split( df, y, test_size=0.1, random_state=42)


# In[ ]:


X_train


# In[ ]:


set_rf_samples(2000)


# In[ ]:


m=RandomForestClassifier(n_estimators=400, n_jobs=-1, min_samples_leaf=5,max_depth=10)
m.fit(X_train, y_train)
y_pred= m.predict(X_val)
print(y_pred)
accuracy_score(y_val,y_pred)


# In[ ]:


train_cats(test)
apply_cats(df=test, trn=train)


# In[ ]:


test


# In[ ]:


X_test,_,nas = proc_df(test, na_dict=nas)
X, y , nas = proc_df(train, 'Churn', na_dict=nas)


# In[ ]:


X_test


# In[ ]:


prediction =m.predict(X_test)


# In[ ]:


prediction


# In[ ]:


submission = pd.DataFrame()


# In[ ]:


X_test


# In[ ]:


submission['customerID']=test2.customerID


# In[ ]:


submission['Churn']=prediction


# In[ ]:


submission


# In[ ]:


submission['Churn']=submission['Churn'].astype(str)


# In[ ]:


submission['Churn'].replace(('1','0'),('Yes', 'No'), inplace=True)


# In[ ]:


sub=submission.to_csv("submission7.csv",index=False)


# In[ ]:




